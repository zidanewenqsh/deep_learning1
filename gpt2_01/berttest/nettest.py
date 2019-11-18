import configs.config as cfg
from berttest.header import *
class Attention(nn.Module):

    def __init__(self, isMask=False):
        super().__init__()
        self.dk = (cfg.embed_dim // cfg.head_num) ** 0.5
        print(self.dk)
        self.isMask = isMask

        self.c_attn = nn.Linear(cfg.embed_dim, cfg.embed_dim * 3)
        print("c_attn",self.c_attn.weight.size())

        self.attn_drop = nn.Dropout(0.1)
        self.resi_drop = nn.Dropout(0.1)

        self.c_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)

        if self.isMask:
            self.register_buffer("mask", torch.tril(torch.ones(cfg.pos_num, cfg.pos_num)))

    def forward(self, x):
        x = self.c_attn(x)  # torch.Size([3, 2304])
        # print("1", x.shape)
        # print(x.shape[:-1])  # N
        x = x.reshape(*x.shape[:-1], cfg.head_num, -1)  # torch.Size([3, 12, 192])
        # print(x.shape[:-1])
        # print(*x.shape[:-1])
        # print("2", x.shape)
        x = x.transpose(-2, -3)  # torch.Size([12, 3, 192])
        # print("3", x.shape)
        q, k, v = x.chunk(3, dim=-1)  # torch.Size([12, 3, 64]
        # print("4", q.size(), k.size(), v.size())
        # print('6', q.shape,k.shape)
        w = (q @ k.transpose(-1, -2)) / self.dk  # torch.Size([12, 3, 3])
        # print("5", w.size())

        if self.isMask:
            mask = self.mask[0:w.size(-2), 0:w.size(-1)]  # torch.Size([3, 3])
            print("mask",mask.shape)
            # print(mask)
            # print(w[0])
            w = w * mask - (1 - mask) * 1e5
            # print(1-mask)
            # print(w[0])
            w = torch.softmax(w, dim=-1)
            # print("7",w[0])
            w = self.attn_drop(w)
            # print(self.attn_drop.parameters())
            # for x in self.attn_drop.parameters():
            #     print(x)
            # print("8", w[0])

        a = w @ v  # torch.Size([12, 3, 64]) torch.Size([12, 3, 3]) torch.Size([12, 3, 64])
        # print(a.shape,w.shape,v.shape)

        a = a.transpose(-2, -3)  # torch.Size([3, 12, 64])
        # print("1",a.shape)
        a = a.reshape(*a.shape[:-2], cfg.embed_dim)  # torch.Size([3, 768])
        # print("2", a.shape)

        h = self.c_proj(a)  # torch.Size([3, 768])
        # print("3", h.shape)
        h = self.resi_drop(h)  # torch.Size([3, 768])
        # print("4", h.shape)

        return h

if __name__ == '__main__':
    net = Attention(isMask=True)
    a = torch.randn(2, 2, cfg.embed_dim)
    # a = torch.randn(10, 500)
    b = net(a)
    print(b.size())