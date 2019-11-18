import configs.config as cfg
from berttest.header import *


class Attention(nn.Module):

    def __init__(self, isMask=False):
        super().__init__()
        self.dk = (cfg.embed_dim // cfg.head_num) ** 0.5
        self.isMask = isMask

        self.c_attn = nn.Linear(cfg.embed_dim, cfg.embed_dim * 3)

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
            # print("mask",mask.shape)
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


class Block(nn.Module):

    def __init__(self, isMask=False):
        super().__init__()

        self.layer_normal_1 = nn.LayerNorm(cfg.embed_dim)

        self.attention = Attention(isMask)

        self.layer_normal_2 = nn.LayerNorm(cfg.embed_dim)

        self.proj = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.multi * cfg.embed_dim),
            nn.LeakyReLU(),
            nn.Linear(cfg.multi * cfg.embed_dim, cfg.embed_dim),
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        h = self.layer_normal_1(x)  # torch.Size([3, 768])
        # print("1", h.size())
        a = self.attention(h)  # torch.Size([3, 768])
        # print("2", a.size())
        a = a + x  # torch.Size([3, 768])
        # print("3", a.size())
        a = self.layer_normal_2(a)  # torch.Size([3, 768])
        # print("4", a.size())
        h = self.proj(a)  # torch.Size([3, 768])
        # print("5", h.size())
        h = self.dropout(h)  # torch.Size([3, 768])
        # print("6", h.size())
        y = h + a  # torch.Size([3, 768])
        # print("7", y.size())
        return y


class GPT2(nn.Module):

    def __init__(self):
        super().__init__()

        self.vocab_embed = nn.Embedding(cfg.vocab_num, cfg.embed_dim)
        self.pos_embed = nn.Embedding(cfg.pos_num, cfg.embed_dim)
        self.type_embed = nn.Embedding(cfg.type_num, cfg.embed_dim)

        self.blocks = []
        for _ in range(cfg.block_num):
            self.blocks.append(Block())

        self.drop = nn.Dropout(0.1)

        self.sequential = nn.Sequential(*self.blocks)

        self.output_layer = nn.Linear(cfg.embed_dim, cfg.vocab_num, bias=False)

    def forward(self, x, p, t=torch.Tensor([1, ]).long().cuda()):
        e = self.vocab_embed(x)
        p = self.pos_embed(p)
        t = self.type_embed(t)
        h = self.drop(e + p + t)
        h = self.sequential(h)
        return self.output_layer(h)


if __name__ == '__main__':
    # x = torch.randint(0, 4000, size=(2, cfg.pos_num)).to(torch.device(cfg.device))
    # p = torch.arange(0, cfg.pos_num)[None, :].repeat(x.shape[0], 1).to(torch.device(cfg.device))
    #
    # gpt = GPT2()
    # gpt.to(torch.device(cfg.device))
    # gpt.eval()
    # # gpt.load_state_dict(torch.load("weights/apt2_k.pt"))
    # # y = gpt(x, p)
    # # print(y.shape)
    # #
    # # print(sum(param.numel() for param in block.parameters()))
    #
    # os = []
    # x = torch.tensor([[0]]).cuda()
    # p = torch.tensor([[0]]).cuda()
    # for i in range(100):
    #     y = gpt(x, p)
    #     y = y[:, -1:]
    #     v, y = torch.topk(y, 8, dim=-1)
    #
    #     v, y = v.reshape(-1, 8), y.reshape(-1, 8)
    #     v = torch.multinomial(torch.softmax(v, dim=-1), 1)
    #     y = torch.gather(y, -1, v)
    #
    #     x = torch.cat([x, y], dim=1)
    #     p = torch.tensor([range(i + 2)]).cuda()
    #
    #     print(x)

    att = Attention(isMask=True)
    a = torch.randn(3, 768)
    b = att(a)
    print(b.size())
    block = Block(isMask=True)
    c = block(b)
    print(c.shape)
    gpt2 = GPT2()
    # print(a[0][:10])
    # print(b[0][:10])
