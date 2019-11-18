import configs.config as cfg
from berttest.header import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Attention(nn.Module):

    def __init__(self, isMask=True):
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

        # print("x1", x.shape)#[1, 3, 768]
        x = self.c_attn(x)  # [1, 3, 2304]
        # print("x2", x.shape, *x.shape[:-1])
        x = x.reshape(*x.shape[:-1], cfg.head_num, -1)  # [1, 3, 12, 192]
        # print("x3", x.shape)
        x = x.transpose(-2, -3)  # [1, 12, 3, 192]
        # print("x4", x.shape)
        q, k, v = x.chunk(3, dim=-1)  # [1, 12, 3, 64]
        # print("qkv", q.shape, k.shape, v.shape)
        w = (q @ k.transpose(-1, -2)) / self.dk  # [1, 12, 3, 3]
        # print("w", w.shape, self.dk, self.c_attn.weight.shape)
        if self.isMask:
            # print("self.mask", self.mask.shape, w.size(), w.size(-2), w.size(-1))
            mask = self.mask[0:w.size(-2), 0:w.size(-1)]  # [3, 3]
            # print("mask", mask.shape)
            w = w * mask - (1 - mask) * 1e5  # [1, 12, 3, 3]
            # print("w1", w.shape, self.dk, self.c_attn.weight.shape)
            # print(w)
        w = torch.softmax(w, dim=-1)
        # print(w)
        w = self.attn_drop(w)
        # print("w2",w)

        a = w @ v
        # print("a",a.shape)#[1, 12, 3, 64]
        a = a.transpose(-2, -3)  # [1, 3, 12, 64]
        # print("a0", a.shape)
        a = a.reshape(*a.shape[:-2], cfg.embed_dim)  # [1, 3, 768]
        # print("a1",a.shape)

        h = self.c_proj(a)
        h = self.resi_drop(h)  # [1, 3, 768]
        # print("h",h.shape)

        return h


class Block(nn.Module):

    def __init__(self, isMask=True):
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
        # print("x_2", x.shape)#[1, 3, 768]

        h = self.layer_normal_1(x)  # [1, 3, 768]
        # print("h0", h.shape)
        a = self.attention(h)  # [1, 3, 768]
        # print("a", a.shape)
        a = a + x

        # print("a1", a.shape)
        a = self.layer_normal_2(a)  # [1, 3, 768]
        # print("a2", a.shape)
        h = self.proj(a)  # [1, 3, 768]
        # print("h1", h.shape)
        h = self.dropout(h)

        y = h + a  # [1, 3, 768]
        # print("h2", h.shape)
        return y


class Gpt2(nn.Module):

    def __init__(self, isMask=True):
        super().__init__()

        self.vocab_embed = nn.Embedding(cfg.vocab_num, cfg.embed_dim)  # [4413, 768]
        self.pos_embed = nn.Embedding(cfg.pos_num, cfg.embed_dim)  # [500, 768]
        # self.type_embed = nn.Embedding(cfg.type_num, cfg.embed_dim)  # [30, 768]

        self.blocks = []
        for _ in range(cfg.block_num):
            self.blocks.append(Block(isMask))  # 6
        # print("self.blocks", len(self.blocks), self.type_embed.weight.shape)

        self.drop = nn.Dropout(0.1)

        self.sequential = nn.Sequential(*self.blocks)
        # for n,p in self.sequential.named_parameters():
        #     print(n)

        self.output_layer = nn.Linear(cfg.embed_dim, cfg.vocab_num, bias=False)  # [4413, 768]
        # print(self.output_layer.weight.shape)

    def forward(self, x, p):
        # print("x_3", x.shape)#[1, 3]
        e = self.vocab_embed(x)  # [1, 3, 768]
        p = self.pos_embed(p)  # [1, 3, 768]
        # t = self.type_embed(t)#[1, 3, 768]
        # print("ept", x.shape, e.shape, p.shape, self.vocab_embed.weight.shape)
        # a = e+p
        # print(a.shape)
        # b = self.drop(a)
        h = self.drop(e + p)  # [1, 3, 768]
        # print("h0",h.shape)
        h = self.sequential(h)  # [1, 3, 768]
        # print("h1", h.shape)
        # print("output",self.output_layer(h).shape) #[1, 3, 4413]
        return self.output_layer(h)


if __name__ == '__main__':
    # x = torch.randint(0, 4000, size=(2, cfg.pos_num)).to(torch.device(cfg.device))
    # p = torch.arange(0, cfg.pos_num)[None, :].repeat(x.shape[0], 1).to(torch.device(cfg.device))
    #
    gpt = Gpt2()
    # gpt.to(torch.device(cfg.device))
    gpt.cuda()
    gpt.eval()
    # gpt.load_state_dict(torch.load("weights/apt2_k.pt"))
    # y = gpt(x, p)
    # print(y.shape)
    #
    # print(sum(param.numel() for param in block.parameters()))

    os = []
    x = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]).cuda()
    print(x)
    p = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]).cuda()

    # x = torch.randint(0, 4434, size=(1, 500)).cuda()
    # x = torch.randint(0, 4434, size=(1, 50)).cuda()
    #
    # p = torch.arange(50)[None, :].long().cuda()
    # x = torch.randint(0, 4434, size=(1, 500)).cuda()
    # x = torch.randint(0, 4434, size=(1, 500)).cuda()
    # p = torch.arange(500)[None,:].repeat(1,1).long().cuda()
    print(x.shape, p.shape, x.dtype, p.dtype, "00000000000000")
    # print(x)
    # t = torch.tensor([[0]]).cuda()  # 改

    for i in range(100):
        print("x0", x.shape)  # [1, 3]
        y = gpt(x, p)  # [1, 3, 4413]
        print("gpt", y.shape)
        y = y[:, -1:]  # [1, 1, 4413]
        # y = y[:, -1] # [1, 4413]
        print("gpt1", y.shape)
        v, y = torch.topk(y, 8, dim=-1)  # [1, 1, 8]
        print("vy", v.shape, y.shape, v, y)

        v, y = v.reshape(-1, 8), y.reshape(-1, 8)  # [1, 8]
        print("vy1", v.shape, y.shape)
        print(v)
        print(y)
        v = torch.multinomial(torch.softmax(v, dim=-1), 1)  # [1, 1]
        print("v", v.shape, y.shape, v, y)
        y = torch.gather(y, -1, v)  # [1, 1]
        print("y", y.shape, y, x)

        x = torch.cat([x, y], dim=1)  # [1, 4]
        print("x", x.shape, x)
        # p = torch.tensor([range(i + 2)]).cuda() #[1, 1]
        p = torch.arange(0, x.size(-1)).cuda()
        print("p", p.shape, p)
        print("range", range(i + 2))
        break
        # if i > 1:
        #     break
