import torch
import torch.nn as nn
from tools.utils import *
from configs.config import *


class Attention(nn.Module):
    def __init__(self, isMask=True):
        # self.device = device
        self.isMask = isMask
        super(Attention, self).__init__()
        self.dk = (embed_dim // head_num) ** 2
        self.c_atten = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(0.1)
        self.resi_drop = nn.Dropout(0.1)
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        if self.isMask:
            self.register_buffer("mask", torch.tril(torch.ones(pos_num, pos_num)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.c_atten(input)
        x = x.reshape(*x.shape[:-1], head_num, -1)
        x = x.transpose(-2, -3)
        q, k, v = x.chunk(3, dim=-1)
        w = q @ k.transpose(-1, -2)
        if self.isMask:
            # mask = creatMask(pos_num)[:, :, w.size(-2), w.size(-1)].cuda()
            mask = self.mask[0:w.size(-2), 0:w.size(-1)]  # [3, 3]
            w = w * mask + (1 - mask) * 1e5

        w = torch.softmax(w, dim=1)
        # print("w",w.shape)
        w = self.attn_drop(w)
        a = w @ v
        a = a.transpose(-2, -3)
        a = a.reshape(*a.shape[:-2], -1)

        h = self.c_proj(a)
        h = self.resi_drop(h)
        return h


class Block(nn.Module):
    def __init__(self,isMask=True):
        super(Block, self).__init__()
        self.layer_normal_1 = nn.LayerNorm(embed_dim)
        self.attention = Attention(isMask)
        self.layer_normal_2 = nn.LayerNorm(embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, multi*embed_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(multi*embed_dim, embed_dim)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        layer_normal_1 = self.layer_normal_1(input)
        atten = self.attention(layer_normal_1)
        atten = atten + layer_normal_1
        layer_normal_2 = self.layer_normal_2(atten)
        h = self.proj(layer_normal_2)
        h = self.dropout(h)
        return h


class Gpt2(nn.Module):
    def __init__(self, isMask=True):
        super().__init__()
        self.word_embd = nn.Embedding(vocab_num, embed_dim)
        self.pos_embd = nn.Embedding(pos_num, embed_dim)
        self.blocks = []
        for i in range(block_num):
            self.blocks.append(Block(isMask))
        self.dropout = nn.Dropout(0.1)
        self.seque = nn.Sequential(*self.blocks)
        self.output = nn.Linear(embed_dim, vocab_num,bias=False)

    def forward(self, word: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        word_embd = self.word_embd(word)
        pos_embd = self.pos_embd(pos)
        drop = self.dropout(word_embd+pos_num)
        # word_seque = self.seque(word_embd)
        # pos_seque = self.seque(pos_embd)
        h = self.seque(drop)
        # w_p_drop = self.dropout(word_seque + pos_seque)
        output = self.output(h)
        return output

class Gpt21(nn.Module):

    def __init__(self, isMask=True):
        super().__init__()

        self.vocab_embed = nn.Embedding(vocab_num, embed_dim)  # [4413, 768]
        self.pos_embed = nn.Embedding(pos_num, embed_dim)  # [500, 768]
        # self.type_embed = nn.Embedding(cfg.type_num, cfg.embed_dim)  # [30, 768]

        self.blocks = []
        for _ in range(block_num):
            self.blocks.append(Block(isMask))  # 6
        # print("self.blocks", len(self.blocks), self.type_embed.weight.shape)

        self.drop = nn.Dropout(0.1)

        self.sequential = nn.Sequential(*self.blocks)
        # for n,p in self.sequential.named_parameters():
        #     print(n)

        self.output_layer = nn.Linear(embed_dim, vocab_num, bias=False)  # [4413, 768]
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
    word = torch.tensor([[0, 1, 2]]).cuda()
    # RuntimeError: Expected tensor for argument #1 'indices' to have scalar type Long; but got torch.FloatTensor instead
    # pos = torch.tensor([[0, 1, 2]]).cuda()
    poses = torch.arange(pos_num).reshape(1,-1).repeat(word.size(0),1).cuda()
    pos = poses[:,:word.size(-1)]
    gpt2 = Gpt2().cuda()
    y = gpt2(word, pos)[:, -1:]
    print(y.shape)
    v, y = torch.topk(y, 8, dim=-1)
    v = v.reshape(-1, 8)
    y = y.reshape(-1, 8)
    print(v, y)
    index = torch.multinomial(torch.softmax(v,dim=-1),1)
    print(index.shape)
    y = torch.gather(y, dim=-1,index=index)
    print(y)
    word = torch.cat((word,y),dim=-1)
    print(word)
