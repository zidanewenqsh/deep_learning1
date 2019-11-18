import torch

def creatMask(pos_num):
    mask = torch.zeros(1,1,pos_num,pos_num)
    for i in range(pos_num):
        for j in range(i+1):
            mask[:,:,i,j]=1
    return mask

if __name__ == '__main__':
    pos_num=10
    mask = creatMask(pos_num)
    print(mask)