import os

SRC_DIRS = [r"../datas/files"]
DST_DIRS = r"../datas/tockens"
VOCAB_FILE = r"../datas/vocabs/vocab1.txt"

# if not os.path.exists(DST_DIRS):
#     os.makedirs(DST_DIRS)

with open(VOCAB_FILE, 'r+', encoding='utf-8') as f:
    tokens = f.read().split()
    print(len(tokens))
count = 0
for SRC_DIR in SRC_DIRS:
    for i, filename in enumerate(os.listdir(SRC_DIR)):
        if i > 2:
            break
        # if i < 2:
        #     continue
        f_file = os.path.join(SRC_DIR, filename)
        print(f_file)
        with open(f_file, 'r+', encoding='utf-8') as f:
            dst = ['0']
            w = f.read(1)
            while w:
                if w == '\n' or w == '\r' or w == '\t' or ord(w) == 12288:  # w ==chr(12288)
                    dst.append('1')
                elif w == ' ':
                    dst.append('3')
                else:
                    try:
                        # print(f_file)
                        print('w',w)
                        # print("index", tokens.index(w))
                        dst.append(str(tokens.index(w)))
                    except BaseException as e:
                        print("ord",ord(w))
                        print("e",e)
                        # exit()
                        dst.append('2')
                w = f.read(1)
            count += 1
        with open(os.path.join(DST_DIRS, "{0}.txt".format(count)),'w',encoding='utf-8') as df:
            df.write(' '.join(dst))

