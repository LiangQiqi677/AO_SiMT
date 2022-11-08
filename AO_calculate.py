import sys
import math
import numpy as np

waitk=1
align='data/wmt15_deen_align'
output='data/wmt15_deen_AO'
src='data/train.norm.tok.bpe.de'
trg='data/train.norm.tok.bpe.en'

align_line, output_line = [], []
with open(align, "r", encoding="utf-8") as f:
    for line in f:
        align_line.append(line.strip())

src_len_line, trg_len_line = [], []
with open(src, "r", encoding="utf-8") as f:
    for line in f:
        src_len_line.append(len(line.strip().split(" ")))
with open(trg, "r", encoding="utf-8") as f:
    for line in f:
        trg_len_line.append(len(line.strip().split(" ")))

j = 0
for line in align_line:
    sub_line = line.split(" ")
    src_line, trg_line = [] , []
    for token_algin in sub_line:
        try:
            src_align, trg_align = token_algin.split("-")
        except ValueError:
            src_align, trg_align = 0, 0
        src_line.append(int(src_align))
        trg_line.append(int(trg_align))

    AO_line = [0] * trg_len_line[j]
    count_line = [0] * trg_len_line[j]
    for i in range(len(src_align)):
        if src_align[i] - trg_align[i] - k > 0:
            AO_line[trg_align[i]] += src_align[i] - trg_align[i] - (waitk -1)
        count_line[trg_align[i]] += 1

    count_line = np.clip(np.array(count_line), 1, 100000).tolist()
    for i in range(len(AO_line)):
        AO_line[i] = AO_line[i]/count_line[i]
    trg_line.append(' '.join(AO_line))
    j += 1

with open(output, "w", encoding="utf-8") as f:
    for i in range(len(output_line)):
        f.write(output_line[i] + '\n')
