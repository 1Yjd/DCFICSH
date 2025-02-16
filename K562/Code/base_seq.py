# base_seq



import pandas as pd
import os
import numpy as np
def parse_fasta(file_path):
    sequences = []
    with open(file_path, 'r') as file:
        header = None
        seq = []
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if header:
                    full_sequence = ''.join(seq)
                    if 'N' not in full_sequence :
                        sequences.append((header, full_sequence))

                header = line[1:]
                seq = []
            else:
                seq.append(line)
        if header:
            full_sequence = ''.join(seq)
            if 'N' not in full_sequence :
                sequences.append((header, full_sequence))

    df = pd.DataFrame(sequences, columns=['Header', 'sequence'])
    return df


import numpy as np


def load_omics_feature(filename):
    dataMat = []

    file = np.loadtxt(filename, encoding='utf_8_sig', delimiter=' ')
    for line in file:
        curLine = list(line)
        floatLine = list(map(float, curLine))
        dataMat.append(floatLine[0:13])

    labelMat = []
    for i in dataMat:
        labelMat.append(int(i[-1]))
        del (i[-1])

    dataMat = np.array(dataMat)
    labelMat = np.array(labelMat)
    return dataMat, labelMat


filename1 = "../dataset/0K562_silencer_feature12_6924_lable.txt"
filename2 = "../dataset/0K562_negative_feature12_6924_lable.txt"


X_p_omics, Y_p_omics = load_omics_feature(filename1)
X_n_omics, Y_n_omics = load_omics_feature(filename2)

omics_data = np.vstack((X_p_omics, X_n_omics))



p_file_path = '../dataset/0K562_200bp.fa'
p_seq = parse_fasta(p_file_path)

n_file_path = '../dataset/0K562_negative_set6924_200bp.fa'
n_seq = parse_fasta(n_file_path)

p_seq['label'] = 1
n_seq['label'] = 0

all_seq = pd.concat([p_seq, n_seq], axis=0).reset_index(drop=True)

all_seq['id'] = range(len(all_seq))


shuffled_indices = all_seq.sample(frac=1, random_state=42).index

all_seq = all_seq.iloc[shuffled_indices].reset_index(drop=True)
omics_data = omics_data[shuffled_indices]

all_seq = all_seq.drop(columns=['id'])


save_dir = './data/datasets'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


save_path_all_seq = os.path.join(save_dir, 'sequence.csv')
all_seq[['sequence', 'label']].to_csv(save_path_all_seq, index=False)

