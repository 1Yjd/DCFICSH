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

def load_omics_data():
    # # Load omics feature
    filename1 = "../dataset/X_omics_train.txt"
    filename2 = "../dataset/X_omics_test.txt"


    X_train_omics, Y_train_omics = load_omics_feature(filename1)
    X_test_omics, Y_test_omics = load_omics_feature(filename2)

    return X_train_omics, X_test_omics,Y_train_omics, Y_test_omics

X_train_omics, X_test_omics,Y_train_omics, Y_test_omics = load_omics_data()

train_file_path = '../dataset/X_fa_train_200bp.fa'
train_seq = parse_fasta(train_file_path)

test_file_path = '../dataset/X_fa_test_200bp.fa'
test_seq = parse_fasta(test_file_path)

train_seq['label'] = Y_train_omics
test_seq['label'] = Y_test_omics



save_dir = './data/datasets'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


save_path_train_seq = os.path.join(save_dir, 'train_sequence.csv')
save_path_test_seq = os.path.join(save_dir, 'test_sequence.csv')
train_seq[['sequence', 'label']].to_csv(save_path_train_seq, index=False)
test_seq[['sequence', 'label']].to_csv(save_path_test_seq, index=False)

print(X_train_omics.shape)
print(X_test_omics.shape)