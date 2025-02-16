import tensorflow as tf
import sonnet as snt
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import model_DeepSEQ_con
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score, accuracy_score
import itertools
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import gc
from base_seq import omics_data

def get_tokenizer():
    f = ['a', 'c', 'g', 't']
    c = itertools.product(f, f, f, f, f, f)
    res = []
    for i in c:
        temp = i[0] + i[1] + i[2] + i[3] + i[4] + i[5]
        res.append(temp)
    res = np.array(res)
    NB_WORDS = 4097
    tokenizer = Tokenizer(num_words=NB_WORDS)
    tokenizer.fit_on_texts(res)
    acgt_index = tokenizer.word_index
    acgt_index['null'] = 0
    return tokenizer


def sentence2word(str_set):
    word_seq = []
    for sr in str_set:
        tmp = []
        for i in range(len(sr) - 5):
            if ('N' in sr[i:i + 6]):
                tmp.append('null')
            else:
                tmp.append(sr[i:i + 6])
        word_seq.append(' '.join(tmp))
    return word_seq


def word2num(wordseq, tokenizer, MAX_LEN):
    sequences = tokenizer.texts_to_sequences(wordseq)
    numseq = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    return numseq


def sentence2num(str_set, tokenizer, MAX_LEN):
    wordseq = sentence2word(str_set)
    numseq = word2num(wordseq, tokenizer, MAX_LEN)
    return numseq


def encode_sequences(sequences):

    Encode = {"A": [1, 0, 0, 0], "T": [0, 1, 0, 0], "G": [0, 0, 1, 0], "C": [0, 0, 0, 1]}

    encoded_list = [[Encode[x] for x in seq] for seq in sequences]

    return encoded_list


def get_data(train, test, omics_train, omics_test):

    tokenizer = get_tokenizer()
    MAX_LEN = 200


    X_tr = sentence2num(train, tokenizer, MAX_LEN)
    X_te = sentence2num(test, tokenizer, MAX_LEN)

    X_tr_onehot = encode_sequences(train)
    X_te_onehot = encode_sequences(test)

    X_tr_omics = omics_train
    X_te_omics = omics_test

    return X_tr, X_te, X_tr_onehot, X_te_onehot, X_tr_omics, X_te_omics

def create_dataset(Kfold):
    print('Create dataset...')

    dataPath = './dataset_seq/fold'
    train = './data/datasets\sequence' + '.csv'
    train_csv = pd.read_csv(train)

    omics = pd.DataFrame(omics_data)

    train_seq_num = 0
    test_seq_num = 0

    for i in range(Kfold):
        if not os.path.exists(dataPath + '/fold_' + str(i + 1)):
            os.makedirs(dataPath + '/fold_' + str(i + 1))

        train_writer = tf.io.TFRecordWriter(dataPath + '/fold_' + str(i + 1) + "/train")
        test_writer = tf.io.TFRecordWriter(dataPath + '/fold_' + str(i + 1) + "/valid")

        train_part = train_csv.drop(train_csv.index[(len(train_csv)//5) * i:(len(train_csv)//5) * (i+1)], inplace=False)
        test_part = train_csv[(len(train_csv)//5) * i:(len(train_csv)//5) * (i+1)]
        train_label = train_part['label'].tolist()
        train_seq = train_part['sequence'].tolist()
        test_label = test_part['label'].tolist()
        test_seq = test_part['sequence'].tolist()

        train_seq_num = len(train_part)
        test_seq_num = len(test_part)


        train_omics = omics.iloc[train_part.index].values.tolist()
        test_omics = omics.iloc[test_part.index].values.tolist()

        print(i, train_seq_num, test_seq_num)

        X_train, X_test, X_train_onehot, X_test_onehot, X_train_omics, X_test_omics = get_data(train_seq, test_seq, train_omics, test_omics)


        for j in range(train_seq_num):
            sequence = X_train[j]
            onehot_sequence = X_train_onehot[j]
            omics_sequence = X_train_omics[j]

            feature = {
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(train_label[j])])),
                'sequence': tf.train.Feature(int64_list=tf.train.Int64List(value=sequence)),
                'onehot_sequence': tf.train.Feature(int64_list=tf.train.Int64List(value=np.ravel(onehot_sequence))),  # 一维化onehot序列
                'omics': tf.train.Feature(float_list=tf.train.FloatList(value=omics_sequence))
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            train_writer.write(example.SerializeToString())
        train_writer.close()


        for j in range(test_seq_num):
            sequence = X_test[j]
            onehot_sequence = X_test_onehot[j]
            omics_sequence = X_test_omics[j]
            feature = {
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(test_label[j])])),
                'sequence': tf.train.Feature(int64_list=tf.train.Int64List(value=sequence)),
                'onehot_sequence': tf.train.Feature(int64_list=tf.train.Int64List(value=np.ravel(onehot_sequence))),
                'omics': tf.train.Feature(float_list=tf.train.FloatList(value=omics_sequence))
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            test_writer.write(example.SerializeToString())
        test_writer.close()

    return train_seq_num, test_seq_num


def get_dataset(dataPath):

    dataset = tf.data.TFRecordDataset([dataPath], num_parallel_reads=8)

    dataset = dataset.map(_parse_function, num_parallel_calls=8)
    return dataset




def _parse_function(record):

    features = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'sequence': tf.io.FixedLenFeature([200], tf.int64),
        'onehot_sequence': tf.io.FixedLenFeature([200 * 4], tf.int64),
        'omics': tf.io.FixedLenFeature([12], tf.float32)
    }

    example = tf.io.parse_single_example(record, features)

    label = tf.cast(example['label'], tf.int64)

    sequence = tf.cast(example['sequence'], tf.int64)
    onehot_sequence = tf.cast(example['onehot_sequence'], tf.int64)
    omics = tf.cast(example['omics'], tf.float32)

    return {
        'sequence': sequence,
        'onehot_sequence': onehot_sequence,
        'label': label,
        'omics': omics
    }

def create_step_function(model, optimizer):
    @tf.function
    def train_step(batch, optimizer_clip_norm_global=0.2):
        with tf.GradientTape() as tape:

            sequence = batch['sequence']
            onehot_sequence = batch['onehot_sequence']
            omics = batch['omics']

            outputs = model(sequence, onehot_sequence, omics, is_training=True)


            label = tf.expand_dims(batch['label'], axis=1)
            label = tf.cast(label, dtype=tf.float32)


            loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=outputs)
            )
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply(gradients, model.trainable_variables)

        return loss

    return train_step


def get_metric(y_test, y_test_pred):
    test_pred_class = y_test_pred >= 0.5
    test_acc = accuracy_score(y_test, test_pred_class)
    test_auc = roc_auc_score(y_test, y_test_pred)
    test_aupr = average_precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, test_pred_class, pos_label=1)
    test_precision = precision_score(y_test, test_pred_class, pos_label=1)

    result = [round(test_acc, 4), round(test_auc, 4), round(test_aupr, 4), round(test_recall, 4),
              round(test_precision, 4)]

    return result


def evaluate_model(model, dataset, max_steps=None):
    y_true = np.array([])
    y_predict = np.array([])


    @tf.function
    def predict(sequence, onehot, inputs_omics):


        return model(sequence, onehot, inputs_omics, is_training=False)


    for i, batch in tqdm(enumerate(dataset)):
        if max_steps is not None and i > max_steps:
            break

        y_true = np.append(y_true, np.array(batch['label']))
        y_predict = np.append(y_predict, np.array(tf.nn.sigmoid(predict(batch['sequence'],onehot=batch['onehot_sequence'], inputs_omics=batch['omics']))))

    return y_true, y_predict


def save_model_with_signature(model, save_path):

    if not os.path.exists(save_path):
        os.makedirs(save_path)


    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=save_path, max_to_keep=1)
    checkpoint_manager.save()

    # 创建并保存模型的签名
    signature = create_model_signature(model)
    signature_path = os.path.join(save_path, "model_signature")
    tf.saved_model.save(model, signature_path, signatures=signature)

    print(f"Model and signature saved to {save_path}")


def create_model_signature(model):

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 200], dtype=tf.int64),
                                  tf.TensorSpec(shape=[None, 200 * 4], dtype=tf.int64),
                                  tf.TensorSpec(shape=[None, 12], dtype=tf.float32)])
    def predict_signature(sequence, onehot_sequence, omics):
        return model(sequence, onehot_sequence, omics, is_training=False)

    return predict_signature

def compute_average_result(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()[1:]
        acc_list, auc_list, aupr_list, rec_list, pre_list = [], [], [], [], []

        for line in lines:
            values = line.strip().split('\t')
            acc_list.append(float(values[0]))
            auc_list.append(float(values[1]))
            aupr_list.append(float(values[2]))
            rec_list.append(float(values[3]))
            pre_list.append(float(values[4]))

        # 计算每个指标的平均值
        avg_acc = sum(acc_list) / len(acc_list)
        avg_auc = sum(auc_list) / len(auc_list)
        avg_aupr = sum(aupr_list) / len(aupr_list)
        avg_rec = sum(rec_list) / len(rec_list)
        avg_pre = sum(pre_list) / len(pre_list)

    return avg_acc, avg_auc, avg_aupr, avg_rec, avg_pre



os.environ["CUDA_VISIBLE_DEVICES"] = '0'
batch_size = 64
learning_rate = 0.005
num_epochs = 5
Kfold = 5

train_seq_num, test_seq_num = create_dataset(Kfold)
train_seq_num = 2685
test_seq_num = 671




heads_param = 2
channels_param = 64
transformer_param = 1

best_result = [0, 0, 0, 0, 0]


result_writer = open(f"final/result_{heads_param}_{channels_param}_{transformer_param}.txt",
                     'a')
result_writer.write('ACC' + '\t' + 'AUC' + '\t' + 'AUPR' + '\t' + 'REC' + '\t' + 'PRE' + '\n')

for fold in range(Kfold):
    bestResult = [0, 0, 0, 0, 0]

    train_dataset = get_dataset(f"dataset_seq/fold/fold_{fold + 1}/train").batch(
        batch_size).repeat().prefetch(batch_size * 2)



    model = model_DeepSEQ_con.Enformer(
        channels=channels_param, num_transformer_layers=transformer_param, num_heads=heads_param)
    optimizer = snt.optimizers.Adam(learning_rate=learning_rate)
    train_step = create_step_function(model, optimizer)

    steps_per_epoch = train_seq_num // batch_size

    data_it = iter(train_dataset)
    for epoch_i in range(num_epochs):
        for i in tqdm(range(steps_per_epoch)):
            batch_train = next(data_it)
            loss = train_step(batch=batch_train)

        y_true, y_predict = evaluate_model(model, dataset=get_dataset(
            f"dataset_seq/fold/fold_{fold + 1}/valid").batch(batch_size * 2).prefetch(
            batch_size * 2))
        result = get_metric(y_true, y_predict)


        if result[1] > bestResult[1]:
            bestResult = result
            save_path = f"model/fold_{fold + 1}_best_model"
            save_model_with_signature(model, save_path)

        print(f"Fold {fold}, Epoch {epoch_i}: {result}")

    result_writer.write(
        f"{bestResult[0]}\t{bestResult[1]}\t{bestResult[2]}\t{bestResult[3]}\t{bestResult[4]}\n")
    result_writer.flush()


result_writer.close()


avg_acc, avg_auc, avg_aupr, avg_rec, avg_pre = compute_average_result(
    f"final/result_{heads_param}_{channels_param}_{transformer_param}.txt")


print(f"Average result for heads={heads_param}, channels={channels_param}, transformer_layer={transformer_param}:")
print(f"ACC: {avg_acc}, AUC: {avg_auc}, AUPR: {avg_aupr}, REC: {avg_rec}, PRE: {avg_pre}")



