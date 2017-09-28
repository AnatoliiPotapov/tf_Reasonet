import os
import logging

from tensorflow.python.platform import gfile
import numpy as np
from os.path import join
import tensorflow as tf
from ptpython.repl import embed
import random
from tqdm import tqdm
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def get_minibatches(data, minibatch_size, dataset):
    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    indices = np.arange(data_size)
    np.random.shuffle(indices)
    if dataset == 'squad':
        squad_flag = True
    else:
        squad_flag = False
    if squad_flag:
        data = negative_sampling(data, minibatch_size)
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
        batches = [minibatch(d, minibatch_indices) for d in data] if list_data else minibatch(data, minibatch_indices)
        yield batches

def negative_sampling(batches, minibatch_size):
    q_final_batch = []
    q_final_len_batch = []
    c_final_batch = []
    c_final_len_batch = []
    cf_final_batch = []
    cf_final_len_batch = []
    a_final_batch = []
    a_final_len_batch = []
    infer_label_batch = []
    for q, q_l, c, c_l, cf, cf_l,a, a_l in zip(*batches):
        q_final_batch.append(q)
        q_final_len_batch.append(q_l)
        c_final_batch.append(c)
        c_final_len_batch.append(c_l)
        cf_final_batch.append(cf)
        cf_final_len_batch.append(cf_l)
        a_final_batch.append(a)
        a_final_len_batch.append(a_l)
        infer_label_batch.append(1)

        for i, qq in enumerate(batches[0]):
            cc = batches[2][i]
            if qq != q and cc != c:
                q_final_batch.append(qq)
                q_final_len_batch.append(batches[1][i])
                c_final_batch.append(c)
                c_final_len_batch.append(c_l)
                cf_final_batch.append(cf)
                cf_final_len_batch.append(cf_l)
                a_final_batch.append(a)
                a_final_len_batch.append(a_l)
                infer_label_batch.append(0)
                break #
    data = [q_final_batch, q_final_len_batch, c_final_batch, c_final_len_batch, cf_final_batch, cf_final_len_batch ,a_final_batch, a_final_len_batch, infer_label_batch]
    return data


def minibatch(data, minibatch_idx):
    batches = data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]
    return batches


def minibatches(data, batch_size, dataset):
    batches = [np.array(col) for col in zip(*data)]
    return get_minibatches(batches, batch_size, dataset)


def load_glove_embeddings(glove_path):
    glove = np.load(glove_path)['glove']
    logger.info("Loading glove embedding")
    logger.info("Dimension: {}".format(glove.shape[1]))
    logger.info("Vocabulary: {}" .format(glove.shape[0]))
    glove = tf.to_float(glove)
    return glove

def load_dontknow_dataset(data_size, max_question_length, max_context_length):
    train_data = join("data","dont_know","dn_features.train")
    dev_data = join("data","dont_know","dn_features.dev")
    train = []
    valid = []

    with gfile.GFile(train_data, 'r') as f:
        for line in tqdm(f, desc="Reading training data"):
            line = line.strip().replace('[', '').replace(']', '')
            tokens = line.split(", ';;;',") # Hacky.. #TODO fix the preprocessing script
            data = [list(map(int, x.strip().split(','))) for x in tokens]
            #   data = sent1,sent2,pos1,pos2,sim_word,sim_lemma,sim_lower,label
            #sent1 = data[0] + data[2] + data[4] + data[5] + data[6] + data[7]# sent1 + pos1 +  + sim_word + sim_lemma + sim_lower
            sent1 = data[0] # sent1
            f_sent1 = data[4] + data[5] + data[6] # sim_word + sim_lemma + sim_lower
            sent2 = data[1] # + data[3] #removing pos info
            label = data[-1]
            train.append([sent1, len(sent1), sent2, len(sent2), f_sent1,len(f_sent1), [], 0, label[0]])


    with gfile.GFile(dev_data, 'r') as f:
        for line in tqdm(f, desc="Reading validation data"):
            line = line.strip().replace('[', '').replace(']', '')
            tokens = line.split(", ';;;',") # Hacky.. #TODO fix the preprocessing script
            data = [list(map(int, x.strip().split(','))) for x in tokens]
            #   data = sent1,sent2,pos1,pos2,sim_word,sim_lemma,sim_lower,label
            sent1 = data[0] # sent1
            f_sent1 = data[4] + data[5] + data[6] # sim_word + sim_lemma + sim_lower
            sent2 = data[1] # + data[3] #removing pos info
            label = data[-1]
            valid.append([sent1, len(sent1), sent2, len(sent2), f_sent1,len(f_sent1), [], 0, label[0]])

    if data_size=="tiny":
        train = train[:100]
        valid = valid[:10]

    dataset = {"training":train, "validation":valid,"training_raw":[],"validation_raw":[]}
    return dataset

def load_dataset(source_dir, data_size, max_q_toss, max_c_toss, data_pfx_list=None):
    '''
    From Stanford Assignment 4 starter code
    '''
    assert os.path.exists(source_dir)
    train_pfx = join(source_dir, "train")
    valid_pfx = join(source_dir, "val")
    dev_pfx = join(source_dir, "dev")

    train = []
    valid = []
    train_raw = []
    valid_raw = []

    max_c_len = 0
    max_q_len = 0
    max_a_len = 0

    if data_pfx_list is None:
        data_pfx_list = [train_pfx, valid_pfx]
    else:
        data_pfx_list = [join(source_dir, data_pfx) for data_pfx in data_pfx_list]

    for data_pfx in data_pfx_list:
        if data_pfx == train_pfx:
            data_list = train
            data_list_raw = train_raw

        if data_pfx == valid_pfx:
            data_list = valid
            data_list_raw = valid_raw

        c_ids_path = data_pfx + ".ids.context"
        c_raw_path = data_pfx + ".context"
        q_ids_path = data_pfx + ".ids.question"
        q_raw_path = data_pfx + ".question"
        label_path = data_pfx + ".span"

        uuid_list = []
        if data_pfx == dev_pfx:
            uuid_path = data_pfx + ".uuid"
            with gfile.GFile(uuid_path, mode="rb") as uuid_file:
                for line in uuid_file:
                    uuid_list.append(line.strip())

        with gfile.GFile(q_raw_path, mode="r") as r_q_file:
            with gfile.GFile(c_raw_path, mode="r") as r_c_file:
                with gfile.GFile(q_ids_path, mode="r") as q_file:
                    with gfile.GFile(c_ids_path, mode="r") as c_file:
                        with gfile.GFile(label_path, mode="r") as l_file:
                            for line in l_file:
                                label = list(map(int,line.strip().split(" ")))
                                context_plus_features = list(map(int, c_file.readline().strip().split(" ")))
                                question = list(map(int,q_file.readline().strip().split(" ")))
                                context_raw = r_c_file.readline().strip().split(" ")
                                question_raw = r_q_file.readline().strip().split(" ")

                                answers = list(map(int,context_plus_features[label[0]:label[1]]))
                                answer_raw = context_raw[label[0]:label[1]]
                                c_len = int(len(context_plus_features)/4)
                                q_len = len(question)
                                a_len = len(answers)

                                context = context_plus_features[:c_len]
                                context_features = context_plus_features[c_len:]
                                if q_len > max_q_toss:
                                    if data_pfx == dev_pfx:
                                        q_len = max_q_toss
                                        question = question[:max_q_toss]
                                if c_len > max_c_toss:
                                    if data_pfx == dev_pfx:
                                        c_len = max_c_toss
                                        context = context[:max_c_toss]

                                max_c_len = max(max_c_len, c_len)
                                max_q_len = max(max_q_len, q_len)
                                max_a_len = max(max_a_len, a_len)
                                entry = [question, q_len, context, c_len, context_features, len(context_features), answers,a_len]
                                data_list.append(entry)

                                raw_entry = [question_raw, context_raw, answer_raw]
                                data_list_raw.append(raw_entry)

        if data_size=="tiny":
            train = train[:100]
            valid = valid[:10]


    dataset = {"training":train, "validation":valid, "training_raw":train_raw, "validation_raw":valid_raw}
    return dataset

def load_snli_dataset(source_dir, data_size, max_sent1_len, max_sent2_len):

    train_data = join(source_dir,'mnli.train')
    dev_data = join(source_dir,'mnli.dev.matched')
    train = []
    valid = []
    with gfile.GFile(train_data, 'r') as f:
        for line in f:
            line = line.strip().replace('[', '').replace(']', '')
            tokens = line.split(',')
            label = int(tokens[-1])
            pos = tokens.index(" ';'")
            sent1 = tokens[:pos]
            sent2 = tokens[pos+1: len(tokens) - 2]
            map(int,sent1)
            map(int,sent2)
            train.append([sent1, len(sent1), sent2, len(sent2), label])

    with gfile.GFile(dev_data, 'r') as f:
        for line in f:
            line = line.strip().replace('[', '').replace(']', '')
            tokens = line.split(',')
            label = int(tokens[-1])
            pos = tokens.index(" ';'")
            sent1 = tokens[:pos]
            sent2 = tokens[pos+1: len(tokens) - 2]
            map(int,sent1)
            map(int,sent2)
            train.append([sent1, len(sent1), sent2, len(sent2), label])

    dataset = {"training":train, "validation":valid}
    return dataset, max_sent1_len, max_sent2_len
