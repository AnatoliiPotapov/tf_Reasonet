import json
from os.path import join
from tqdm import tqdm
from tensorflow.python.platform import gfile
from collections import defaultdict
import re
import numpy as np
import nltk

from ptpython.repl import embed


POS_TAGS = []

def parse_json(data_file):
    data = []
    with open(data_file) as f:
        for line in f:
            tokens = line.rstrip().split('\t')
            datum = {'sentence1':tokens[0],
                    'sentence2':tokens[1],
                    'gold_label':tokens[2]}
            data.append(datum)
    return data


def process_dataset(data):
    '''
    Converts raw data to training examples (sent1,sent2,label)
    '''
    final_data = []
    for id in tqdm(range(len(data)), desc="Converting to training examples"):
        sent1 = data[id]['sentence1']
        sent2 = data[id]['sentence2']
        label = data[id]['gold_label']
        final_data.append([sent1, sent2, label])
    return final_data


def tokenizer(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(" ", space_separated_fragment))
    return [w for w in words if w]


def create_vocabulary(data, vocab_path):
    if not gfile.Exists(vocab_path):
        vocab = defaultdict(int)
        for id in tqdm(range(len(data)), desc="Creating vocabulary"):
            sentences = ' '.join([data[id][0], data[id][1]])
            tokens = tokenizer(sentences)
            for w in tokens:
                vocab[w] += 1
        # 0 is used for padding
        vocab_list = ['<PAD>', '<UNK>'] + sorted(vocab, key=vocab.get,
                                                 reverse=True)
        with gfile.GFile(vocab_path, mode="wb") as f:
            for w in vocab_list:
                f.write(w + '\n')
        with gfile.GFile("pos_vocab.dat", mode="wb") as f:
            for w in POS_TAGS:
                f.write(w + '\n')



def init_vocabulary(vocab_path):
    if gfile.Exists(vocab_path):
        rev_vocab = []
        vocab = {}
        with gfile.GFile(vocab_path, mode='r') as f:
            for line in f:
                rev_vocab.append(line.strip('\n'))
        vocab = dict([(y, x) for x, y in enumerate(rev_vocab)])
        return vocab, rev_vocab


def sent_token_ids(vocab, sentence):
    '''
    return vocab mapping for each word, and 1 for UNK
    '''
    return [vocab.get(w, 1) for w in sentence]


def process_glove(vocab_list, glove_path, save_path, size=4e5):
    '''
    create a trimmed glove repr
    '''
    if not gfile.Exists(save_path + ".npz"):
        glove = np.zeros((len(vocab_list), 100))
        found = 0
        with open(glove_path, 'r') as fh:
            for line in tqdm(fh, total=size, desc="Processing Glove"):
                array = line.lstrip().rstrip().split(" ")
                word = array[0]
                vector = list(map(float, array[1:]))
                if word in vocab_list:
                    idx = vocab_list.index(word)
                    glove[idx, :] = vector
                    found += 1
                if word.capitalize() in vocab_list:
                    idx = vocab_list.index(word.capitalize())
                    glove[idx, :] = vector
                    found += 1
                if word.upper() in vocab_list:
                    idx = vocab_list.index(word.upper())
                    glove[idx, :] = vector
                    found += 1

        print("{}/{} of word vocab have corresponding vectors in {}".format(found, len(vocab_list), glove_path))
        np.savez_compressed(save_path, glove=glove)
        print("saved trimmed glove matrix at: {}".format(save_path))

def pos_id(x):
    '''
    returns an id for pos_tag, uses POS_TAG
    '''
    if x not in POS_TAGS:
        POS_TAGS.append(x)
    return POS_TAGS.index(x)

def convert_data(vocab, data, save_path):
    '''
    converts sentences to glove ids, and saves
    '''
    if not gfile.Exists(save_path):
        wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()
        with gfile.GFile(save_path, 'w') as f:
            for i in tqdm(range(len(data)), desc="creating {}".format(save_path)):
                sent1, sent2, label = data[i]
                tokens1 = nltk.word_tokenize(sent1)
                tokens2 = nltk.word_tokenize(sent2)
                lemmas1 = [wordnet_lemmatizer.lemmatize(x) for x in tokens1]
                lemmas2 = [wordnet_lemmatizer.lemmatize(x) for x in tokens2]
                lower1 = [x.lower() for x in tokens1]
                lower2 = [x.lower() for x in tokens2]
                to_write = []

                _ = [to_write.append(x) for x in sent_token_ids(vocab, sent1)]
                to_write.append(';;;')
                _ = [to_write.append(x) for x in sent_token_ids(vocab, sent2)]
                to_write.append(';;;')
                pos1 = nltk.pos_tag(sent1)
                _ = [to_write.append(pos_id(x)) for (y,x) in pos1]
                to_write.append(';;;')
                pos2 = nltk.pos_tag(sent2)
                _ = [to_write.append(pos_id(x)) for (y,x) in pos2]
                to_write.append(';;;')
                _ = [to_write.append(1) if x in tokens2 else to_write.append(0) for x in tokens1]
                to_write.append(';;;')
                _ = [to_write.append(1) if x in lemmas2 else to_write.append(0) for x in lemmas1]
                to_write.append(';;;')
                _ = [to_write.append(1) if x in lower2 else to_write.append(0) for x in lower1]
                to_write.append(';;;')
                to_write.append(int(label))
                f.write(str(to_write) + '\n')


if __name__ == '__main__':

    data_dir = './data/dont_know'
    train_file = 'train_data_balanced.txt'
    dev_matched = 'dev_data.txt'
    dev_mismatched = 'multinli_0.9_dev_mismatched.jsonl'
    vocab_path = 'vocab.dat'
    glove_path = 'glove.6B.100d.txt'
    train_save_path = 'dn_features.train'
    dev_matched_save_path = 'dn_features.dev'

    train_data = parse_json(join(data_dir, train_file))
    dev_matched_data = parse_json(join(data_dir, dev_matched))

    train_x = process_dataset(train_data)
    valid_x = process_dataset(dev_matched_data)

    create_vocabulary(train_x + valid_x,
                      join(data_dir, vocab_path))
    vocab, rev_vocab = init_vocabulary(join(data_dir, vocab_path))

    #process_glove(rev_vocab, join(data_dir, glove_path), join(data_dir, 'glove.trimmed.100'))

    convert_data(vocab, train_x, join(data_dir, train_save_path))
    convert_data(vocab, valid_x, join(data_dir, dev_matched_save_path))
