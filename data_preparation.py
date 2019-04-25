import os
import re
import pandas
import string
import numpy as np
import gensim.models.keyedvectors as word2vec
from mlxtend.preprocessing import one_hot
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA

stop_words = stopwords.words('english')
embeddings_index = dict()


dimensions = {
    "IMDB": 52,
    "ProcCons": 28,
    'MR': 28,
    'SST-1': 28,
    'SST-2': 28,
    'SUBJ': 28,
    'TREC': 28
}


def clean_str(s):
    s = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", s)
    s = re.sub(r"\'s", " \'s", s)
    s = re.sub(r"\'ve", " \'ve", s)
    s = re.sub(r"n\'t", " n\'t", s)
    s = re.sub(r"\'re", " \'re", s)
    s = re.sub(r"\'d", " \'d", s)
    s = re.sub(r"\'ll", " \'ll", s)
    s = re.sub(r",", " , ", s)
    s = re.sub(r"!", " ! ", s)
    s = re.sub(r"\(", " \( ", s)
    s = re.sub(r"\)", " \) ", s)
    s = re.sub(r"\?", " \? ", s)
    s = re.sub(r"\s{2,}", " ", s)
    s.strip('\"')
    s.strip('\'')

    # split into words
    tokens = word_tokenize(s)

    # convert to lower case
    tokens = [w.lower() for w in tokens]

    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]

    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]

    # filter out stop words
    return [w for w in words if not w in stop_words]


def prepare_x(x_text, dimension):
    vec = CountVectorizer(tokenizer=clean_str)
    vec.fit_transform(x_text)
    total_empty = 0

    vocab_names = vec.get_feature_names()
    vocab_values = np.zeros((len(vocab_names), 300))

    for i in range(len(vocab_names)):
        embedding_vector = embeddings_index.get(vocab_names[i])

        if embedding_vector is not None:
            vocab_values[i] = embedding_vector
        else:
            total_empty += 1

    pca = PCA(n_components=dimension)
    vocab = pca.fit_transform(vocab_values)
    x_final = [clean_str(sent) for sent in x_text]

    X = np.zeros((len(x_final), 1, dimension, dimension))

    for i in range(len(x_final)):
        x = x_final[i]
        text = np.zeros((dimension, dimension))

        for j in range(dimension):
            if j < len(x):
                if x[j] in vocab_names:
                    text[j] = vocab[vocab_names.index(x[j])]
            else:
                break

        X[i][0] = text

    return X, total_empty


def load_imdb(folder, output, dimension=dimensions['IMDB']):
    x_text = list()
    y_text = list()

    for file in os.listdir(folder + '/test/pos'):
        review_file = open(folder + '/test/pos/' + file, 'r', encoding='utf-8')
        x_text.append(review_file.readline())
        y_text.append(1)
        review_file.close()

    for file in os.listdir(folder + '/test/neg'):
        review_file = open(folder + '/test/neg/' + file, 'r', encoding='utf-8')
        x_text.append(review_file.readline())
        y_text.append(0)
        review_file.close()

    for file in os.listdir(folder + '/train/pos'):
        review_file = open(folder + '/train/pos/' + file, 'r', encoding='utf-8')
        x_text.append(review_file.readline())
        y_text.append(1)
        review_file.close()

    for file in os.listdir(folder + '/train/neg'):
        review_file = open(folder + '/train/neg/' + file, 'r', encoding='utf-8')
        x_text.append(review_file.readline())
        y_text.append(0)
        review_file.close()

    # Generate X
    X, total_empty = prepare_x(x_text, dimension)

    Y = y_text

    np.save(output + '/X', X)
    np.save(output + '/Y', Y)

    f = open(output + '/empty_words', 'w+')
    f.write(str(total_empty))
    f.close()

    print('imdb done')


def load_pc(pos, neg, output, dimension=dimensions['ProcCons']):
    positive_examples = list(open(pos, encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(neg, encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]

    # Split by words
    x_text = negative_examples + positive_examples

    # Generate X
    X, total_empty = prepare_x(x_text, dimension)

    # Generate labels
    negative_labels = [0 for _ in negative_examples]
    positive_labels = [1 for _ in positive_examples]
    Y = np.concatenate([negative_labels, positive_labels], 0)

    np.save(output + '/X', X)
    np.save(output + '/Y', Y)

    f = open(output + '/empty_words', 'w+')
    f.write(str(total_empty))
    f.close()
    print('cr done')


def load_mr(pos, neg, output, dimension=dimensions['MR']):
    positive_examples = list(open(pos, encoding='latin-1').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(neg, encoding='latin-1').readlines())
    negative_examples = [s.strip() for s in negative_examples]

    # Split by words
    x_text = negative_examples + positive_examples

    # Generate X
    X, total_empty = prepare_x(x_text, dimension)

    # Generate labels
    negative_labels = [0 for _ in negative_examples]
    positive_labels = [1 for _ in positive_examples]
    Y = np.concatenate([negative_labels, positive_labels], 0)

    np.save(output + '/X', X)
    np.save(output + '/Y', Y)

    f = open(output + '/empty_words', 'w+')
    f.write(str(total_empty))
    f.close()
    print('mr done')


def load_sst1(train, dev, test, output, dimension=dimensions['SST-1']):
    x_text = list()
    y_text = list()

    # Split by words
    for line in [line.split(',', 1) for line in open(train, encoding='utf-8').readlines()]:
        y_text.append(int(line[0])-1)
        x_text.append(line[1])

    for line in [line.split(',', 1) for line in open(dev, encoding='utf-8').readlines()]:
        y_text.append(int(line[0])-1)
        x_text.append(line[1])

    for line in [line.split(',', 1) for line in open(test, encoding='utf-8').readlines()]:
        y_text.append(int(line[0])-1)
        x_text.append(line[1])

    # Generate X
    X, total_empty = prepare_x(x_text, dimension)

    # Generate labels
    Y = y_text

    np.save(output + '/X', X)
    np.save(output + '/Y', Y)

    f = open(output + '/empty_words', 'w+')
    f.write(str(total_empty))
    f.close()
    print('sst1 done')


def load_sst2(train, dev, test, output, dimension=dimensions['SST-2']):
    x_text = list()
    y_text = list()

    # Split by words
    for line in [line.split(',', 1) for line in open(train, encoding='utf-8').readlines()]:
        y_text.append(int(line[0])-1)
        x_text.append(line[1])

    for line in [line.split(',', 1) for line in open(dev, encoding='utf-8').readlines()]:
        y_text.append(int(line[0])-1)
        x_text.append(line[1])

    for line in [line.split(',', 1) for line in open(test, encoding='utf-8').readlines()]:
        y_text.append(int(line[0])-1)
        x_text.append(line[1])

    # Generate X
    X, total_empty = prepare_x(x_text, dimension)

    # Generate labels
    Y = y_text

    np.save(output + '/X', X)
    np.save(output + '/Y', Y)

    f = open(output + '/empty_words', 'w+')
    f.write(str(total_empty))
    f.close()
    print('sst2 done')


def load_subj(pos, neg, output, dimension=dimensions['SUBJ']):
    positive_examples = list(open(pos, encoding='latin-1').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(neg, encoding='latin-1').readlines())
    negative_examples = [s.strip() for s in negative_examples]

    # Split by words
    x_text = negative_examples + positive_examples

    # Generate X
    X, total_empty = prepare_x(x_text, dimension)

    # Generate labels
    negative_labels = [0 for _ in negative_examples]
    positive_labels = [1 for _ in positive_examples]
    Y = np.concatenate([negative_labels, positive_labels], 0)

    np.save(output + '/X', X)
    np.save(output + '/Y', Y)

    f = open(output + '/empty_words', 'w+')
    f.write(str(total_empty))
    f.close()
    print('subj done')


def load_trec(dev, test, output, dimension=dimensions['TREC']):
    categories = {'ABBR':0, 'ENTY':1, 'DESC':2, 'HUM':3, 'LOC':4, 'NUM':5}

    x_text = list()
    y_text = list()

    # Split by words
    for line in [line.split(' ', 1) for line in open(dev, encoding='utf-8').readlines()]:
        i = line[0].split(':')
        y_text.append(categories[i[0]])
        x_text.append(line[1])

    for line in [line.split(' ', 1) for line in open(test, encoding='utf-8').readlines()]:
        i = line[0].split(':')
        y_text.append(categories[i[0]])
        x_text.append(line[1])

    # Generate X
    X, total_empty = prepare_x(x_text, dimension)

    # Generate labels
    Y = y_text

    np.save(output + '/X', X)
    np.save(output + '/Y', Y)

    f = open(output + '/empty_words', 'w+')
    f.write(str(total_empty))
    f.close()
    print('trec done')


def load_glove(file_name):
    f = open(file_name, encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))


def load_word2vec(file_name):
    word2vecDict = word2vec.KeyedVectors.load_word2vec_format(file_name, binary=True)

    for word in word2vecDict.wv.vocab:
        embeddings_index[word] = word2vecDict.word_vec(word)
    print('Loaded %s word vectors.' % len(embeddings_index))


def load_fasttext(file_name):
    f = open(file_name, encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))


def data_prepare(name, input='./datasets', output='./datasets_prepared'):
    print(name)
    if name == 'glove':
        load_glove('embeddings/glove.6B.300d.txt')
    elif name == 'word2vec':
        load_word2vec('embeddings/GoogleNews-vectors-negative300.bin')
    elif name == 'fasttext':
        load_fasttext('embeddings/wiki-news-300d-1M.vec')

    load_imdb(input + '/IMDB/IMDB', output + '/IMDB' + '/' + name)
    load_pc(input + '/ProcCons/ProcCons/IntegratedPros.txt', input + '/ProcCons/ProcCons/IntegratedCons.txt', output + '/ProcCons' + '/' + name)
    load_mr(input + '/MR/MR/rt-polarity.pos', input + '/MR/MR/rt-polarity.neg', output + '/MR' + '/' + name)
    load_sst1(input + '/SST-1/train.csv', input + '/SST-1/dev.csv', input + '/SST-1/test.csv', output + '/SST-1' + '/' + name)
    load_sst2(input + '/SST-2/train.csv', input + '/SST-2/dev.csv', input + '/SST-2/test.csv', output + '/SST-2' + '/' + name)
    load_subj(input + '/SUBJ/Subj/plot.tok.gt9.5000', input + '/SUBJ/Subj/quote.tok.gt9.5000', output + '/SUBJ' + '/' + name)
    load_trec(input + '/TREC/TREC/train_5500.label.txt', input + '/TREC/TREC/TREC_10.label.txt', output + '/TREC' + '/' + name)

data_prepare('glove')
data_prepare('word2vec')
data_prepare('fasttext')

