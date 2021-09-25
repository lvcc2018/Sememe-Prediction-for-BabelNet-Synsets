from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import thulac
import pickle
from tqdm import tqdm

babel_data_file = 'data/babel_data_full.txt'    # BaBelNet原始数据
babel_sememe_file = 'data/synset_sememes.txt'   # BabelSememe原始数据
data = 'data/babel_data'                        # BabelNet原始数据词典
clean_data = 'data/clean_data'                  # BabelNet清洗后的数据

def read_list(line):
    line = line[:-1].split('\t')
    num = int(line[0])
    assert(num == len(line[1:]))
    return line[1:]

def read_synset(f):
    synset = {}
    synset['id'] = f.readline()
    if not synset['id']:
        return
    synset['id'] = synset['id'][:-1]
    for k in ['w_e','w_c','w_f']:
        synset[k] = read_list(f.readline())
    for k in ['d_e_m','d_c_m','d_f_m']:
        synset[k] = f.readline()[:-1]
    for k in ['d_e','d_c','d_f']:
        synset[k] = read_list(f.readline())
    synset['i_m'] = f.readline()[:-1]
    synset['i'] = read_list(f.readline())
    return synset

def read_babel_data(f):
    babel_data ={}
    while True:
        d = read_synset(f)
        if not d:
            return babel_data
        babel_data[d['id']] = d

def read_babel_sememe(f):
    lines = f.readlines()
    babel_sememe = {line[:-1].split()[0] : line[:-1].split()[1:] for line in lines}
    return babel_sememe

def get_babel_data():
    babel_data = read_babel_data(open(babel_data_file))
    babel_sememe = read_babel_sememe(open(babel_sememe_file))

    for k in babel_data:
        babel_data[k]['s'] = babel_sememe[k]
    pickle.dump(babel_data, open(data,'wb'))
