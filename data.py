# coding:utf-8
import json
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import string
import re
import random
import torch
import os
from tqdm import tqdm
from transformers import XLMRobertaTokenizer
import OpenHowNet
import thulac
from PIL import Image
from torchvision import transforms

# 含定义的synset文件
babel_glosses_file = './data/babel_glosses.txt'

# 含图片url的synset文件
babel_images_file = './data/babel_images.txt'

# 带义原标注的synset数据文件
synset_sememe_file = './data/synset_sememes.txt'

# 带义原标注的synset数据文件
sememe_file = './data/sememe_all.txt'

# BabelNet需要的synset数据
babel_data_file = './data/babel_data.json'

# 分词器
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

# 图片文件夹
babel_images_path = '/data2/private/lvchuancheng/babel_images'

# hownet接口
hownet_dict = OpenHowNet.HowNetDict()

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def en_lemmatize(wnl, sentence):
    tokens = word_tokenize(sentence)
    tagged_sent = pos_tag(tokens)
    lemmas_sent = []
    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
    return lemmas_sent

def cn_t2s(lac, sentence):
    result = lac.cut(sentence, text = True)
    result = result.split(' ')
    return result

def read_list(fin):
    line = fin.readline()
    line = line.strip().split('\t')
    try:
        num = eval(line[0])
        line = line[1:]
    except:
        return []
    return line

def read_synset_glosses(fin):
    bn = fin.readline()
    if not bn:
        return None
    bn = bn.strip()
    lemmas_EN = read_list(fin)
    lemmas_ZH = read_list(fin)
    lemmas_FR = read_list(fin)
    glosses_EN = read_list(fin)
    glosses_ZH = read_list(fin)
    glosses_FR = read_list(fin)
    fin.readline()
    return [bn, lemmas_EN, lemmas_ZH, lemmas_FR, glosses_EN, glosses_ZH, glosses_FR]

def read_synset_sememes(fin):
    sememes = []
    synset_dic = {}
    synset_id_list = []
    while True:
        line = fin.readline()
        if not line:
            return sememes, synset_dic, synset_id_list
        line = line.strip().split()
        synset_id = line[0]
        synset_sememes = line[1:]
        synset_id_list.append(synset_id)
        if synset_id not in synset_dic.keys():
            synset_dic[synset_id] = {}
        synset_dic[synset_id]['sememes'] = synset_sememes
        for sememe in synset_sememes:
            if sememe not in sememes:
                sememes.append(sememe)

def gen_babel_data():
    fin_ss = open(synset_sememe_file,'r',encoding = 'utf-8')
    fin_def = open(babel_glosses_file,'r',encoding = 'utf-8')
    fout = open(babel_data_file, 'w', encoding = 'utf-8')
    sememes, synset_dic, synset_id_list = read_synset_sememes(fin_ss)
    fin_ss.close()
    while True:
        item = read_synset_glosses(fin_def)
        if item == None:
            break
        if item[0] not in synset_id_list:
            continue
        synset_dic[item[0]]['word_en'] = item[1]
        synset_dic[item[0]]['word_cn'] = item[2]
        synset_dic[item[0]]['word_fr'] = item[3]
        synset_dic[item[0]]['definition_en'] = item[4]
        synset_dic[item[0]]['definition_cn'] = item[5]
        synset_dic[item[0]]['definition_fr'] = item[6]
    fin_def.close()
    synset_json = json.dump(synset_dic,fout,ensure_ascii=False)
    fout.close()

def data_clean(synset_dic_json):
    synset_list = []
    synset_dic = json.load(open(synset_dic_json))

    wnl = WordNetLemmatizer()
    lac = thulac.thulac(T2S=True,seg_only=True)

    for k in tqdm(synset_dic.keys()):
        synset = {}
        synset['s'] = [ss.split('|')[1] for ss in synset_dic[k]['sememes']]
        synset['b'] = k
        if 'definition_en' not in synset_dic[k].keys():
            print(synset_dic[k])
            return
        if len(synset_dic[k]['definition_en']) != 0:
            synset['w_e'] = (' | '.join([w.lower() for w in synset_dic[k]['word_en']])).split(' ')
            synset['d_e'] = en_lemmatize(wnl, synset_dic[k]['definition_en'][0].lower())
        
        if len(synset_dic[k]['definition_cn']) != 0:
            temp_w_c = [cn_t2s(lac,w) for w in synset_dic[k]['word_cn']]
            synset['w_c'] = []
            for i in range(len(temp_w_c)):
                synset['w_c'] += temp_w_c[i]
                synset['w_c'].append('|')
            if len(synset['w_c']) > 0:
                synset['w_c'].pop()
            synset['d_c'] = cn_t2s(lac,synset_dic[k]['definition_cn'][0])
        
        if len(synset_dic[k]['definition_fr']) != 0:
            synset['w_f'] = (' | '.join([w.lower() for w in synset_dic[k]['word_fr']])).split(' ')
            synset['d_f'] = synset_dic[k]['definition_fr'][0].lower().split(' ')

        synset_list.append(synset)
    with open('./data/data_all.json', 'w', encoding='utf-8') as f:
        json.dump(synset_list, f, ensure_ascii=False) 
    f.close()

def random_split_data(data_file, save_path):
    data = json.load(open(data_file))
    l = [i for i in range(len(data))]
    random.shuffle(l)
    train_data = [data[i] for i in l[0:int(0.8*len(data))]]
    valid_data = [data[i] for i in l[int(0.8*len(data)):int(0.9*len(data))]]
    test_data = [data[i] for i in l[int(0.9*len(data)):]]
    print(len(train_data), len(valid_data), len(test_data))
    json.dump(train_data, open(save_path+'/train_data.json', 'w', encoding='utf-8'), ensure_ascii=False)
    json.dump(valid_data, open(save_path+'/valid_data.json', 'w', encoding='utf-8'), ensure_ascii=False)
    json.dump(test_data, open(save_path+'/test_data.json', 'w', encoding='utf-8'), ensure_ascii=False)

def get_ids(word_list, hownet_dict, sememe_list, index_offset = 0):

    result_ids = []
    result_i2s = []
    idx = index_offset
    for w in word_list:
        idx_list = []
        word_ids = tokenizer(w)['input_ids']
        for i in range(1,len(word_ids)-1):
            idx += 1
            idx_list.append(idx)
            result_ids.append(word_ids[i])
        ids_sememe = hownet_dict.get_sememes_by_word(w,structured=False,lang="zh",merge=True)
        if ids_sememe:
            if isinstance(ids_sememe, dict):
                ids_sememe = list(list(ids_sememe.items())[0][1])
            elif isinstance(ids_sememe, set):
                ids_sememe = list(ids_sememe)
            temp = []
            for s in ids_sememe:
                if s in sememe_list:
                    temp.append(sememe_list.index(s))
            if temp:
                result_i2s.append([idx_list, temp])
    if len(result_ids) > 512:
        result_ids = result_ids[:511]
        temp = 0
        for i in range(len(result_i2s)):
            for j in result_i2s[i][0]:
                if j > 512 and temp == 0:
                    temp = i
                    break
        if temp != 0:
            result_i2s = result_i2s[:temp]
    return result_ids, result_i2s

def gen_training_data(input_data_file,  output_dir, lang = 'ecf'):
    with open('./data/sememe_all.txt', 'r', encoding='utf-8') as f:
        sememe_str = f.read()
    f.close()
    sememe_list = sememe_str.split(' ')
    input_data = json.load(open(input_data_file))
    output_data = []
    for instance in tqdm(input_data):
        temp = {}
        temp['s'] = [sememe_list.index(ss) for ss in instance['s']]
        temp['b'] = instance['b']
        temp['di'] = [0]
        temp['di_tw'] = [0]
        temp['si'] = []
        temp['si_tw'] = []
        index = 0
        index_tw = 0
        if 'e' in lang:
            if 'd_e' in instance.keys():
                result_ids, result_i2s = get_ids(instance['d_e'], hownet_dict, sememe_list, index)
                result_ids_tw, result_i2s_tw = get_ids(instance['w_e'] + [':'] + instance['d_e'], hownet_dict, sememe_list, index_tw)
                temp['di'] += result_ids + [2]
                temp['di_tw'] += result_ids_tw + [2]
                temp['si'] += result_i2s
                temp['si_tw'] += result_i2s_tw
                index += len(temp['di'])
                index_tw += len(temp['di_tw'])
        if 'c' in lang:
            if 'd_c' in instance.keys():
                if lang.index('c') != 0:
                    temp['di'] += [2]
                    temp['di_tw'] += [2]
                    index += 1
                    index_tw += 1
                result_ids, result_i2s = get_ids(instance['d_c'], hownet_dict, sememe_list, index)
                result_ids_tw, result_i2s_tw = get_ids(instance['w_c'] + [':'] + instance['d_c'], hownet_dict, sememe_list, index_tw)
                temp['di'] += result_ids + [2]
                temp['di_tw'] += result_ids_tw + [2]
                temp['si'] += result_i2s
                temp['si_tw'] += result_i2s_tw
                index += len(temp['di'])
                index_tw += len(temp['di_tw'])
        if 'f' in lang:
            if 'd_f' in instance.keys():
                if lang.index('f') != 0:
                    temp['di'] += [2]
                    temp['di_tw'] += [2]
                    index += 1
                    index_tw += 1
                result_ids, result_i2s = get_ids(instance['d_f'], hownet_dict, sememe_list, index)
                result_ids_tw, result_i2s_tw = get_ids(instance['w_f'] + [':'] + instance['d_f'], hownet_dict, sememe_list, index_tw)
                temp['di'] += result_ids + [2]
                temp['di_tw'] += result_ids_tw + [2]
                temp['si'] += result_i2s
                temp['si_tw'] += result_i2s_tw
                index += len(temp['di'])
                index_tw += len(temp['di_tw'])
        output_data.append(temp)
    
    with open(output_dir + '/data.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False) 
    f.close()

def split_data(data_dir):
    train = json.load(open('./data/train_list.json'))
    valid = json.load(open('./data/valid_list.json'))
    test = json.load(open('./data/test_list.json'))

    train_data = []
    valid_data = []
    test_data = []

    data = json.load(open(data_dir+'/data.json'))
    for instance in data:
        if instance['b'] in train:
            train_data.append(instance)
        elif instance['b'] in valid:
            valid_data.append(instance)
        elif instance['b'] in test:
            test_data.append(instance)
    json.dump(train_data, open(data_dir + '/train_data.json', 'w'))
    json.dump(valid_data, open(data_dir + '/valid_data.json', 'w'))
    json.dump(test_data, open(data_dir + '/test_data.json', 'w'))

def gen_image_tensor(input_data_dir, output_file):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    file_names = os.listdir(input_data_dir)
    babel_images = {}
    for file_name in tqdm(file_names):
        try:
            bn = file_name[:-6]
            input_image = Image.open(babel_images_path+'/'+file_name).convert('RGB')
            input_tensor = preprocess(input_image)
            input_batch = input_tensor.unsqueeze(0)
            if bn not in babel_images.keys():
                babel_images[bn] = input_batch
            else:
                babel_images[bn] = torch.cat((babel_images[bn], input_batch), 0)
        except:
            continue
    json.dump(babel_images, open(output_file,'w',encoding='utf-8'))


        
if __name__ == "__main__":
    # gen_babel_data()
    # data_clean('./data/babel_data.json')
    # gen_training_data('./data/data_all.json', './data/ecf_data', 'ecf')
    gen_image_tensor(babel_images_path, './data/babel_images.json')

