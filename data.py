# coding:utf-8
import json
from nltk.stem import WordNetLemmatizer
import string
import re
import random
import torch
from tqdm import tqdm
import opencc
from transformers import XLMRobertaTokenizer, XLMRobertaModel, AdamW
import OpenHowNet
import thulac

babel_data_file_fr = '/data2/private/yiyuan/data/babel_indice/babel/BabelNet-API-4.0.1/babel_synset_list_lcc.txt'

# 带义原标注的synset数据文件
synset_with_sememe_file = './data/synset_sememes.txt'

# BabelNet 原数据文件
babel_entire_file = '/data2/private/yiyuan/data/babel_indice/babel_entire.txt'

# BabelNet 需要的synset数据
# Uncleaned
babel_data_file = './data/babel_data.json'

# BabelNet 需要的synset数据(fr)
# Uncleaned
babel_data_file = './real_final_data_ex/babel_data.json'

# 中文标点
punc_cn = '＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。'

tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

def read_num(fin):
    try:
        line = fin.readline().strip()
        return eval(line)
    except:
        print(line)

def read_list(fin, **kword):
    line = fin.readline()
    if 'sep' in kword:
            line = line.strip().split('\t')
    else:
            line = line.strip().split()
    try:
            num = eval(line[0])
            line = line[1:]
    except:
            return []
    return line

def read_synset_id(fin):
    num_str = fin.readline()
    if not num_str:
        return None
    while num_str[0:3]!='bn:':
        num_str = fin.readline()
    num_str = num_str.strip()
    return num_str

def read_synset_definition(fin):
    num_str = fin.readline()
    if not num_str:
        return None
    num_str = num_str.strip()
    lemmas_EN = read_list(fin, sep = '\t')
    lemmas_ZH = read_list(fin, sep = '\t')
    lemmas_FR = read_list(fin, sep = '\t')
    glosses_EN = read_list(fin, sep = '\t')
    glosses_ZH = read_list(fin, sep = '\t')
    glosses_FR = read_list(fin, sep = '\t')
    return [num_str, lemmas_EN, lemmas_ZH, lemmas_FR, glosses_EN, glosses_ZH, glosses_FR]

def read_synset_id_sememes(fin, with_noun = True):
    '''
    读取义原列表，同义词义原词典，同义词id
    '''
    synset_dic = {}
    synset_id_list = []
    sememes = []
    while True:
        line = fin.readline()
        if not line:
            return sememes, synset_dic, synset_id_list
        line = line.strip().split()
        synset_id = line[0]
        if not with_noun:
            if synset_id[-1]!='n':
                continue
        synset_sememes = line[1:]
        synset_id_list.append(synset_id)
        if synset_id not in synset_dic:
            synset_dic[synset_id] = {}
        synset_dic[synset_id]['sememes'] = synset_sememes
        for sememe in synset_sememes:
            if sememe not in sememes:
                sememes.append(sememe)

def data_clean(word_str, lg):
    '''
    清洗中文与英文定义
    繁简体转换，词形还原
    '''
    if lg == 'cn':
        pattern = re.compile(r'[^\u4e00-\u9fa5]')
    else:
        pattern = re.compile('[^a-z^A-Z^\s]')
    word_str = re.sub(pattern, '', word_str)
    return word_str

def gen_babel_data():
    '''
    获取BabelNet需要的synset数据
    '''
    fin_ss = open(synset_with_sememe_file,'r',encoding = 'utf-8')
    fin_def = open(babel_data_file_fr,'r',encoding = 'utf-8')
    fout = open(babel_data_file, 'w', encoding = 'utf-8')
    sememes, synset_dic, synset_id_list = read_synset_id_sememes(fin_ss)
    fin_ss.close()
    while True:
        item = read_synset_definition(fin_def)
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

def gen_cleaned_data(synset_dic_json):
    synset_list = []
    synset_dic = json.load(open(synset_dic_json))

    # 英文词干化    
    wnl = WordNetLemmatizer()
    # 中文繁转简
    cc = opencc.OpenCC('t2s')

    # sememe list
    with open('./sememe_all.txt', 'r', encoding='utf-8') as f:
        sememe_str = f.read()
    f.close() 
    sememe_list = sememe_str.split(' ')

    for k in tqdm(synset_dic.keys()):
        synset = {}
        # 如果包含英文定义（实际上都会包含）
        if 'word_en' in synset_dic[k].keys() and 'definition_en' in synset_dic[k].keys():
            if len(synset_dic[k]['definition_en']) != 0:
                synset['w_e'] = [w.lower() for w in synset_dic[k]['word_en']]
                synset['s'] = [ss.split('|')[1] for ss in synset_dic[k]['sememes']]
                synset['d_e'] = wnl.lemmatize(data_clean(synset_dic[k]['definition_en'][0],'en')).lower()
        
        # 如果包含中文定义
        if 'word_cn' in synset_dic[k].keys() and 'definition_cn' in synset_dic[k].keys():
            if len(synset_dic[k]['definition_cn']) != 0:
                synset['w_c'] = list(set([cc.convert(data_clean(w,'cn')) for w in synset_dic[k]['word_cn']]))
                synset['d_c'] = cc.convert(data_clean(synset_dic[k]['definition_cn'][0],'cn'))
        
        # 如果包含法文定义
        if 'word_fr' in synset_dic[k].keys() and 'definition_fr' in synset_dic[k].keys():
            if len(synset_dic[k]['definition_fr']) != 0:
                synset['w_f'] = [w.lower() for w in synset_dic[k]['word_fr']]
                synset['d_f'] = synset_dic[k]['definition_fr'][0]

        assert('s' in synset.keys())
        synset['s_i'] = [sememe_list.index(ss) for ss in synset['s']]
        '''
        if 'w_c' in synset.keys() and 'w_e' in synset.keys():
            synset_list_ce.append(synset)
        elif 'w_e' in synset.keys():
            synset_list_e.append(synset)
        '''
        synset_list.append(synset)
    '''
    with open('./data/data_e.json', 'w', encoding='utf-8') as f:
        json.dump(synset_list_e, f, ensure_ascii=False) 
    f.close()

    with open('./data/data_ce.json', 'w', encoding='utf-8') as f:
        json.dump(synset_list_ce, f, ensure_ascii=False) 
    f.close()
    '''
    with open('./real_final_data_ex/data_all.json', 'w', encoding='utf-8') as f:
        json.dump(synset_list, f, ensure_ascii=False) 
    f.close()

def split_data(data_file, save_path):
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

def gen_tlm_data():
    seg_thulac = thulac.thulac(seg_only=True, T2S=True)
    with open('./sememe_all.txt', 'r', encoding='utf-8') as f:
        sememe_str = f.read()
    f.close() 
    sememe_list = sememe_str.split(' ')
    data = json.load(open('./real_final_data_ex/data_all.json'))
    tlm_data = []
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    hownet_dict = OpenHowNet.HowNetDict()
    for instance in tqdm(data):
        tlm_instance = {}
        # tlm_instance['w_e'] = instance['w_e'].lower()
        # tlm_instance['w_c'] = instance['w_c']
        # tlm_instance['d_e'] = instance['d_e'].lower()
        # tlm_instance['d_c'] = instance['d_c']
        tlm_instance['s_i'] = instance['s_i']
        tlm_instance['d_i'] = []
        tlm_instance['d_i_tw'] = []
        tlm_instance['i2s'] = []
        tlm_instance['i2s_tw'] = []
        
        # def_en = (instance['w_e']+':'+instance['d_e']).lower().split(' ')
        # def_cn = seg_thulac.cut(instance['w_c']+':'+instance['d_c'], text = True).split(' ')
        
        def_en = instance['d_e'].lower().split(' ')
        def_cn = seg_thulac.cut(instance['d_c'], text = True).split(' ')

        assert(isinstance(def_cn, list))
        assert(isinstance(def_en, list))

        idx = 0
        tlm_instance['d_i'].append(0)

        for w in def_en:
            idx_list = []
            word_ids = tokenizer(w)['input_ids']
            for i in range(1, len(word_ids)-1):
                idx += 1
                idx_list.append(idx)
                tlm_instance['d_i'].append(word_ids[i])
            ids_sememe = hownet_dict.get_sememes_by_word(w,structured=False,lang="zh",merge=True)
            # 如果有义原标注
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
                    tlm_instance['i2s'].append([idx_list,temp])
        tlm_instance['d_i'] += [2,2]
        idx += 2

        for w in def_cn:
            idx_list = []
            word_ids = tokenizer(w)['input_ids']
            for i in range(1, len(word_ids)-1):
                idx += 1
                idx_list.append(idx)
                tlm_instance['d_i'].append(word_ids[i])
            ids_sememe = hownet_dict.get_sememes_by_word(w,structured=False,lang="zh",merge=True)
            if ids_sememe:
                temp = []
                for s in ids_sememe:
                    if s in sememe_list:
                        temp.append(sememe_list.index(s))
                if temp:
                    tlm_instance['i2s'].append([idx_list,temp])
        tlm_instance['d_i'].append(2)

        tlm_data.append(tlm_instance)
        
    fout = open('./tlm_data/data_wordpiece_notw.json', 'w', encoding = 'utf-8')
    json.dump(tlm_data, fout, ensure_ascii=False)

def gen_tlm_ids():
    with open('./sememe_all.txt', 'r', encoding='utf-8') as f:
        sememe_str = f.read()
    f.close() 
    sememe_list = sememe_str.split(' ')
    tlm_data = json.load(open('./tlm_data/data.json'))
    tlm_ids_data = []
    hownet_dict = OpenHowNet.HowNetDict()
    for instance in tqdm(tlm_data):
        tlm_ids = {}
        tlm_ids['d_i'] = instance['d_i']
        tlm_ids['s_i'] = [sememe_list.index(s) for s in instance['s']]
        # 英文定义的长度
        token_len = tlm_ids['d_i'].index(2)
        #1-token_len, token_len+3-
        ids_sememes = {}
        replacable_ids_en = []
        replacable_ids_cn = []
        # en
        for i in range(1,token_len):
            current_t = instance['t'][i]
            if current_t[0] == '_':
                current_t = current_t[1:]
            ids_sememe = hownet_dict.get_sememes_by_word(current_t,structured=False,lang="zh",merge=True)
            if not ids_sememe:
                continue
            if isinstance(ids_sememe, dict):
                ids_sememe = list(list(ids_sememe.items())[0][1])
            elif isinstance(ids_sememe, set):
                ids_sememe = list(ids_sememe)
            
            temp = []
            for s in ids_sememe:
                if s in sememe_list:
                    temp.append(sememe_list.index(s))
            if not temp:
                continue
            replacable_ids_en.append(i)
            ids_sememes[i] = temp
        # cn
        for i in range(token_len+3, len(instance['t'])-1):
            current_t = instance['t'][i]
            ids_sememe = list(hownet_dict.get_sememes_by_word(current_t,structured=False,lang="zh",merge=True))
            if not ids_sememe:
                continue
            temp = []
            for s in ids_sememe:
                if s in sememe_list:
                    temp.append(sememe_list.index(s))
            if not temp:
                continue
            replacable_ids_cn.append(i)
            ids_sememes[i] = temp

        tlm_ids['i2s'] = ids_sememes
        tlm_ids['i_en'] = replacable_ids_en
        tlm_ids['i_cn'] = replacable_ids_cn

        tlm_ids_data.append(tlm_ids)
    fout = open('./tlm_data/ids_data.json', 'w', encoding = 'utf-8')
    json.dump(tlm_ids_data, fout, ensure_ascii=False)
    
def data_check(index):
    with open('./tlm_data/sememe_all.txt', 'r', encoding='utf-8') as f:
        sememe_str = f.read()
    f.close() 
    sememe_list = sememe_str.split(' ')
    data = json.load(open('./tlm_data/ids_data.json'))
    print(data[index]['d_i'])
    for k in data[index]['i2s'].keys():
        print(k)
        print([sememe_list[i] for i in data[index]['i2s'][k]])

def process_data():
    data = json.load(open('./data/data_ce.json'))
    processed_data = []
    # sememe list
    with open('./tlm_data/sememe_all.txt', 'r', encoding='utf-8') as f:
        sememe_str = f.read()
    f.close() 
    sememe_list = sememe_str.split(' ')
    for instance in tqdm(data):
        temp = {}
        temp['w_e'] = instance['w_e']
        temp['w_c'] = instance['w_c']
        temp['d_e'] = instance['d_e'].lower()
        temp['d_c'] = instance['d_c']
        temp['s'] = [ss.split('|')[1] for ss in instance['s']]
        temp['s_i'] = [sememe_list.index(ss) for ss in temp['s']]
        processed_data.append(temp)
    with open('./data/data_ce_processed.json', 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False) 
    f.close()


if __name__ == "__main__":
    # gen_babel_data()
    gen_cleaned_data('./real_final_data_ex/babel_data.json')
    # split_data('./tlm_data/data_wordpiece.json', './tlm_data_tw')
    # data_check()
    # gen_tlm_data()
    # gen_tlm_ids()
    # split_data('./tlm_data/ids_data.json', './tlm_data')
    # data_check()
    # process_data()
