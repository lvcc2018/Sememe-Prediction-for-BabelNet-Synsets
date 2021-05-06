import json

import numpy as np
import OpenHowNet
import thulac
import torch
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from transformers import XLMRobertaModel, XLMRobertaTokenizer
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

sememe_number = 2187


def get_sememe_label(sememes):
    l = np.zeros(sememe_number, dtype=np.float32)
    for s in sememes:
        l[s] = 1
    return l


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
    result = lac.cut(sentence, text=True)
    result = result.split(' ')
    return result


def get_ids(word_list, tokenizer, hownet_dict, sememe_list, index_offset=0):
    result_ids = []
    result_i2s = []
    idx = index_offset
    for w in word_list:
        idx_list = []
        word_ids = tokenizer(w)['input_ids']
        for i in range(1, len(word_ids)-1):
            idx += 1
            idx_list.append(idx)
            result_ids.append(word_ids[i])
        ids_sememe = hownet_dict.get_sememes_by_word(
            w, structured=False, lang="zh", merge=True)
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
    return result_ids, result_i2s


def preprocess_data(data_dir, synset_image_dic_file, babel_data_file, tokenizer, lang='ecf'):
    sememe_str = open('sememe_all.txt', 'r', encoding='utf-8').read()
    sememe_list = sememe_str.split(' ')
    babel_data = json.load(open(babel_data_file))
    synset_image_dic = json.load(open(synset_image_dic_file))
    print(len(synset_image_dic))
    data_dic = {}
    wnl = WordNetLemmatizer()
    lac = thulac.thulac(T2S=True, seg_only=True)
    hownet_dict = OpenHowNet.HowNetDict()
    for bn in tqdm(babel_data.keys()):
        data = {}
        data['sememes'] = [sememe_list.index(s) for s in [ss.split(
            '|')[1] for ss in babel_data[bn]['sememes']]]
        if len(babel_data[bn]['definition_en']) != 0:
            data['w_e'] = (' | '.join(
                [w.lower() for w in babel_data[bn]['word_en']])).split(' ')
            data['d_e'] = en_lemmatize(
                wnl, babel_data[bn]['definition_en'][0].lower())
        if len(babel_data[bn]['definition_cn']) != 0:
            temp_w_c = [cn_t2s(lac, w)
                        for w in babel_data[bn]['word_cn']]
            data['w_c'] = []
            for i in range(len(temp_w_c)):
                data['w_c'] += temp_w_c[i]
                data['w_c'].append('|')
            if len(data['w_c']) > 0:
                data['w_c'].pop()
            data['d_c'] = cn_t2s(
                lac, babel_data[bn]['definition_cn'][0])
        if len(babel_data[bn]['definition_fr']) != 0:
            data['w_f'] = (' | '.join(
                [w.lower() for w in babel_data[bn]['word_fr']])).split(' ')
            data['d_f'] = babel_data[bn]['definition_fr'][0].lower().split(' ')
        data['di'] = [0]
        data['di_tw'] = [0]
        data['si'] = []
        data['si_tw'] = []
        index = 0
        index_tw = 0
        if 'e' in lang:
            if 'd_e' in data.keys():
                result_ids, result_i2s = get_ids(
                    data['d_e'], tokenizer, hownet_dict, sememe_list, index)
                result_ids_tw, result_i2s_tw = get_ids(
                    data['w_e'] + [':'] + data['d_e'], tokenizer, hownet_dict, sememe_list, index_tw)
                data['di'] += result_ids + [2]
                data['di_tw'] += result_ids_tw + [2]
                data['si'] += result_i2s
                data['si_tw'] += result_i2s_tw
                index = len(data['di'])
                index_tw = len(data['di_tw'])
        if 'c' in lang:
            if 'd_c' in data.keys():
                if lang.index('c') != 0:
                    data['di'] += [2]
                    data['di_tw'] += [2]
                    index += 1
                    index_tw += 1
                result_ids, result_i2s = get_ids(
                    data['d_c'], tokenizer, hownet_dict, sememe_list, index)
                result_ids_tw, result_i2s_tw = get_ids(
                    data['w_c'] + [':'] + data['d_c'], tokenizer, hownet_dict, sememe_list, index_tw)
                data['di'] += result_ids + [2]
                data['di_tw'] += result_ids_tw + [2]
                data['si'] += result_i2s
                data['si_tw'] += result_i2s_tw
                index = len(data['di'])
                index_tw = len(data['di_tw'])
        if 'f' in lang:
            if 'd_f' in data.keys():
                if lang.index('f') != 0:
                    data['di'] += [2]
                    data['di_tw'] += [2]
                    index += 1
                    index_tw += 1
                result_ids, result_i2s = get_ids(
                    data['d_f'], tokenizer, hownet_dict, sememe_list, index)
                result_ids_tw, result_i2s_tw = get_ids(
                    data['w_f'] + [':'] + data['d_f'], tokenizer, hownet_dict, sememe_list, index_tw)
                data['di'] += result_ids + [2]
                data['di_tw'] += result_ids_tw + [2]
                data['si'] += result_i2s
                data['si_tw'] += result_i2s_tw
        if len(data['di']) > 512:
            data['di'] = data['di'][:510] + [2]
            temp = 0
            for i in range(len(data['si'])):
                for j in data['si'][i][0]:
                    if j > 512 and temp == 0:
                        temp = i
                        break
            if temp != 0:
                data['si'] = data['si'][:temp]
        if len(data['di_tw']) > 512:
            data['di_tw'] = data['di_tw'][:510] + [2]
            temp = 0
            for i in range(len(data['si_tw'])):
                for j in data['si_tw'][i][0]:
                    if j > 512 and temp == 0:
                        temp = i
                        break
            if temp != 0:
                data['si_tw'] = data['si_tw'][:temp]
        if 'd_e' in data.keys():
            data.pop('w_e')
            data.pop('d_e')
        if 'd_c' in data.keys():
            data.pop('w_c')
            data.pop('d_c')
        if 'd_f' in data.keys():
            data.pop('w_f')
            data.pop('d_f')
    json.dump(data_dic, open('./data.json', "w"))


def gen_image_tensor(image_dir, synset_image_dic_file, transform):
    synset_image_dic = json.load(open(synset_image_dic_file))
    for bn in tqdm(synset_image_dic.keys()):
        temp = torch.empty((0, 3, 224, 224))
        for image_file in synset_image_dic[bn]:
            input_image = Image.open(image_dir+'/'+image_file).convert('RGB')
            input_tensor = transform(input_image).unsqueeze(0)
            temp = torch.cat((temp, input_tensor), 0)
        torch.save(temp, '/data2/private/lvchuancheng/babel_tensor/'+bn+'.pt')


class MultiSrcDataset(torch.utils.data.Dataset):
    def __init__(self, babel_data_file, image_tensor_path, synset_list):
        self.babel_data = json.load(open(babel_data_file))
        self.synset_list = json.load(open(data_list))
        self.data_list = []

        for bn in tqdm(self.synset_list):
            data = self.babel_data[bn]
            data['img'] = torch.load(image_tensor_path+bn+'.pt')
            self.data_list.append(data)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]


if __name__ == '__main__':
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225]),
    ])
    preprocess_data('all_data', 'synset_image_dic.json',
                    'babel_data.json', tokenizer)
    gen_image_tensor('/data2/private/lvchuancheng/babel_images',
                     'synset_image_dic.json', transform)
