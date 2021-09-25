import os
import logging

import pickle
from tqdm import tqdm

import torch

SEMEME_NUM = 1961

class InputSample(object):
    def __init__(self, text, label):
        self.text = text
        self.label = label


class InputFeature(object):
    def __init__(self, input_ids, input_mask, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_id = label_id


class DataSet(torch.utils.data.Dataset):
    def __init__(self, feature_list):
        self.feature_list = feature_list

    def __len__(self):
        return len(self.feature_list)

    def __getitem__(self, index):
        return self.feature_list[index]


class DataProcesser(object):

    def __init__(self, babel_data_file, sememe_idx_file, tokenizer):
        self.babel_data = self.__read_file(babel_data_file)
        self.sememe_idx = self.__read_file(sememe_idx_file)
        self.tokenizer = tokenizer
        self.idx_sememe = {self.sememe_idx[i]:i for i in self.sememe_idx.keys()}

    def __read_file(self, file_name):
        data = pickle.load(open(file_name, 'rb'))
        return data

    def get_sememe_idx(self):
        return self.sememe_idx

    def get_babel_data(self):
        return self.babel_data

    def __get_text(self, synset, lang='e', gloss=True, word=False):
        text = ''
        if word:
            if len(synset['w_' + lang]) > 0:
                text += '|'.join(synset['w_'+lang])
        if gloss:
            if len(synset['d_'+lang+'_m']) > 0:
                text += ':' + synset['d_'+lang+'_m']
            elif len(synset['d_'+lang]) > 0:
                text += ':' + synset['d_'+lang][0]
        if len(text) > 0 and text[0] == ':':
            text = text[1:]
        return text

    def __create_sample(self, synset, en_lang=True, zh_lang=False, fr_lang=False, gloss=True, word=False):
        text = []
        if en_lang:
            text.append(self.__get_text(
                synset, lang='e', gloss=gloss, word=word))
        if zh_lang:
            text.append(self.__get_text(
                synset, lang='c', gloss=gloss, word=word))
        if fr_lang:
            text.append(self.__get_text(
                synset, lang='f', gloss=gloss, word=word))
        label = [self.sememe_idx[i] for i in synset['s']]
        sample = InputSample(text, label)
        return sample

    def __convert_sample_to_feature(self, sample):
        ids = ['<s>']
        for t in sample.text:
            tokens = self.tokenizer.tokenize(t)
            ids += tokens
            ids += ['</s>', '</s>']
        if ids[-2] == '</s>':
            ids = ids[:-1]
        ids = self.tokenizer.convert_tokens_to_ids(ids)
        ids_mask = [1]*len(ids)
        feature = InputFeature(ids, ids_mask, sample.label)
        return feature

    def create_features(self, en_lang=True, zh_lang=False, fr_lang=False, gloss=True, word=False):
        data_file_name = '{}{}{}{}data'.format(
            'en_' if en_lang else '', 'zh_' if zh_lang else '', 'fr_' if fr_lang else '', 'ex_' if word else '')
        if os.path.exists('data/feature_data/'+data_file_name):
            return self.__read_file('data/feature_data/'+data_file_name)
        feature_dict = {}
        for k in tqdm(self.babel_data.keys()):
            sample = self.__create_sample(
                self.babel_data[k], en_lang=en_lang, zh_lang=zh_lang, fr_lang=fr_lang, gloss=gloss, word=word)
            feature = self.__convert_sample_to_feature(sample)
            feature_dict[k] = feature
        pickle.dump(feature_dict, open(
            'data/feature_data/'+data_file_name, 'wb'))
        return feature_dict

    def create_dataset(self, data_list, en_lang=True, zh_lang=False, fr_lang=False, gloss=True, word=False):
        feature_dict = self.create_features(
            en_lang=en_lang, zh_lang=zh_lang, fr_lang=fr_lang, gloss=gloss, word=word)
        feature_list = [feature_dict[i] for i in data_list]
        return DataSet(feature_list)

    def __padding(self, batch, padding_id):
        max_length = max([len(i) for i in batch])
        res = [i + [padding_id] * (max_length - len(i)) for i in batch]
        return res
    
    def __create_target(self, batch, label_num):
        res = []
        for i in batch:
            temp = [0]*label_num
            for j in i:
                temp[j] = 1
            res.append(temp)
        return res


    def text_collate_fn(self, batch):
        ids = self.__padding([i.input_ids for i in batch], 1)
        masks = self.__padding([i.input_mask for i in batch], 0)
        labels = self.__create_target([i.label_id for i in batch], SEMEME_NUM)
        ids = torch.tensor(ids, dtype=torch.int64)
        masks = torch.tensor(masks, dtype=torch.int64)
        labels = torch.tensor(labels, dtype=torch.int64)
        return ids, masks, labels

    def create_dataloader(self, dataset, batch_size, shuffle, collate_fn):
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    
    def convert_ids_to_sample(self, ids, labels):
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        sememes = []
        for i in range(len(labels)):
            if labels[i] == 1:
                sememes.append(self.idx_sememe[i])
        return InputSample(tokens, sememes)


