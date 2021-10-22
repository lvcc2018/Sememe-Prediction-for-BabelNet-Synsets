import os
import logging

import pickle
from tqdm import tqdm
import OpenHowNet
import torch
from transformers import XLMRobertaTokenizer
from multiprocessing.pool import ThreadPool
import random
import requests
from PIL import Image
from torchvision import transforms

SEMEME_NUM = 1961


class InputSample(object):
    def __init__(self, text, label):
        self.text = text
        self.label = label


class ImgInputFeature(object):
    def __init__(self, input_ids, label_id):
        self.input_ids = input_ids
        self.label_id = label_id


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


class MaskInputFeature(object):
    def __init__(self, input_ids, input_mask, mask_idx, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.mask_idx = mask_idx
        self.label_id = label_id


class MaskDataSet(torch.utils.data.Dataset):
    def __init__(self, feature_list):
        self.feature_list = feature_list

    def __len__(self):
        return len(self.feature_list)

    def __getitem__(self, index):
        features = self.feature_list[index]
        feature = features[random.randint(0, len(features)-1)]
        return feature


class DataProcesser(object):

    def __init__(self, babel_data_file, sememe_idx_file, tokenizer):
        self.babel_data = self.__read_file(babel_data_file)
        self.sememe_idx = self.__read_file(sememe_idx_file)
        self.tokenizer = tokenizer
        self.idx_sememe = {self.sememe_idx[i]                           : i for i in self.sememe_idx.keys()}
        self.hownet_dict = OpenHowNet.HowNetDict()

    def __read_file(self, file_name):
        data = pickle.load(open(file_name, 'rb'))
        return data

    def get_sememe_idx(self):
        return self.sememe_idx

    def get_babel_data(self):
        return self.babel_data

    def __get_text(self, synset, lang='e', gloss=True, word=False):
        text = []
        if word:
            if len(synset['w_' + lang]) > 0:
                text += ' | '.join(synset['w_'+lang]).split(' ')
        if gloss:
            if len(synset['d_'+lang+'_m']) > 0:
                text += [':'] + synset['d_'+lang+'_m']
            elif len(synset['d_'+lang]) > 0:
                text += [':'] + synset['d_'+lang][0]
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

    def __convert_sample_to_mask_feature_list(self, sample):
        feature_list = []
        ids = ['<s>']
        word_idx = 0
        token_idx = 1
        word_idx_token_idx_dic = {}
        text_merged = []
        for t in sample.text:
            text_merged += t
            for i in t:
                token = self.tokenizer.tokenize(i)
                ids += token
                word_idx_token_idx_dic[word_idx] = [
                    token_idx + j for j in range(len(token))]
                word_idx += 1
                token_idx += len(token)
            ids += ['</s>', '</s>']
            token_idx += 2
        if ids[-2] == '</s>':
            ids = ids[:-1]
        for i in range(len(text_merged)):
            if text_merged[i] != '|':
                sememe_idx = self.__get_sememes_by_word(text_merged[i])
                if len(sememe_idx) > 0:
                    ids_instance = []
                    flag = 0
                    mask_idx = -1
                    for j in range(len(ids)):
                        if j not in word_idx_token_idx_dic[i]:
                            ids_instance.append(ids[j])
                        elif flag == 0:
                            mask_idx = j
                            ids_instance.append('<mask>')
                            flag = 1
                    ids_instance = self.tokenizer.convert_tokens_to_ids(
                        ids_instance)
                    ids_mask = [1]*len(ids_instance)
                    if mask_idx != -1:
                        feature = MaskInputFeature(
                            ids_instance, ids_mask, mask_idx, sememe_idx)
                        feature_list.append(feature)
        return feature_list

    def __get_sememes_by_word(self, word):
        try:
            sememes = [i.en_zh for i in self.hownet_dict.get_sememes_by_word(
                word, merge=True)]
            sememe_idx = []
            for i in sememes:
                if i in self.sememe_idx.keys():
                    sememe_idx.append(self.sememe_idx[i])
            return sememe_idx
        except:
            return []

    def __convert_sample_to_feature(self, sample):
        ids = ['<s>']
        for t in sample.text:
            for i in t:
                token = self.tokenizer.tokenize(i)
                ids += token
            ids += ['</s>', '</s>']
        if ids[-2] == '</s>':
            ids = ids[:-1]
        ids = self.tokenizer.convert_tokens_to_ids(ids)
        ids_mask = [1]*len(ids)
        feature = InputFeature(ids, ids_mask, sample.label)
        return feature

    def create_features(self, en_lang=True, zh_lang=False, fr_lang=False, gloss=True, word=False):
        data_file_name = '{}{}{}{}{}data'.format(
            'en_' if en_lang else '', 'zh_' if zh_lang else '', 'fr_' if fr_lang else '', 'ex_' if word else '', 'gloss_' if gloss else '')
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

    def create_pretrain_features(self, en_lang=True, zh_lang=False, gloss=True, word=False):
        data_file_name = '{}{}{}{}pretrain_data'.format(
            'en_' if en_lang else '', 'zh_' if zh_lang else '',  'ex_' if word else '', 'gloss_' if gloss else '')
        if os.path.exists('data/pretrain_feature_data/'+data_file_name):
            print('Loading {}.'.format(data_file_name))
            return self.__read_file('data/pretrain_feature_data/'+data_file_name)
        feature_dict = {}
        for k in tqdm(self.babel_data.keys()):
            sample = self.__create_sample(
                self.babel_data[k], en_lang=en_lang, zh_lang=zh_lang, fr_lang=False, gloss=gloss, word=word)
            feature = self.__convert_sample_to_mask_feature_list(sample)
            if len(feature) == 0:
                continue
            feature_dict[k] = feature
        pickle.dump(feature_dict, open(
            'data/pretrain_feature_data/'+data_file_name, 'wb'))
        return feature_dict

    def create_dataset(self, data_list, en_lang=True, zh_lang=False, fr_lang=False, gloss=True, word=False):
        feature_dict = self.create_features(
            en_lang=en_lang, zh_lang=zh_lang, fr_lang=fr_lang, gloss=gloss, word=word)
        feature_list = [feature_dict[i] for i in data_list]
        return DataSet(feature_list)

    def create_pretrain_dataset(self, data_list, en_lang=True, zh_lang=True, gloss=True, word=False):
        feature_dict = self.create_pretrain_features(
            en_lang=en_lang, zh_lang=zh_lang, gloss=gloss, word=word)
        feature_list = [feature_dict[i] for i in data_list]
        return MaskDataSet(feature_list)

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

    def pretrain_text_collate_fn(self, batch):
        ids = self.__padding([i.input_ids for i in batch], 1)
        masks = self.__padding([i.input_mask for i in batch], 0)
        labels = self.__create_target([i.label_id for i in batch], SEMEME_NUM)
        ids = torch.tensor(ids, dtype=torch.int64)
        masks = torch.tensor(masks, dtype=torch.int64)
        labels = torch.tensor(labels, dtype=torch.int64)
        mask_idx = torch.tensor(
            [[i.mask_idx]*768 for i in batch], dtype=torch.int64)
        return ids, masks, labels, mask_idx

    def create_dataloader(self, dataset, batch_size, shuffle, collate_fn):
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    def convert_ids_to_sample(self, ids, labels):
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        sememes = []
        for i in range(len(labels)):
            if labels[i] == 1:
                sememes.append(self.idx_sememe[i])
        return InputSample(tokens, sememes)


class ImgDataProcesser(object):
    def __init__(self):
        super().__init__()
        self.babel_img_path = '/data2/private/lvchuancheng/imgs/'
        self.babel_data = pickle.load(open('./data/babel_data', 'rb'))
        self.babel_data_list = list(self.babel_data.keys())
        self.sememe_idx = pickle.load(open('./data/sememe_idx', 'rb'))
        self.img_list = os.listdir(self.babel_img_path)
        self.data_preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])

    def download_imgs(self, babel_id, urls):
        num = 0
        if len(urls) > 0:
            for u in urls:
                img_name = babel_id + str(num) + '.' + u.split('.')[-1]
                try:
                    r = requests.get(u, timeout=10)
                    if r.status_code == 200:
                        with open(self.babel_img_path+img_name, 'wb') as f:
                            f.write(r.content)
                        num += 1
                        print("download image successfully:{}".format(u))
                        if num == 10:
                            break
                except:
                    continue

    def download_image_thread(self, num_processes, Async=True):
        pool = ThreadPool(processes=num_processes)
        for k in self.babel_data.keys():
            if Async:
                pool.apply_async(func=self.download_imgs, args=(
                    self.babel_data[k]['id'], self.babel_data[k]['i']))  # 异步
            else:
                pool.apply(func=self.download_imgs, args=(
                    self.babel_data[k]['id'], self.babel_data[k]['i']))  # 同步
        pool.close()
        pool.join()

    def create_img_features(self):
        img_feature_dic = {}
        for img_file in tqdm(self.img_list):
            try:
                input_image = Image.open(self.babel_img_path + img_file)
                input_tensor = self.data_preprocess(input_image).unsqueeze(0)
                bn = img_file[:12]
                if bn not in img_feature_dic.keys():
                    img_feature_dic[bn] = ImgInputFeature(torch.empty((0, 3, 224, 224)), [
                                                          self.sememe_idx[s] for s in self.babel_data[bn]['s']])
                img_feature_dic[bn].input_ids = torch.cat(
                    (img_feature_dic[bn].input_ids, input_tensor), dim=0)
            except:
                continue
        pickle.dump(img_feature_dic, open(
            './data/image_feature_data/img_feature_dic', 'wb'))

    def create_img_embeddings(self):
        img_feature_dic = pickle.load(
            open('./data/image_feature_data/img_feature_dic', 'rb'))
        img_embedding_dic = {}
        model = torch.hub.load('pytorch/vision:v0.10.0',
                               'resnet152', pretrained=True)
        model.to('cuda:5')
        with torch.no_grad():
            for k in tqdm(img_feature_dic.keys()):
                input_feature = img_feature_dic[k].input_ids
                input_feature = input_feature.to('cuda:5')
                output_embedding = model(input_feature)
                output_embedding = output_embedding.cpu()
                img_embedding_dic[k] = ImgInputFeature(
                    output_embedding, img_feature_dic[k].label_id)
        pickle.dump(img_embedding_dic, open(
            './data/image_feature_data/img_embedding_dic', 'wb'))

    def create_dataset(self, data_set_list):
        img_feature_dic = pickle.load(
            open('data/image_feature_data/img_embedding_dic', 'rb'))
        img_feature_list = []
        for k in data_set_list:
            for i in range(img_feature_dic[k].input_ids.shape[0]):
                img_feature_list.append(ImgInputFeature(
                    img_feature_dic[k].input_ids[i], img_feature_dic[k].label_id))
        return DataSet(img_feature_list)

    def img_collate_fn(self, batch):
        ids = [i.input_ids.numpy() for i in batch]
        labels = self.__create_target([i.label_id for i in batch], SEMEME_NUM)
        ids = torch.tensor(ids)
        labels = torch.tensor(labels, dtype=torch.int64)
        return ids, labels

    def __create_target(self, batch, label_num):
        res = []
        for i in batch:
            temp = [0]*label_num
            for j in i:
                temp[j] = 1
            res.append(temp)
        return res

    def create_dataloader(self, dataset, batch_size, shuffle, collate_fn):
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


class MultiSourceInputFeature(object):
    def __init__(self, text_ids, text_mask, img_ids, label_id):
        self.text_ids = text_ids
        self.text_mask = text_mask
        self.img_ids = img_ids
        self.label_id = label_id


class MultiSourceDataProcesser(object):
    def __init__(self, babel_data_file, sememe_idx_file, tokenizer):
        self.babel_data = self.__read_file(babel_data_file)
        self.sememe_idx = self.__read_file(sememe_idx_file)
        self.tokenizer = tokenizer
        self.hownet_dict = OpenHowNet.HowNetDict()

    def __read_file(self, file_name):
        data = pickle.load(open(file_name, 'rb'))
        return data

    def __get_text(self, synset, lang='e', gloss=True, word=False):
        text = []
        if word:
            if len(synset['w_' + lang]) > 0:
                text += ' | '.join(synset['w_'+lang]).split(' ')
        if gloss:
            if len(synset['d_'+lang+'_m']) > 0:
                text += [':'] + synset['d_'+lang+'_m']
            elif len(synset['d_'+lang]) > 0:
                text += [':'] + synset['d_'+lang][0]
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
            for i in t:
                token = self.tokenizer.tokenize(i)
                ids += token
            ids += ['</s>', '</s>']
        if ids[-2] == '</s>':
            ids = ids[:-1]
        ids = self.tokenizer.convert_tokens_to_ids(ids)
        ids_mask = [1]*len(ids)
        feature = InputFeature(ids, ids_mask, sample.label)
        return feature

    def __convert_sample_to_mask_feature_list(self, sample):
        feature_list = []
        ids = ['<s>']
        word_idx = 0
        token_idx = 1
        word_idx_token_idx_dic = {}
        text_merged = []
        for t in sample.text:
            text_merged += t
            for i in t:
                token = self.tokenizer.tokenize(i)
                ids += token
                word_idx_token_idx_dic[word_idx] = [
                    token_idx + j for j in range(len(token))]
                word_idx += 1
                token_idx += len(token)
            ids += ['</s>', '</s>']
            token_idx += 2
        if ids[-2] == '</s>':
            ids = ids[:-1]
        for i in range(len(text_merged)):
            if text_merged[i] != '|':
                sememe_idx = self.__get_sememes_by_word(text_merged[i])
                if len(sememe_idx) > 0:
                    ids_instance = []
                    flag = 0
                    mask_idx = -1
                    for j in range(len(ids)):
                        if j not in word_idx_token_idx_dic[i]:
                            ids_instance.append(ids[j])
                        elif flag == 0:
                            mask_idx = j
                            ids_instance.append('<mask>')
                            flag = 1
                    ids_instance = self.tokenizer.convert_tokens_to_ids(
                        ids_instance)
                    ids_mask = [1]*len(ids_instance)
                    if mask_idx != -1:
                        feature = MaskInputFeature(
                            ids_instance, ids_mask, mask_idx, sememe_idx)
                        feature_list.append(feature)
        return feature_list

    def __get_sememes_by_word(self, word):
        try:
            sememes = [i.en_zh for i in self.hownet_dict.get_sememes_by_word(
                word, merge=True)]
            sememe_idx = []
            for i in sememes:
                if i in self.sememe_idx.keys():
                    sememe_idx.append(self.sememe_idx[i])
            return sememe_idx
        except:
            return []

    def create_text_features(self, en_lang=True, zh_lang=False, fr_lang=False, gloss=True, word=False):
        data_file_name = '{}{}{}{}{}data'.format(
            'en_' if en_lang else '', 'zh_' if zh_lang else '', 'fr_' if fr_lang else '', 'ex_' if word else '', 'gloss_' if gloss else '')
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

    def create_pretrain_features(self, en_lang=True, zh_lang=False, gloss=True, word=False):
        data_file_name = '{}{}{}{}pretrain_data'.format(
            'en_' if en_lang else '', 'zh_' if zh_lang else '',  'ex_' if word else '', 'gloss_' if gloss else '')
        if os.path.exists('data/pretrain_feature_data/'+data_file_name):
            print('Loading {}.'.format(data_file_name))
            return self.__read_file('data/pretrain_feature_data/'+data_file_name)
        feature_dict = {}
        for k in tqdm(self.babel_data.keys()):
            sample = self.__create_sample(
                self.babel_data[k], en_lang=en_lang, zh_lang=zh_lang, fr_lang=False, gloss=gloss, word=word)
            feature = self.__convert_sample_to_mask_feature_list(sample)
            if len(feature) == 0:
                continue
            feature_dict[k] = feature
        pickle.dump(feature_dict, open(
            'data/pretrain_feature_data/'+data_file_name, 'wb'))
        return feature_dict

    def create_pretrain_dataset(self, data_list, en_lang=True, zh_lang=True, gloss=True, word=False):
        feature_dict = self.create_pretrain_features(
            en_lang=en_lang, zh_lang=zh_lang, gloss=gloss, word=word)
        feature_list = []
        for i in data_list:
            if i in feature_dict.keys():
                feature_list.append(feature_dict[i])
        return MaskDataSet(feature_list)

    def create_multi_source_features(self, en_lang=True, zh_lang=False, fr_lang=False, gloss=True, word=False):
        data_file_name = 'img_{}{}{}{}{}data'.format(
            'en_' if en_lang else '', 'zh_' if zh_lang else '', 'fr_' if fr_lang else '', 'ex_' if word else '', 'gloss_' if gloss else '')
        if os.path.exists('data/feature_data/'+data_file_name):
            return self.__read_file('data/feature_data/'+data_file_name)

        text_feature_dic = self.create_text_features(
            en_lang=en_lang, zh_lang=zh_lang, fr_lang=fr_lang, gloss=gloss, word=word)
        feature_dict = {}
        img_feature_dic = pickle.load(
            open('data/image_feature_data/img_embedding_dic', 'rb'))
        for k in tqdm(self.babel_data.keys()):
            if k in img_feature_dic.keys():
                assert(img_feature_dic[k].label_id == text_feature_dic[k].label_id)
                img_feature = img_feature_dic[k].input_ids
            else:
                img_feature = None
            feature_dict[k] = MultiSourceInputFeature(
                text_feature_dic[k].input_ids, text_feature_dic[k].input_mask, img_feature, text_feature_dic[k].label_id)
        pickle.dump(feature_dict, open(
            'data/multi_source_feature_data/'+data_file_name, 'wb'))

    def create_dataset(self, feature_dic, data_set_list):
        feature_dic = self.__read_file(
            'data/multi_source_feature_data/'+feature_dic)
        feature_list = [feature_dic[i] for i in data_set_list]
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

    def ms_collate_fn(self, batch):
        text_ids = self.__padding([i.text_ids for i in batch], 1)
        text_mask = self.__padding([i.text_mask for i in batch], 0)
        labels = self.__create_target([i.label_id for i in batch], SEMEME_NUM)
        img_ids = [torch.mean(i.img_ids, dim=0).numpy() for i in batch]

        text_ids = torch.tensor(text_ids, dtype=torch.int64)
        text_mask = torch.tensor(text_mask, dtype=torch.int64)
        labels = torch.tensor(labels, dtype=torch.int64)
        img_ids = torch.tensor(img_ids)
        return text_ids, text_mask, img_ids, labels

    def pretrain_text_collate_fn(self, batch):
        ids = self.__padding([i.input_ids for i in batch], 1)
        masks = self.__padding([i.input_mask for i in batch], 0)
        labels = self.__create_target([i.label_id for i in batch], SEMEME_NUM)
        ids = torch.tensor(ids, dtype=torch.int64)
        masks = torch.tensor(masks, dtype=torch.int64)
        labels = torch.tensor(labels, dtype=torch.int64)
        mask_idx = torch.tensor(
            [[i.mask_idx]*768 for i in batch], dtype=torch.int64)
        return ids, masks, labels, mask_idx

    def create_dataloader(self, dataset, batch_size, shuffle, collate_fn):
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


if __name__ == '__main__':
    pass
