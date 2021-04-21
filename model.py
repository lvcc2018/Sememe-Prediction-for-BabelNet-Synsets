import argparse
import json
import os

import numpy as np
import OpenHowNet
import thulac
import torch
import torch.utils.data
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm
from transformers import AdamW, XLMRobertaModel, XLMRobertaTokenizer

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
    result = lac.cut(sentence, text = True)
    result = result.split(' ')
    return result

def get_ids(word_list, tokenizer, hownet_dict, sememe_list, index_offset = 0):
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
    return result_ids, result_i2s

class ImageEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.encoder = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
        # self.encoder = torch.hub.load('pytorch/vision:v0.9.0', 'resnet34', pretrained=True)
        self.encoder = models.resnet50(pretrained=True)

    def forward(self, x):
        output = self.encoder(x)
        return output

class DefEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = XLMRobertaModel.from_pretrained('xlm-roberta-base', return_dict = True)
    
    def forward(self, operation, x, mask):
        # x: Tensor(batch, sequence_length) float32
        # mask: Tensor(batch, sequence_length) float32
        output = self.encoder(input_ids = x, attention_mask = mask)
        if operation == 'pretrain':
            # h: Tensor(batch, sequence_length, hidden_size)
            h = output.last_hidden_state
        elif operation == 'train':
            # h: Tensor(batch, hidden_size)
            h = output.pooler_output
            # h = output.last_hidden_state
        return h

class MSSP(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.sememe_number = args.sememe_number
        self.hidden_size = args.hidden_size
        self.def_encoder = DefEncoder()
        self.img_encoder = ImageEncoder()
        self.batch_size = args.batch_size
        self.fc = torch.nn.Linear(self.hidden_size, sememe_number)
        self.loss = torch.nn.MultiLabelSoftMarginLoss()
    
    def forward(self, operation, device, x=None, y=None, mask=None, index=None, index_mask=None, image=None):
        
        if image:
            image_state = torch.empty((0, 1000), dtype=torch.float32, device=device)
            for image_emb in image:
                image_tensor = self.img_encoder(image_emb)
                image_tensor = torch.mean(image_tensor,0,True)
                # batch_size * 1000
                image_state = torch.cat((image_state, image_tensor))
            if operation == 'train':
                defin_state = self.def_encoder(operation = 'train', x = x, mask = mask)
                hidden_state = torch.cat((image_state, defin_state), dim=1)
                pos_score = self.fc(hidden_state)
                _, indices = torch.sort(pos_score, descending=True)
                loss = self.loss(pos_score, y)
                return loss, pos_score, indices 
            elif operation == 'pretrain':
                # TODO:预训练时定义Embedding是batch_size*sequence_length*hidden_size
                h = self.def_encoder(operation = 'pretrain', x = x, mask = mask)
                piece_state = torch.empty((0, index_mask.shape[1], 768), dtype=torch.float32, device=device)
                for i in range(index_mask.shape[0]):
                    idx_state = torch.empty((0, 768), dtype = torch.float32, device=device)
                    for j in index[i]:
                        idx_state = torch.cat((idx_state, h[i][j].unsqueeze(0)))
                    piece_state = torch.cat((piece_state, idx_state.unsqueeze(0)))
                pos_score = self.fc(piece_state)
                mask_3 = index_mask.to(torch.float32).unsqueeze(2)
                pos_score = pos_score * mask_3 + (-1e7) * (1 - mask_3)
                score, _ = torch.max(pos_score, dim=1)
                _, indices = torch.sort(score, descending=True)
                loss = self.loss(score, y)
                return loss, score, indices
        else:
            if operation == 'train':
                h = self.def_encoder(operation = 'train', x = x, mask = mask)
                pos_score = self.fc(h)
                mask_3 = mask.to(torch.float32).unsqueeze(2)
                pos_score = pos_score * mask_3 + (-1e7) * (1 - mask_3)
                score, _ = torch.max(pos_score, dim=1)
                _, indices = torch.sort(score, descending=True)
                loss = self.loss(score, y)
                return loss, score, indices 
            elif operation == 'pretrain':
                h = self.def_encoder(operation = 'pretrain', x = x, mask = mask)
                piece_state = torch.empty((0, index_mask.shape[1], 768), dtype=torch.float32, device=device)
                for i in range(index_mask.shape[0]):
                    idx_state = torch.empty((0, 768), dtype = torch.float32, device=device)
                    for j in index[i]:
                        idx_state = torch.cat((idx_state, h[i][j].unsqueeze(0)))
                    piece_state = torch.cat((piece_state, idx_state.unsqueeze(0)))
                pos_score = self.fc(piece_state)
                mask_3 = index_mask.to(torch.float32).unsqueeze(2)
                pos_score = pos_score * mask_3 + (-1e7) * (1 - mask_3)
                score, _ = torch.max(pos_score, dim=1)
                _, indices = torch.sort(score, descending=True)
                loss = self.loss(score, y)
                return loss, score, indices

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, image_folder, transform):
        self.image_folder = image_folder
        with open('./data/sememe_all.txt', 'r', encoding='utf-8') as f:
            sememe_str = f.read()
        f.close()
        self.sememe_list = sememe_str.split(' ')
        self.synset_dic = {}
        self.preprocess = transform
        f = open('./data/synset_sememes.txt','r',encoding = 'utf-8')
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip().split()
            synset_id = line[0]
            synset_sememes = line[1:]
            if synset_id not in self.synset_dic.keys():
                self.synset_dic[synset_id] = []
            self.synset_dic[synset_id] = [self.sememe_list.index(s) for s in [ss.split('|')[1] for ss in synset_sememes]]
        self.file_list = json.load(open(data_list))

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        label = torch.tensor(get_sememe_label(self.synset_dic[self.file_list[index][:-6]]), dtype = torch.long)
        input_image = Image.open(self.image_folder+'/'+self.file_list[index]).convert('RGB')
        input_tensor = self.preprocess(input_image)
        return (label, input_tensor)

class MultiSrcDataset(torch.utils.data.Dataset):
    def __init__(self, synset_image_dic, babel_data, image_folder, tokenizer, transform, lang = 'ecf'):
        # self.synset_list = json.load(open(synset_list))
        # self.image_list = json.load(open(image_list))
        self.synset_image_dic = json.load(open(synset_image_dic))
        self.synset_list = self.synset_image_dic.keys()
        self.babel_data = json.load(open(babel_data))
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.preprocess = transform
        sememe_str = open('./data/sememe_all.txt', 'r', encoding='utf-8').read()
        self.sememe_list = sememe_str.split(' ')
        self.data_list = []
        wnl = WordNetLemmatizer()
        lac = thulac.thulac(T2S=True,seg_only=True)
        hownet_dict = OpenHowNet.HowNetDict()
        for bn in tqdm(self.synset_list):
            data = {}
            data['sememes'] = [self.sememe_list.index(s) for s in [ss.split('|')[1] for ss in self.babel_data[bn]['sememes']]]
            if len(self.babel_data[bn]['definition_en']) != 0:
                data['w_e'] = (' | '.join([w.lower() for w in self.babel_data[bn]['word_en']])).split(' ')
                data['d_e'] = en_lemmatize(wnl, self.babel_data[bn]['definition_en'][0].lower())
            if len(self.babel_data[bn]['definition_cn']) != 0:
                temp_w_c = [cn_t2s(lac,w) for w in self.babel_data[bn]['word_cn']]
                data['w_c'] = []
                for i in range(len(temp_w_c)):
                    data['w_c'] += temp_w_c[i]
                    data['w_c'].append('|')
                if len(data['w_c']) > 0:
                    data['w_c'].pop()
                data['d_c'] = cn_t2s(lac, self.babel_data[bn]['definition_cn'][0])            
            if len(self.babel_data[bn]['definition_fr']) != 0:
                data['w_f'] = (' | '.join([w.lower() for w in self.babel_data[bn]['word_fr']])).split(' ')
                data['d_f'] = self.babel_data[bn]['definition_fr'][0].lower().split(' ')
            data['di'] = [0]
            data['di_tw'] = [0]
            data['si'] = []
            data['si_tw'] = []
            index = 0
            index_tw = 0
            if 'e' in lang:
                if 'd_e' in data.keys():
                    result_ids, result_i2s = get_ids(data['d_e'], tokenizer, hownet_dict, self.sememe_list, index)
                    result_ids_tw, result_i2s_tw = get_ids(data['w_e'] + [':'] + data['d_e'], tokenizer, hownet_dict, self.sememe_list, index_tw)
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
                    result_ids, result_i2s = get_ids(data['d_c'], tokenizer, hownet_dict, self.sememe_list, index)
                    result_ids_tw, result_i2s_tw = get_ids(data['w_c'] + [':'] + data['d_c'], tokenizer, hownet_dict, self.sememe_list, index_tw)
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
                    result_ids, result_i2s = get_ids(data['d_f'], tokenizer, hownet_dict, self.sememe_list, index)
                    result_ids_tw, result_i2s_tw = get_ids(data['w_f'] + [':'] + data['d_f'], tokenizer, hownet_dict, self.sememe_list, index_tw)
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
            data['image_file'] = self.synset_image_dic[bn]
            self.data_list.append(data)
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        
        for image_file in self.data_list[index]['image_file']:
            input_image = Image.open(self.image_folder+'/'+image_file).convert('RGB')
            input_tensor = self.preprocess(input_image).unsqueeze(0)
            if 'image' not in self.data_list[index].keys():
                self.data_list[index]['image'] = input_tensor
            else:
                self.data_list[index]['image'] = torch.cat((self.data_list[index]['image'], input_tensor), 0)
        return self.data_list[index]
        
            
        
        
