import json
import os
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm
import argparse
from transformers import XLMRobertaTokenizer, XLMRobertaModel, AdamW
from PIL import Image
from torchvision import transforms, models


sememe_number = 2187

def get_sememe_label(sememes):
    l = np.zeros(sememe_number, dtype=np.float32)
    for s in sememes:
        l[s] = 1
    return l

class ImageEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.encoder = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
        # self.encoder = torch.hub.load('pytorch/vision:v0.9.0', 'resnet34', pretrained=True)
        self.encoder = models.resnet50(pretrained=True)
        self.fc = torch.nn.Linear(1000, sememe_number)
        self.loss = torch.nn.MultiLabelSoftMarginLoss()

    def forward(self, x, y):
        output = self.encoder(x)
        pos_score = self.fc(output)
        _, indices = torch.sort(pos_score, descending=True)
        loss = self.loss(pos_score, y)
        return loss, pos_score, indices 

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
        return h

class MSSP(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.sememe_number = args.sememe_number
        self.hidden_size = args.hidden_size
        self.def_encoder = DefEncoder()
        self.batch_size = args.batch_size
        self.synset2idx = json.load(open('./data/synset2idx.json'))
        self.fc = torch.nn.Linear(self.hidden_size, sememe_number)
        self.loss = torch.nn.MultiLabelSoftMarginLoss()
    
    def forward(self, operation, x=None, y=None, mask=None, index=None, index_mask=None):
        if operation == 'train':
            h = self.def_encoder(operation = 'train', x = x, mask = mask)
            pos_score = self.fc(h)
            _, indices = torch.sort(pos_score, descending=True)
            loss = self.loss(pos_score, y)
            return loss, pos_score, indices 
        elif operation == 'pretrain':
            h = self.def_encoder(operation = 'pretrain', x = x, mask = mask)
            piece_state = torch.empty((0, index_mask.shape[1], self.hidden_size), dtype=torch.float32, device=device)
            for i in range(index_mask.shape[0]):
                idx_state = torch.empty((0, self.hidden_size), dtype = torch.float32, device=device)
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

        file_names = os.listdir(self.image_folder)
        self.file_list = json.load(open(data_list))

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        label = torch.tensor(get_sememe_label(self.synset_dic[self.file_list[index][:-6]]), dtype = torch.long)
        input_image = Image.open(self.image_folder+'/'+self.file_list[index]).convert('RGB')
        input_tensor = self.preprocess(input_image)
        return (label, input_tensor)

class MultiSrcDataset(torch.utils.data.Dataset):
    def __init__(self, sememe_list, synset_list, image_list, babel_data, image_folder, tokenizer, transform):
        self.synset_list = json.load(open(synset_list))
        self.image_list = json.load(open(image_list))
        self.babel_data = json.load(open(babel_data))
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.preprocess = transform
        sememe_str = open(sememe_list, 'r', encoding='utf-8').read()
        self.sememe_list = sememe_str.split(' ')

        self.data_list = []
        for bn in synset_list:
            
        
        
        
