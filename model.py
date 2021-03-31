import json
import os
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm
import argparse
from transformers import XLMRobertaTokenizer, XLMRobertaModel, AdamW
from PIL import Image
from torchvision import transforms

sememe_number = 2187

def get_sememe_label(sememes):
    l = np.zeros(sememe_number, dtype=np.float32)
    for s in sememes:
        l[s] = 1
    return l

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

class ImageEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
        self.fc = torch.nn.Linear(1000, sememe_number)
        self.loss = torch.nn.MultiLabelSoftMarginLoss()

    def forward(self, x, y):
        output = self.encoder(x)
        pos_score = self.fc(output)
        _, indices = torch.sort(pos_score, descending=True)
        loss = self.loss(pos_score, y)
        return loss, pos_score, indices 

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder):
        self.image_folder = image_folder
        with open('./data/sememe_all.txt', 'r', encoding='utf-8') as f:
            sememe_str = f.read()
        f.close()
        self.sememe_list = sememe_str.split(' ')
        self.synset_dic = {}
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
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
        count = 0

        file_names = os.listdir(self.image_folder)
        self.bn_list = []
        for file_name in file_names:
            bn = file_name[:-4]
            if bn not in self.bn_list:
                self.bn_list.append(bn)
    
    def __len__(self):
        return len(self.bn_list)
    
    def __getitem__(self, index):
        label = torch.tensor(get_sememe_label(self.synset_dic[self.bn_list[index]]), dtype = torch.long)
        input_image = Image.open(self.image_folder+'/'+self.bn_list[index]+'.jpg').convert('RGB')
        input_tensor = self.preprocess(input_image)
        return (label, input_tensor)
