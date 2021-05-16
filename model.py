import argparse
import json
import os

import torch
import torch.utils.data
from torchvision import models, transforms
from torchvision.transforms import *
from PIL import Image
from tqdm import tqdm
from transformers import XLMRobertaModel, XLMRobertaTokenizer

sememe_number = 2187


class ImageEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = models.resnet152(pretrained=True)

    def forward(self, x):
        output = self.encoder(x)
        return output


class DefEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = XLMRobertaModel.from_pretrained(
            'xlm-roberta-base', return_dict=True)

    def forward(self, operation, x, mask):
        output = self.encoder(input_ids=x, attention_mask=mask)
        if operation == 'pretrain':
            h = output.last_hidden_state
        elif operation == 'train':
            h = output.pooler_output
        return h


class MSSP(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.sememe_number = args.sememe_number
        self.def_hidden_size = args.def_hidden_size
        self.img_hidden_size = args.img_hidden_size

        self.img_emb = torch.load('./babel_data/img_emb/img_emb_10_nonorm.pt')
        self.img_emb.requires_grad = False
        self.def_encoder = DefEncoder()

        self.M = torch.nn.Linear(self.img_hidden_size, self.def_hidden_size)
        self.predict_fc = torch.nn.Linear(self.def_hidden_size, sememe_number)

        self.loss = torch.nn.MultiLabelSoftMarginLoss()

    def forward(self, operation, device, defin=None, label=None, mask=None, index=None, index_mask=None, img_idx_list=None, img_idx_mask=None):
        if operation == 'pretrain':
            hidden_state = self.def_encoder(
                operation='pretrain', x=defin, mask=mask)
            defin_state = torch.empty(
                (0, index_mask.shape[1], self.def_hidden_size), dtype=torch.float32, device=device)
            for i in range(index_mask.shape[0]):
                idx_state = torch.empty(
                    (0, self.def_hidden_size), dtype=torch.float32, device=device)
                for j in index[i]:
                    idx_state = torch.cat(
                        (idx_state, hidden_state[i][j].unsqueeze(0)))
                defin_state = torch.cat((defin_state, idx_state.unsqueeze(0)))
            pos_score = self.predict_fc(defin_state)
            index_mask = index_mask.unsqueeze(2)
            pos_score = pos_score * index_mask + (-1e7) * (1 - index_mask)
            score, _ = torch.max(pos_score, dim=1)
            _, indices = torch.sort(score, descending=True)
            loss = self.loss(score, label)
            return loss, score, indices
        elif operation == 'train':
            def_state = self.def_encoder(operation='train', x=defin, mask=mask)
            hidden_state = torch.empty((0, self.def_hidden_size), dtype=torch.float32, device=device)
            img_embedding = torch.index_select(self.img_emb, 0, img_idx_list)
            img_embedding = self.M(img_embedding)
            attention = torch.empty((0, img_embedding.shape[1]), dtype=torch.float32, device=device)
            for i in range(def_state.shape[0]):
                if img_idx_mask[i] == 0:
                    hidden_state = torch.cat((hidden_state, def_state[i].unsqueeze(0)), dim=0)
                else:
                    distance = torch.matmul(img_embedding[i], def_state[i].unsqueeze(1)).squeeze(1)
                    distance = distance[0:img_idx_mask[i]]
                    attention_piece = torch.nn.functional.softmax(distance, dim=0)
                    attention_piece = torch.cat((attention_piece, torch.zeros((img_embedding.shape[1]-img_idx_mask[i]), device = device)))
                    img_state = torch.matmul(attention_piece.unsqueeze(0), img_embedding[i])
                    hidden_state = torch.cat((hidden_state, 0.5 * def_state[i].unsqueeze(0) + 0.5 * img_state), dim=0)
            pos_score = self.predict_fc(hidden_state)
            _, indices = torch.sort(pos_score, descending=True)
            loss = self.loss(pos_score, label)
            return loss, pos_score, indices
            
if __name__ == '__main__':
    device = torch.device('cuda:0')
    synset_image_dic_file = './babel_data/synset_image_dic.json'
    synset_image_dic = json.load(open(synset_image_dic_file))
    img_emb = torch.empty((0, 10, 1000),  dtype=torch.float32, device=device)
    transform = Compose([
        TenCrop(224),
        Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops]))
    ])
    model = ImageEncoder()
    model.to(device)
    with torch.no_grad():
        for k in tqdm(synset_image_dic.keys()):
            temp = torch.empty((0, 1000),  dtype=torch.float32, device=device)
            for img_file in synset_image_dic[k]:
                try:
                    input_image = Image.open('/data/private/lvchuancheng/babel_images_full/'+img_file).convert('RGB')
                    input_tensor = transform(input_image).to(device).unsqueeze(0)
                    output = model(x = input_tensor)
                    temp = torch.cat((temp, output))
                except:
                    continue
            for i in range(100 - temp.shape[0]):
                temp = torch.cat((temp, torch.zeros((1,1000), dtype = torch.float32, device=device)))
            img_emb = torch.cat((img_emb, temp.unsqueeze(0)))
    torch.save(img_emb, './img_emb/img_emb_100.pt')
    print(img_emb.shape)


    
