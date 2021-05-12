import argparse
import json
import os

import torch
import torch.utils.data
from torchvision import models, transforms
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

        self.img_emb = torch.load('./img_emb_10_modified.pt')
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
            img_embedding = torch.index_select(self.img_emb, 0, img_idx_list)
            img_embedding = self.M(img_embedding)
            attention = torch.nn.functional.softmax(torch.matmul(img_embedding, def_state.unsqueeze(2)).squeeze(2)* img_idx_mask, dim=1)
            img_state = torch.matmul(attention.unsqueeze(1), img_embedding).squeeze(1)
            # hidden_state = torch.empty((img_state.shape[0], self.def_hidden_size), dtype=torch.float32, device=device)
            hidden_state = 0.5 * def_state + 0.5 * img_state
            # hidden_state = def_state
            pos_score = self.predict_fc(hidden_state)
            _, indices = torch.sort(pos_score, descending=True)
            loss = self.loss(pos_score, label)
            return loss, pos_score, indices
            
if __name__ == '__main__':
    device = torch.device('cuda:0')
    synset_image_dic_file = 'synset_image_dic_10.json'
    synset_image_dic = json.load(open(synset_image_dic_file))
    img_emb = torch.empty((0, 10, 1000),  dtype=torch.float32, device=device)
    transform = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
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
            for i in range(10 - temp.shape[0]):
                temp = torch.cat((temp, torch.zeros((1,1000), dtype = torch.float32, device=device)))
            img_emb = torch.cat((img_emb, temp.unsqueeze(0)))
    torch.save(img_emb, './img_emb_10_nonorm.pt')
    print(img_emb.shape)




    
