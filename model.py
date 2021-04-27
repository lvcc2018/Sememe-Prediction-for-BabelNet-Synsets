import argparse
import json
import os

import torch
import torch.utils.data
from torchvision import models, transforms
from transformers import XLMRobertaModel, XLMRobertaTokenizer

sememe_number = 2187


class ImageEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = models.resnet50(pretrained=True)

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

        self.def_encoder = DefEncoder()
        self.img_encoder = ImageEncoder()

        self.pretrain_fc = torch.nn.Linear(self.def_hidden_size, sememe_number)
        self.def_fc = torch.nn.Linear(self.def_hidden_size, sememe_number)
        self.img_fc = torch.nn.Linear(self.img_hidden_size, sememe_number)
        self.ms_fc = torch.nn.Linear(
            self.def_hidden_size+self.img_hidden_size, sememe_number)

        self.loss = torch.nn.MultiLabelSoftMarginLoss()

    def forward(self, operation, device, defin=None, label=None, mask=None, index=None, index_mask=None, image=None):
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
            pos_score = self.pretrain_fc(defin_state)
            index_mask = index_mask.unsqueeze(2)
            pos_score = pos_score * index_mask + (-1e7) * (1 - index_mask)
            score, _ = torch.max(pos_score, dim=1)
            _, indices = torch.sort(score, descending=True)
            loss = self.loss(score, label)
            return loss, score, indices
        elif operation == 'train':
            if image and defin:
                image_state = torch.empty(
                    (0, 1000), dtype=torch.float32, device=device)
                for image_emb in image:
                    image_tensor = self.img_encoder(image_emb)
                    image_tensor = torch.mean(image_tensor, 0, True)
                    image_state = torch.cat((image_state, image_tensor))
                defin_state = self.def_encoder(
                    operation='train', x=defin, mask=mask)
                hidden_state = torch.cat((image_state, defin_state), dim=1)
                pos_score = self.ms_fc(hidden_state)
            elif defin:
                defin_state = self.def_encoder(
                    operation='train', x=defin, mask=mask)
                pos_score = self.def_fc(defin_state)
            elif image:
                image_state = torch.empty(
                    (0, 1000), dtype=torch.float32, device=device)
                for image_emb in image:
                    image_tensor = self.img_encoder(image_emb)
                    image_tensor = torch.mean(image_tensor, 0, True)
                    image_state = torch.cat((image_state, image_tensor))
                pos_score = self.img_fc(image_state)
            _, indices = torch.sort(pos_score, descending=True)
            loss = self.loss(pos_score, label)
            return loss, pos_score, indices
