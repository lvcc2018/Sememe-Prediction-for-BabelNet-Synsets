import json
import os
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm
import argparse
from transformers import XLMRobertaTokenizer, XLMRobertaModel, AdamW



class DefEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = XLMRobertaModel.from_pretrained('xlm-roberta-base', return_dict = True)
    
    def forward(self, x, mask):
        # x: Tensor(batch, sequence_length) float32
        # mask: Tensor(batch, sequence_length) float32
        output = self.encoder(input_ids = x, attention_mask = mask)
        # h: Tensor(batch, sequence_length, hidden_size)
        h = output.last_hidden_state
        # h: Tensor(batch, hidden_size)
        # h = output.pooler_output
        return h

class MSSP(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.sememe_number = args.sememe_number
        self.hidden_size = args.hidden_size
        self.encoder = DefEncoder()
        self.batch_size = args.batch_size
        self.fc = torch.nn.Linear(self.hidden_size, sememe_number)
        self.loss = torch.nn.MultiLabelSoftMarginLoss()
    
    def forward(self, operation, x=None, y=None, mask=None, index=None, index_mask=None):
        h = self.encoder(x = x, mask = mask)
        if operation == 'train':
            pos_score = self.fc(h)
            mask_3 = mask.to(torch.float32).unsqueeze(2)
            pos_score = pos_score * mask_3 + (-1e7) * (1 - mask_3)
            score, _ = torch.max(pos_score, dim=1)
            _, indices = torch.sort(score, descending=True)
            loss = self.loss(score, y)
            return loss, _, indices
        elif operation == 'pretrain':
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
    
    def forward(self, x):
        output = model(x)
        return output