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
        self.fc = torch.nn.Linear(self.hidden_size, self.sememe_number)
        self.loss = torch.nn.MultiLabelSoftMarginLoss()
    
    def forward(self, operation, x=None, y=None, mask=None):
        
        # h: Tensor(batch, sequence_length, hidden_size)
        h = self.encoder(x = x, mask = mask)
        # pos_score: T(batch_size, sequence_length, sememe_number)
        pos_score = self.fc(h)
        mask_3 = mask.to(torch.float32).unsqueeze(2)
        pos_score = pos_score * mask_3 + (-1e7) * (1 - mask_3)
        # score: T(batch_size, sememe_number)
        score, _ = torch.max(pos_score, dim=1)
        _, indices = torch.sort(score, descending=True)
        if operation == 'train':
            loss = self.loss(score, y)
            return loss, score, indices
        elif operation == 'inference':
            return score, indices
        '''
        # h: Tensor(batch, hidden_size)
        h = self.encoder(x = x, mask = mask)
        # pos_score: T(batch_size, sememe_number)
        score = self.fc(h)
        _, indices = torch.sort(score, descending=True)
        if operation == 'train':
            loss = self.loss(score, y)
            return loss, score, indices
        elif operation == 'inference':
            return score, indices
        '''

class ImageEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
    
    def forward(self, x):
        output = model(x)
        return output