import os
import pickle
import torch
from tqdm import tqdm
from transformers import XLMRobertaModel, XLMRobertaTokenizer

from utils.data_utils import *

import torch.nn as nn
import torch.nn.functional as F


class MultiModalForSememePrediction(nn.Module):

    def __init__(self, model, n_labels, hidden_size, dropout_p):
        super().__init__()
        self.n_labels = n_labels
        if model == 'xlm-roberta-base' or model == 'xlm-roberta-large':
            self.text_encoder = XLMRobertaModel.from_pretrained(model)
        else:
            self.text_encoder.load_state_dict(model)
        self.classification_head = nn.Linear(hidden_size, n_labels)
        self.dropout = nn.Dropout(dropout_p)
        self.loss = torch.nn.MultiLabelSoftMarginLoss()

    def forward(self, mode, input_ids, input_mask, labels=None, mask_idx=None):
        if mode == 'train':
            # batch_size * sequence_length
            output = self.text_encoder(
                input_ids=input_ids, attention_mask=input_mask)
            output = output.last_hidden_state
            output = self.dropout(output)
            # batch_size * sequence_length * hidden_size
            output = self.classification_head(output)
            # batch_size * sequence_length * label_num
            mask = input_mask.to(torch.float32).unsqueeze(2)
            output = output * mask + (-1e7) * (1-mask)
            output, _ = torch.max(output, dim=1)
            _, indice = torch.sort(output, descending=True)
            # batch_size * label_num
            if labels != None:
                loss = self.loss(output, labels)
                return loss, output, indice
            else:
                return output, indice
        elif mode == 'pretrain':
            # batch_size * sequence_length
            output = self.text_encoder(
                input_ids=input_ids, attention_mask=input_mask)
            output = output.last_hidden_state
            output = self.dropout(output)
            # batch_size * sequence_length * hidden_size
            output = output.gather(1, mask_idx.unsqueeze(1)).squeeze(1)
            # batch_size * hidden_size
            output = self.classification_head(output)
            # batch_size * label_num
            _, indice = torch.sort(output, descending=True)
            if labels != None:
                loss = self.loss(output, labels)
                return loss, output, indice
            else:
                return output, indice

class ImgForSememePrediction(nn.Module):
    def __init__(self, n_labels, hidden_size):
        super().__init__()
        self.n_labels = n_labels
        self.classification_head = nn.Linear(hidden_size, n_labels)
        self.loss = torch.nn.MultiLabelSoftMarginLoss()
    
    def forward(self, input_ids, labels=None):
        # batch_size * hidden_size
        output = self.classification_head(input_ids)
        # batch_size * label_num
        _, indice = torch.sort(output, descending=True)
        # batch_size * label_num
        if labels != None:
            loss = self.loss(output, labels)
            return loss, output, indice
        else:
            return output, indice