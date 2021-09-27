import os
import pickle
import torch
from tqdm import tqdm
from transformers import XLMRobertaModel, XLMRobertaTokenizer

from utils.data_utils import *

import torch.nn as nn
import torch.nn.functional as F


class MultiModalForSememePrediction(nn.Module):

    def __init__(self, n_labels, hidden_size, dropout_p):
        super().__init__()
        self.n_labels = n_labels
        self.text_encoder = XLMRobertaModel.from_pretrained('xlm-roberta-base')
        self.classification_head = nn.Linear(hidden_size, n_labels)
        self.dropout = nn.Dropout(dropout_p)
        self.loss = torch.nn.MultiLabelSoftMarginLoss()

    def forward(self, input_ids, input_mask, labels=None):
        # batch_size * sequence_length
        output = self.text_encoder(
            input_ids=input_ids, attention_mask=input_mask)
        output = output.last_hidden_state
        output = self.dropout(output)
        output = self.classification_head(output)
        # batch_size * sequence_length * label_num
        mask = input_mask.to(torch.float32).unsqueeze(2)
        output = output * mask + (-1e7) * (1-mask)
        output, _ = torch.max(output, dim=1)
        _, indice = torch.sort(output, descending=True)
        '''
        output = output.pooler_output
        output = F.sigmoid(output)
        output = self.dropout(output)
        output = self.classification_head(output)
        _, indice = torch.sort(output, descending=True)
        '''
        # batch_size * label_num
        if labels != None:
            loss = self.loss(output, labels)
            return loss, indice
        else:
            return output, indice
