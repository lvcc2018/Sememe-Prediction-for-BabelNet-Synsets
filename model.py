import os
import pickle
import torch
from transformers import XLMRobertaModel

from utils.data_utils import *

import torch.nn as nn


class TextForSememePrediction(nn.Module):

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


class MultiSourceForSememePrediction(nn.Module):
    def __init__(self, model, n_labels, text_hidden_size, img_hidden_size, dropout_p):
        super().__init__()
        self.n_labels = n_labels
        self.text_encoder = XLMRobertaModel.from_pretrained(model)
        self.text_pooler_classification_head = nn.Linear(
            text_hidden_size, n_labels)
        self.text_max_classification_head = nn.Linear(
            text_hidden_size, n_labels)
        self.text_pretrain_classification_head = nn.Linear(
            text_hidden_size, n_labels)
        self.img_classification_head = nn.Linear(img_hidden_size, n_labels)
        self.img_encoder_classification_head = nn.Linear(2048, img_hidden_size)
        self.classification_head = nn.Linear(
            text_hidden_size+img_hidden_size, n_labels)
        self.dropout = nn.Dropout(dropout_p)
        self.loss = torch.nn.MultiLabelSoftMarginLoss()

    def forward(self, mode, text_ids, text_mask, img_ids=None, labels=None, mask_idx=None):
        if mode == 'pretrain':
            output = self.text_encoder(
                input_ids=text_ids, attention_mask=text_mask)
            output = output.last_hidden_state
            output = self.dropout(output)
            output = output.gather(1, mask_idx.unsqueeze(1)).squeeze(1)
            output = self.text_pretrain_classification_head(output)
        elif mode == 'train_img':
            output = self.img_classification_head(img_ids)
        elif mode == 'train_text_with_pooler_output':
            output = self.text_encoder(
                input_ids=text_ids, attention_mask=text_mask)
            output = output.pooler_output
            output = self.dropout(output)
            output = self.text_pooler_classification_head(output)
        elif mode == 'train_text_with_last_hidden_state':
            output = self.text_encoder(
                input_ids=text_ids, attention_mask=text_mask)
            output = output.last_hidden_state
            output = self.dropout(output)
            output = self.text_max_classification_head(output)
            mask = text_mask.to(torch.float32).unsqueeze(2)
            output = output * mask + (-1e7) * (1-mask)
            output, _ = torch.max(output, dim=1)
        elif mode == 'train_with_multi_source':
            text_output = self.text_encoder(
                input_ids=text_ids, attention_mask=text_mask)
            text_output = text_output.pooler_output
            output = torch.cat([text_output, img_ids], dim=1)
            output = self.classification_head(output)
        elif mode == 'train_with_multi_source_pro':
            text_output = self.text_encoder(
                input_ids=text_ids, attention_mask=text_mask)
            text_output = text_output.pooler_output
            img_output = self.img_encoder_classification_head(img_ids)
            output = torch.cat([text_output, img_output], dim=1)
            output = self.dropout(output)
            output = self.classification_head(output)
        _, indice = torch.sort(output, descending=True)
        loss = self.loss(output, labels)
        return loss, output, indice
