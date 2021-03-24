import json
import os
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm
import argparse
from model import *

tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
device = torch.device('cuda:0')

def evaluate(ground_truth, prediction):
    index = 1
    correct = 0
    point = 0
    for predicted_sememe in prediction:
        if predicted_sememe in ground_truth:
            correct += 1
            point += (correct / index)
        index += 1
    point /= len(ground_truth)
    return point

def build_sentence_numpy(sentences):
    max_length = max([len(sentence) for sentence in sentences])
    sentence_numpy = np.zeros((len(sentences), max_length), dtype=np.int64)
    for i in range(len(sentences)):
        sentence_numpy[i, 0:len(sentences[i])] = np.array(sentences[i])
    return sentence_numpy

def build_mask_numpy(sentences):
    max_length = max([len(sentence) for sentence in sentences])
    mask_numpy = np.zeros((len(sentences), max_length), dtype=np.int64)
    for i in range(len(sentences)):
        mask_numpy[i, 0:len(sentences[i])] = np.ones(len(sentences[i]),dtype=int)
    return mask_numpy

def get_sememe_label(sememes):
    l = np.zeros((len(sememes), 1961), dtype=np.float32)
    for i in range(len(sememes)):
        for s in sememes[i]:
            l[i, s] = 1
    return l

def get_def(instance, extended = True):
    if extended:
        return [instance['w_e'] + ':' + instance['d_e'], instance['w_c'] + ':' + instance['d_c']]
    else:
        return [instance['d_e'], instance['d_c']]

def sp_collate_fn(batch):
    sememes = [instance['s_i'] for instance in batch]
    definition_words = [tokenizer(get_def(instance)[0]+"</s>"+get_def(instance)[1])['input_ids'] for instance in batch]
    sememes_t = torch.tensor(get_sememe_label(sememes), dtype=torch.float32, device=device)
    definition_words_t = torch.tensor(build_sentence_numpy(definition_words), dtype=torch.int64, device=device)
    mask_t = torch.tensor(build_mask_numpy(definition_words), dtype = torch.int64, device=device)
    return sememes_t, definition_words_t, sememes, mask_t

def get_dataloader(batch_size, train_data, valid_data, test_data):
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=sp_collate_fn)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True, collate_fn=sp_collate_fn)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=sp_collate_fn)
    return train_dataloader, valid_dataloader, test_dataloader

def load_map_data(file_path):
    index_sememe = json.load(open(file_path))
    return index_sememe


def load_data(data_path):
    train_data = json.load(open(data_path+'train_data.json'))
    valid_data = json.load(open(data_path+'valid_data.json'))
    test_data = json.load(open(data_path+'test_data.json'))
    return train_data, valid_data, test_data

def train(args):
    index_sememe = load_map_data(args.index2sememe)
    train_data, valid_data, test_data = load_data(args.data_path)
    train_dataloader, valid_dataloader, test_dataloader = get_dataloader(args.batch_size, train_data, valid_data, test_data)

    model = MDSP(args)
    model.to(torch.device(args.device))
    
    sparse_parameters = [para for name, para in model.fc.named_parameters() if para.requires_grad]
    encoder_parameters = [para for name, para in model.encoder.named_parameters() if para.requires_grad]
    optimizer = torch.optim.Adam(sparse_parameters, lr=0.001)
    encoder_optimizer = AdamW(encoder_parameters, lr=1e-5)
    
    max_valid_map = 0
    max_valid_epoch = 0
    
    for epoch in range(args.epoch_num):
        print('epoch', epoch)
        train_map = 0
        train_loss = 0
        for sememes_t, definition_words_t, sememes, mask_t in tqdm(train_dataloader):
            optimizer.zero_grad()
            encoder_optimizer.zero_grad()
            loss, score, indices = model('train', x=definition_words_t, y=sememes_t, mask = mask_t)
            loss.backward()
            optimizer.step()
            encoder_optimizer.step()
            predicted = indices.detach().cpu().numpy().tolist()
            for i in range(len(sememes)):
                train_map += evaluate(sememes[i], predicted[i])
            train_loss += loss.item()
        model.eval()
        valid_map = 0
        valid_loss = 0
        for sememes_t, definition_words_t, sememes, mask_t in tqdm(valid_dataloader):
            loss, score, indices = model('train', x=definition_words_t, y=sememes_t, mask=mask_t)
            predicted = indices.detach().cpu().numpy().tolist()
            score = score.detach().cpu().numpy()
            for i in range(len(sememes)):
                m = evaluate(sememes[i], predicted[i])
                valid_map += m
            valid_loss += loss.item()
        print(f'train loss {train_loss / len(train_data)}, train map {train_map / len(train_data)}, valid loss {valid_loss / len(valid_data)}, valid map {valid_map / len(valid_data)}')
        if valid_map / len(valid_data) > max_valid_map:
            max_valid_epoch = epoch
            max_valid_map = valid_map / len(valid_data)
            torch.save(model.state_dict(), os.path.join('output', 'model_xlmr_concrete'))
    model.load_state_dict(torch.load(os.path.join('output', 'model_xlmr_concrete')))
    test_map = 0
    test_loss = 0
    predictions = dict()
    for sememes_t, definition_words_t, sememes, mask_t in tqdm(test_dataloader):
        loss, score, indices = model('train', x=definition_words_t, y=sememes_t, mask = mask_t)
        score = score.detach().cpu().numpy().tolist()
        predicted = indices.detach().cpu().numpy().tolist()
        for i in range(len(sememes)):
            test_map += evaluate(sememes[i], predicted[i])
        test_loss += loss.item()
    print(f'test loss {test_loss / len(test_data)}, test map {test_map / len(test_data)}')
    

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--sememe_number", type = int, default = 1961)
    parser.add_argument("--data_path", type = str, default = './data/')
    parser.add_argument("--index2sememe", type = str, default = './data/index2sememe.json')
    parser.add_argument("--batch_size", type = int, default = 8)
    parser.add_argument("--hidden_size", type =int ,default = 768)
    parser.add_argument("--epoch_num", type = int, default = 10)
    parser.add_argument("--device", type = str, default = 'cuda:2')
    parser.add_argument("--encoder", type = str, default = 'XLMR')
    args = parser.parse_args()
    print(args)
    print('Training start...')
    train(args)
    print('Training completed!')