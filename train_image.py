import json
import os
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm
import argparse
from model import *
import random

device = torch.device('cuda:0')
sememe_number = 2187


def evaluate(ground_truth, prediction, score, threshold):
    index = 1
    correct = 0
    point = 0
    truth_sum = 0
    for predicted_sememe in prediction:
        if ground_truth[predicted_sememe] == 1:
            correct += 1
            point += (correct / index)
        index += 1
    point /= sum(ground_truth)
    real_prediction = []
    for i in range(len(score)):
        if score[i]>threshold:
            real_prediction.append(prediction[i])
    prediction = real_prediction
    if len(list(set(prediction) & set(ground_truth))) == 0:
        f1 = 0
    else:
        recall = len(list(set(prediction) & set(ground_truth))) / len(ground_truth)
        precision = len(list(set(prediction) & set(ground_truth))) / len(prediction)
        f1 = 2*recall*precision/(recall + precision)
    return point, f1

def train(args):
    with open('./data/sememe_all.txt', 'r', encoding='utf-8') as f:
        sememe_str = f.read()
    f.close()
    index_sememe = sememe_str.split(' ')

    model = ImageEncoder()
    model.to(device)
    
    sparse_parameters = [para for name, para in model.fc.named_parameters() if para.requires_grad]
    encoder_parameters = [para for name, para in model.encoder.named_parameters() if para.requires_grad]
    optimizer = torch.optim.Adam(sparse_parameters, lr=args.classifier_lr, weight_decay = 1e-5)
    encoder_optimizer = torch.optim.Adam(encoder_parameters, lr=args.pretrain_model_lr, weight_decay = 1e-5)
    
    max_valid_map = 0
    max_valid_epoch = 0
    max_valid_f1 = 0
    transform = {
        'train':transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'test':transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    }
    
    datasets = {
        'train':ImageDataset('./image_data/real_train_data.json','/data2/private/lvchuancheng/babel_images',transform['train']),
        'val':ImageDataset('./image_data/real_valid_data.json','/data2/private/lvchuancheng/babel_images',transform['test']),
        'test':ImageDataset('./image_data/real_test_data.json','/data2/private/lvchuancheng/babel_images',transform['test'])
    }
    train_data_num, valid_data_num, test_data_num = [datasets[x].__len__() for x in ['train', 'val', 'test']]
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=8) for x in ['train', 'val', 'test']}


    for epoch in range(args.epoch_num):
        print('Train epoch ', epoch)
        
        model.train()
        train_map = 0
        train_loss = 0
        train_f1 = 0

        for labels, inputs in tqdm(dataloaders['train']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            encoder_optimizer.zero_grad()
            loss, score, indices = model(x=inputs, y=labels)
            loss.backward()
            optimizer.step()
            encoder_optimizer.step()
            predicted = indices.detach().cpu().numpy().tolist()
            score = score.detach().cpu().numpy().tolist()
            
            for i in range(len(labels)):
                m, f = evaluate(labels[i], predicted[i], score[i], args.threshold)
                train_map += m
                train_f1 += f
            train_loss += loss.item()
        
        valid_map = 0
        valid_loss = 0
        valid_f1 = 0
        model.eval()
        for labels, inputs in tqdm(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            loss, score, indices = model(x=inputs, y=labels)
            predicted = indices.detach().cpu().numpy().tolist()
            score = score.detach().cpu().numpy().tolist()
            for i in range(len(labels)):
                m, f  = evaluate(labels[i], predicted[i], score[i], args.threshold)
                valid_map += m
                valid_f1 += f
            valid_loss += loss.item()
        print(f'train loss {train_loss / train_data_num}, train map {train_map / train_data_num}, train f1 {train_f1 / train_data_num}')
        print(f'valid loss {valid_loss / valid_data_num}, valid map {valid_map / valid_data_num}, valid f1 {valid_f1 / valid_data_num}')
        if valid_map / valid_data_num > max_valid_map:
            early_stop = 0
            max_valid_epoch = epoch
            max_valid_map = valid_map / valid_data_num
            max_valid_f1 = valid_f1 / valid_data_num
            torch.save(model.state_dict(), os.path.join('output', args.result))
    
    print(f'train max valid map {max_valid_map}, train max valid f1 {max_valid_f1}')
    model.load_state_dict(torch.load(os.path.join('output', args.result)))

    test_map = 0
    test_loss = 0
    test_f1 = 0
    model.eval()
    for labels, inputs in tqdm(dataloaders['test']):
        inputs = inputs.to(device)
        labels = labels.to(device)
        loss, score, indices = model(x=inputs, y=labels)
        score = score.detach().cpu().numpy().tolist()
        predicted = indices.detach().cpu().numpy().tolist()
        for i in range(len(labels)):
            m, f  = evaluate(labels[i], predicted[i], score[i], args.threshold)
            test_map += m
            test_f1 += f
        test_loss += loss.item()
    print(f'test loss {test_loss / test_data_num}, test map {test_map / test_data_num}, test f1 {test_f1 / test_data_num}')
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type = int, default = 32)
    parser.add_argument("--hidden_size", type =int ,default = 768)
    parser.add_argument("--epoch_num", type = int, default = 100)
    parser.add_argument("--result", type = str, default = 'model')
    parser.add_argument("--threshold", type = int, default = -1)
    parser.add_argument("--pretrain_model_lr", type = float, default = 1e-4)
    parser.add_argument("--classifier_lr", type = float, default = 0.01)
    args = parser.parse_args()
    print(args)
    print('Training start...')
    train(args)
    print('Training completed!')