import argparse
import json
import os
import random

import numpy as np
import OpenHowNet
import torch
from torchvision import models, transforms
from tqdm import tqdm
from pprint import pprint
from transformers import XLMRobertaModel, XLMRobertaTokenizer

from data import *
from data import MultiSrcDataset
from model import *
from model import MSSP

sememe_number = 2187


def evaluate(ground_truth, prediction, score, threshold):
    index = 1
    correct = 0
    point = 0
    for predicted_sememe in prediction:
        if predicted_sememe in ground_truth:
            correct += 1
            point += (correct / index)
        index += 1
    point /= len(ground_truth)
    real_prediction = []
    for i in range(len(score)):
        if score[i] > threshold:
            real_prediction.append(prediction[i])
    prediction = real_prediction
    if len(list(set(prediction) & set(ground_truth))) == 0:
        f1 = 0
    else:
        recall = len(list(set(prediction) & set(ground_truth))) / \
            len(ground_truth)
        precision = len(list(set(prediction) & set(
            ground_truth))) / len(prediction)
        f1 = 2*recall*precision/(recall + precision)
    return point, f1


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
        mask_numpy[i, 0:len(sentences[i])] = np.ones(
            len(sentences[i]), dtype=int)
    return mask_numpy


def get_sememe_label(sememes):
    l = np.zeros((len(sememes), sememe_number), dtype=np.float32)
    for i in range(len(sememes)):
        for s in sememes[i]:
            l[i, s] = 1
    return l


def get_random_idx(instance):
    return random.choice(instance['si_tw'])


def build_idx(idxs):
    max_length = max([len(idx_list) for idx_list in idxs])
    res = []
    for i in range(len(idxs)):
        temp = idxs[i]
        if len(temp) < max_length:
            temp += [0]*(max_length-len(temp))
        res.append(temp)
    return res


def ms_collate_fn(batch):
    sememes = [instance['sememes'] for instance in batch]
    definition_words = [instance['di_tw'] for instance in batch]
    idx = [get_random_idx(instance) for instance in batch]
    idx_sememes = [instance[1] for instance in idx]
    images = []
    for instance in batch:
        if 'image' in instance.keys():
            images.append(instance['image'])
    return sememes, definition_words, idx, idx_sememes, images


def build_input(sememes, definition_words, idx, idx_sememes, images, device):
    sememes_t = torch.tensor(get_sememe_label(
        sememes), dtype=torch.float32, device=device)
    definition_words_t = torch.tensor(build_sentence_numpy(
        definition_words), dtype=torch.int64, device=device)
    mask_t = torch.tensor(build_mask_numpy(definition_words),
                          dtype=torch.float32, device=device)
    idx_sememes_t = torch.tensor(get_sememe_label(
        idx_sememes), dtype=torch.float32, device=device)
    idx = [instance[0] for instance in idx]
    idx_mask = torch.tensor(build_mask_numpy(
        idx), dtype=torch.float32, device=device)
    idx = build_idx(idx)
    image_tensor = [image.to(device) for image in images]
    return sememes_t, definition_words_t, mask_t, idx_sememes_t, idx, idx_mask, image_tensor


def train(args):
    print('Global initializing...')
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    device = torch.device('cuda:'+str(args.device_id))
    transform = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])
    }

    print("Data initializing...")
    datasets = {x: MultiSrcDataset(
        args.data_path+x+'_synset_image_dic.json', args.data_path+'babel_data.json', args.image_train, args.image_path, tokenizer, transform[x]) for x in ['train', 'valid', 'test']
    }
    dataloaders = {x: torch.utils.data.DataLoader(
        datasets[x], batch_size=args.batch_size, shuffle=True, collate_fn=ms_collate_fn) for x in ['train', 'valid', 'test']
    }

    print("Model initializing...")
    train_data_num, valid_data_num, test_data_num = [
        datasets[x].__len__() for x in ['train', 'valid', 'test']]
    model = MSSP(args)
    model.to(device)

    pretrain_fc_parameters = [
        para for name, para in model.pretrain_fc.named_parameters() if para.requires_grad]
    def_fc_parameters = [
        para for name, para in model.def_fc.named_parameters() if para.requires_grad]
    img_fc_parameters = [
        para for name, para in model.img_fc.named_parameters() if para.requires_grad]
    ms_fc_parameters = [
        para for name, para in model.ms_fc.named_parameters() if para.requires_grad]
    def_encoder_parameters = [
        para for name, para in model.def_encoder.named_parameters() if para.requires_grad]
    img_encoder_parameters = [
        para for name, para in model.img_encoder.named_parameters() if para.requires_grad]

    pretrain_fc_optimizer = torch.optim.Adam(
        pretrain_fc_parameters, lr=args.classifier_lr)
    def_fc_optimizer = torch.optim.Adam(
        def_fc_parameters, lr=args.classifier_lr)
    img_fc_optimizer = torch.optim.Adam(
        img_fc_parameters, lr=args.classifier_lr)
    ms_fc_optimizer = torch.optim.Adam(ms_fc_parameters, lr=args.classifier_lr)
    def_encoder_optimizer = torch.optim.Adam(
        def_encoder_parameters, lr=args.pretrain_model_lr)
    img_encoder_optimizer = torch.optim.Adam(
        img_encoder_parameters, lr=args.pretrain_model_lr)

    max_valid_map = 0
    max_valid_epoch = 0
    max_valid_f1 = 0

    if args.load_model:
        print("Load model "+args.load_model)
        model.load_state_dict(torch.load(
            os.path.join('output', args.load_model)))

    if args.pretrain_epoch_num > 0:
        counter = 0
        for epoch in range(args.pretrain_epoch_num):
            print('Pretrain epoch ', epoch)
            model.train()
            pretrain_map = 0
            pretrain_loss = 0
            pretrain_f1 = 0
            for sememes, definition_words, idx, idx_sememes, images in tqdm(dataloaders['train']):
                sememes_t, definition_words_t, mask_t, idx_sememes_t, idx, idx_mask, image_tensor = build_input(
                    sememes, definition_words, idx, idx_sememes, images, device)
                def_encoder_optimizer.zero_grad()
                pretrain_fc_optimizer.zero_grad()
                loss, score, indices = model('pretrain', device=device, defin=definition_words_t,
                                             label=idx_sememes_t, mask=mask_t, index=idx, index_mask=idx_mask)
                loss.backward()
                def_encoder_optimizer.step()
                pretrain_fc_optimizer.step()
                predicted = indices.detach().cpu().numpy().tolist()
                score = score.detach().cpu().numpy().tolist()
                for i in range(len(idx_sememes)):
                    m, f = evaluate(
                        idx_sememes[i], predicted[i], score[i], args.threshold)
                    pretrain_map += m
                    pretrain_f1 += f
                pretrain_loss += loss.item()
            model.eval()
            prevalid_map = 0
            prevalid_loss = 0
            prevalid_f1 = 0
            for sememes, definition_words, idx, idx_sememes, images in tqdm(dataloaders['valid']):
                sememes_t, definition_words_t, mask_t, idx_sememes_t, idx, idx_mask, image_tensor = build_input(
                    sememes, definition_words, idx, idx_sememes, images, device)
                loss, score, indices = model('pretrain', device=device, defin=definition_words_t,
                                             label=idx_sememes_t, mask=mask_t, index=idx, index_mask=idx_mask)
                predicted = indices.detach().cpu().numpy().tolist()
                score = score.detach().cpu().numpy().tolist()
                for i in range(len(idx_sememes)):
                    m, f = evaluate(
                        idx_sememes[i], predicted[i], score[i], args.threshold)
                    prevalid_map += m
                    prevalid_f1 += f
                prevalid_loss += loss.item()
            print(
                f'pretrain loss {pretrain_loss / train_data_num}, pretrain map {pretrain_map / train_data_num}, pretrain f1 {pretrain_f1 / train_data_num}')
            print(
                f'prevalid loss {prevalid_loss / valid_data_num}, prevalid map {prevalid_map / valid_data_num}, prevalid f1 {prevalid_f1 / valid_data_num}')

            if prevalid_map / valid_data_num > max_valid_map:
                counter = 0
                max_valid_epoch = epoch
                max_valid_map = prevalid_map / valid_data_num
                max_valid_f1 = prevalid_f1 / valid_data_num
                torch.save(model.state_dict(),
                           os.path.join('output', args.result))
            else:
                counter += 1
                if counter >= 5:
                    break
        print(
            f'pretrain max valid map {max_valid_map}, pretrain max valid f1 {max_valid_f1}')
        model.load_state_dict(torch.load(os.path.join('output', args.result)))

    max_valid_map = 0
    max_valid_epoch = 0
    max_valid_f1 = 0

    for epoch in range(args.epoch_num):
        counter = 0
        torch.cuda.empty_cache()
        print('Train epoch', epoch)
        train_map = 0
        train_loss = 0
        train_f1 = 0
        model.train()
        for sememes, definition_words, idx, idx_sememes, images in tqdm(dataloaders['train']):
            sememes_t, definition_words_t, mask_t, idx_sememes_t, idx, idx_mask, image_tensor = build_input(
                sememes, definition_words, idx, idx_sememes, images, device)
            if args.image_train:
                def_encoder_optimizer.zero_grad()
                img_encoder_optimizer.zero_grad()
                ms_fc_optimizer.zero_grad()
                loss, score, indices = model('train', device=device, defin=definition_words_t,
                                             label=sememes_t, mask=mask_t, index=idx, index_mask=idx_mask, image=image_tensor)
                loss.backward()
                def_encoder_optimizer.step()
                img_encoder_optimizer.step()
                ms_fc_optimizer.step()
            else:
                def_encoder_optimizer.zero_grad()
                def_fc_optimizer.zero_grad()
                loss, score, indices = model(
                    'train', device=device, defin=definition_words_t, label=sememes_t, mask=mask_t, index=idx, index_mask=idx_mask)
                loss.backward()
                def_encoder_optimizer.step()
                def_fc_optimizer.step()
            predicted = indices.detach().cpu().numpy().tolist()
            score = score.detach().cpu().numpy()
            for i in range(len(sememes)):
                m, f = evaluate(sememes[i], predicted[i],
                                score[i], args.threshold)
                train_map += m
                train_f1 += f
            train_loss += loss.item()

        valid_map = 0
        valid_loss = 0
        valid_f1 = 0
        model.eval()
        for sememes, definition_words, idx, idx_sememes, images in tqdm(dataloaders['valid']):
            sememes_t, definition_words_t, mask_t, idx_sememes_t, idx, idx_mask, image_tensor = build_input(
                sememes, definition_words, idx, idx_sememes, images, device)
            if args.image_train:
                loss, score, indices = model('train', device=device, defin=definition_words_t,
                                             label=sememes_t, mask=mask_t, index=idx, index_mask=idx_mask, image=image_tensor)
            else:
                loss, score, indices = model(
                    'train', device=device, defin=definition_words_t, label=sememes_t, mask=mask_t, index=idx, index_mask=idx_mask)
            predicted = indices.detach().cpu().numpy().tolist()
            score = score.detach().cpu().numpy()
            for i in range(len(sememes)):
                m, f = evaluate(sememes[i], predicted[i],
                                score[i], args.threshold)
                valid_map += m
                valid_f1 += f
            valid_loss += loss.item()
        print(
            f'train loss {train_loss / train_data_num}, train map {train_map / train_data_num}, train f1 {train_f1 / train_data_num}')
        print(
            f'valid loss {valid_loss / valid_data_num}, valid map {valid_map /valid_data_num}, valid f1 {valid_f1 / valid_data_num}')
        if valid_map / valid_data_num > max_valid_map:
            counter = 0
            max_valid_epoch = epoch
            max_valid_map = valid_map / valid_data_num
            max_valid_f1 = valid_f1 / valid_data_num
            torch.save(model.state_dict(), os.path.join('output', args.result))
        else:
            counter += 1
            if counter >= 5:
                break
    print(
        f'train max valid map {max_valid_map}, train max valid map epoch {max_valid_epoch}, train max valid f1 {max_valid_f1}')
    model.load_state_dict(torch.load(os.path.join('output', args.result)))

    test_map = 0
    test_loss = 0
    test_f1 = 0
    for sememes, definition_words, idx, idx_sememes, images in tqdm(dataloaders['test']):
        sememes_t, definition_words_t, mask_t, idx_sememes_t, idx, idx_mask, image_tensor = build_input(
            sememes, definition_words, idx, idx_sememes, images, device)
        if args.image_train:
            loss, score, indices = model('train', device=device, defin=definition_words_t,
                                         label=sememes_t, mask=mask_t, index=idx, index_mask=idx_mask, image=image_tensor)
        else:
            loss, score, indices = model(
                'train', device=device, defin=definition_words_t, label=sememes_t, mask=mask_t, index=idx, index_mask=idx_mask)
        score = score.detach().cpu().numpy().tolist()
        predicted = indices.detach().cpu().numpy().tolist()
        for i in range(len(sememes)):
            m, f = evaluate(sememes[i], predicted[i], score[i], args.threshold)
            test_map += m
            test_f1 += f
        test_loss += loss.item()
    print(
        f'test loss {test_loss / test_data_num}, test map {test_map / test_data_num}, test f1 {test_f1 / test_data_num}')


def test(args):
    transform = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])
    }
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    device = torch.device('cuda:'+str(args.device_id))
    print("Data initializing...")
    datasets = {
        'test': MultiSrcDataset('./data_new/test_synset_image_dic.json', './data/babel_data.json', '/data2/private/lvchuancheng/babel_images', tokenizer, transform['test'])
    }
    dataloaders = {x: torch.utils.data.DataLoader(
        datasets[x], batch_size=1, shuffle=True, collate_fn=ms_collate_fn) for x in ['test']}
    for sememes_t, definition_words_t, sememes, mask_t, idx, idx_sememes_t, idx_sememes, idx_mask in dataloaders['test']:
        print(sememes_t.shape)
        print(definition_words_t)
        print(sememes)
        print(mask_t)
        print(idx)
        print(idx_sememes_t.shape)
        print(idx_sememes)
        print(idx_mask)
        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sememe_number", type=int, default=sememe_number)
    parser.add_argument("--data_path", type=str, default='./data/')
    parser.add_argument("--image_path", type=str,
                        default='/data2/private/lvchuancheng/babel_images')
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--def_hidden_size", type=int, default=768)
    parser.add_argument("--img_hidden_size", type=int, default=1000)
    parser.add_argument("--epoch_num", type=int, default=50)
    parser.add_argument("--pretrain_epoch_num", type=int, default=50)
    parser.add_argument("--result", type=str, default='model')
    parser.add_argument("--threshold", type=int, default=-1)
    parser.add_argument("--pretrain_model_lr", type=float, default=1e-5)
    parser.add_argument("--classifier_lr", type=float, default=0.001)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--image_train", type=bool, default=False)
    parser.add_argument("--defin_train", type=bool, default=True)
    args = parser.parse_args()
    pprint(args)
    print('Training start...')
    train(args)
    print('Training completed!')
