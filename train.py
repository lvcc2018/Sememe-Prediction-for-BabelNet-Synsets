import json
import os
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm
import argparse
from model import *
import random

tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
device = torch.device('cuda:3')
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
    l = np.zeros((len(sememes), sememe_number), dtype=np.float32)
    for i in range(len(sememes)):
        for s in sememes[i]:
            l[i, s] = 1
    return l

def get_random_idx(instance):
    return random.choice(instance['si'])

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
    definition_words = [instance['di'] for instance in batch]
    sememes_t = torch.tensor(get_sememe_label(sememes), dtype=torch.float32, device=device)
    definition_words_t = torch.tensor(build_sentence_numpy(definition_words), dtype=torch.int64, device=device)
    mask_t = torch.tensor(build_mask_numpy(definition_words), dtype = torch.int64, device=device)
    idx = [get_random_idx(instance) for instance in batch]
    idx_sememes = [instance[1] for instance in idx]
    idx_sememes_t = torch.tensor(get_sememe_label(idx_sememes), dtype = torch.float32, device=device)
    idx = [instance[0] for instance in idx]
    idx_mask = torch.tensor(build_mask_numpy(idx), dtype = torch.int64, device=device)
    idx = build_idx(idx)
    image_tensor = [instance['image'] for instance in batch]
    return sememes_t, definition_words_t, sememes, mask_t, idx, idx_sememes_t, idx_sememes, idx_mask, image_tensor

'''
def train(args):
    with open('./data/sememe_all.txt', 'r', encoding='utf-8') as f:
        sememe_str = f.read()
    f.close()
    index_sememe = sememe_str.split(' ')
    
    train_data, valid_data, test_data = load_data(args.data_path)
    train_dataloader, valid_dataloader, test_dataloader = get_dataloader(args.batch_size, train_data, valid_data, test_data)

    model = MSSP(args)
    model.to(device)
    
    sparse_parameters = [para for name, para in model.fc.named_parameters() if para.requires_grad]
    encoder_parameters = [para for name, para in model.encoder.named_parameters() if para.requires_grad]
    optimizer = torch.optim.Adam(sparse_parameters, lr=args.classifier_lr)
    encoder_optimizer = AdamW(encoder_parameters, lr=args.pretrain_model_lr)
    
    max_valid_map = 0
    max_valid_epoch = 0
    max_valid_f1 = 0
    
    early_stop = 0
    
    if args.pretrain_epoch_num>0:
        for epoch in range(args.pretrain_epoch_num):
            if early_stop >= 5:
                break
            early_stop += 1

            print('Pretrain epoch', epoch)
            pretrain_map = 0
            pretrain_loss = 0
            pretrain_f1 = 0
            for sememes_t, definition_words_t, sememes, mask_t, idx, idx_sememes_t, idx_sememes, idx_mask in train_dataloader:
                optimizer.zero_grad()
                encoder_optimizer.zero_grad()
                loss, score, indices = model('pretrain', x=definition_words_t, y=idx_sememes_t, mask = mask_t, index = idx, index_mask = idx_mask)
                loss.backward()
                optimizer.step()
                encoder_optimizer.step()
                predicted = indices.detach().cpu().numpy().tolist()
                score = score.detach().cpu().numpy()
                for i in range(len(idx_sememes)):
                    m, f = evaluate(idx_sememes[i], predicted[i], score, args.threshold)
                    pretrain_map += m
                    pretrain_f1 += f
                pretrain_loss += loss.item()
            model.eval()
            prevalid_map = 0
            prevalid_loss = 0
            prevalid_f1 = 0
            for sememes_t, definition_words_t, sememes, mask_t, idx, idx_sememes_t, idx_sememes, idx_mask in tqdm(valid_dataloader):
                loss, score, indices = model('pretrain', x=definition_words_t, y=idx_sememes_t, mask = mask_t, index = idx, index_mask = idx_mask)
                predicted = indices.detach().cpu().numpy().tolist()
                score = score.detach().cpu().numpy()
                for i in range(len(idx_sememes)):
                    m, f = evaluate(idx_sememes[i], predicted[i], score, args.threshold)
                    prevalid_map += m
                    prevalid_f1 += f
                prevalid_loss += loss.item()
            print(f'pretrain loss {pretrain_loss / len(train_data)}, pretrain map {pretrain_map / len(train_data)}, pretrain f1 {pretrain_f1 / len(train_data)}')
            print(f'prevalid loss {prevalid_loss / len(valid_data)}, prevalid map {prevalid_map / len(valid_data)}, prevalid f1 {prevalid_f1 / len(valid_data)}')
            
            if prevalid_map / len(valid_data) > max_valid_map:
                early_stop = 0
                max_valid_epoch = epoch
                max_valid_map = prevalid_map / len(valid_data)
                max_valid_f1 = prevalid_f1 / len(valid_data)
                torch.save(model.state_dict(), os.path.join('output', args.result))
        print(f'pretrain max valid map {max_valid_map}, pretrain max valid map epoch {max_valid_epoch},  pretrain max valid f1 {max_valid_f1}')
        model.load_state_dict(torch.load(os.path.join('output', args.result)))

    max_valid_map = 0
    max_valid_epoch = 0
    max_valid_f1 = 0
    early_stop = 0

    for epoch in range(args.epoch_num):
        if early_stop >= 5:
            break
        early_stop += 1
        print('Train epoch', epoch)
        if args.mix_train == 1:
            pretrain_map = 0
            pretrain_loss = 0
            pretrain_f1 = 0
            for sememes_t, definition_words_t, sememes, mask_t, idx, idx_sememes_t, idx_sememes, idx_mask in train_dataloader:
                optimizer.zero_grad()
                encoder_optimizer.zero_grad()
                loss, score, indices = model('pretrain', x=definition_words_t, y=idx_sememes_t, mask = mask_t, index = idx, index_mask = idx_mask)
                loss.backward()
                optimizer.step()
                encoder_optimizer.step()
            model.eval()

        train_map = 0
        train_loss = 0
        train_f1 = 0
        for sememes_t, definition_words_t, sememes, mask_t, idx, idx_sememes_t, idx_sememes, idx_mask in train_dataloader:
            optimizer.zero_grad()
            encoder_optimizer.zero_grad()
            loss, score, indices = model('train', x=definition_words_t, y=sememes_t, mask = mask_t)
            loss.backward()
            optimizer.step()
            encoder_optimizer.step()
            predicted = indices.detach().cpu().numpy().tolist()
            score = score.detach().cpu().numpy()
            for i in range(len(sememes)):
                m, f = evaluate(sememes[i], predicted[i], score, args.threshold)
                train_map += m
                train_f1 += f
            train_loss += loss.item()
        model.eval()

        valid_map = 0
        valid_loss = 0
        valid_f1 = 0
        for sememes_t, definition_words_t, sememes, mask_t, idx, idx_sememes_t, idx_sememes, idx_mask in tqdm(valid_dataloader):
            loss, score, indices = model('train', x=definition_words_t, y=sememes_t, mask=mask_t)
            predicted = indices.detach().cpu().numpy().tolist()
            score = score.detach().cpu().numpy()
            for i in range(len(sememes)):
                m, f  = evaluate(sememes[i], predicted[i], score, args.threshold)
                valid_map += m
                valid_f1 += f
            valid_loss += loss.item()
        print(f'train loss {train_loss / len(train_data)}, train map {train_map / len(train_data)}, train f1 {train_f1 / len(train_data)}')
        print(f'valid loss {valid_loss / len(valid_data)}, valid map {valid_map / len(valid_data)}, valid f1 {valid_f1 / len(valid_data)}')
        if valid_map / len(valid_data) > max_valid_map:
            early_stop = 0
            max_valid_epoch = epoch
            max_valid_map = valid_map / len(valid_data)
            max_valid_f1 = valid_f1 / len(valid_data)
            torch.save(model.state_dict(), os.path.join('output', args.result))
    
    print(f'train max valid map {max_valid_map}, train max valid map epoch {max_valid_epoch}, train max valid f1 {max_valid_f1}')
    model.load_state_dict(torch.load(os.path.join('output', args.result)))

    test_map = 0
    test_loss = 0
    test_f1 = 0
    for sememes_t, definition_words_t, sememes, mask_t, idx, idx_sememes_t, idx_sememes, idx_mask in tqdm(test_dataloader):
        loss, score, indices = model('train', x=definition_words_t, y=sememes_t, mask = mask_t)
        score = score.detach().cpu().numpy().tolist()
        predicted = indices.detach().cpu().numpy().tolist()
        for i in range(len(sememes)):
            m, f  = evaluate(sememes[i], predicted[i], args.threshold)
            test_map += m
            test_f1 += f
        test_loss += loss.item()
    print(f'test loss {test_loss / len(test_data)}, test map {test_map / len(test_data)}, test f1 {test_f1 / len(test_data)}')
'''

def train(args):
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
    print("Data initializing...")
    datasets = {
        'train':MultiSrcDataset('./data_new/train_synset_image_dic.json','./data/babel_data.json','/data2/private/lvchuancheng/babel_images',tokenizer, transform['train']),
        'valid':MultiSrcDataset('./data_new/valid_synset_image_dic.json','./data/babel_data.json','/data2/private/lvchuancheng/babel_images',tokenizer, transform['test']),
        'test':MultiSrcDataset('./data_new/test_synset_image_dic.json','./data/babel_data.json','/data2/private/lvchuancheng/babel_images',tokenizer, transform['test'])
    }
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=args.batch_size, shuffle=True, collate_fn = ms_collate_fn) for x in ['train', 'valid', 'test']}
    print("Model initializing...")
    train_data_num, valid_data_num, test_data_num = [datasets[x].__len__() for x in ['train', 'valid', 'test']]
    model = MSSP(args)
    model.to(device)
    sparse_parameters = [para for name, para in model.fc.named_parameters() if para.requires_grad]
    encoder_parameters = [para for name, para in model.def_encoder.named_parameters() if para.requires_grad]
    optimizer = torch.optim.Adam(sparse_parameters, lr=args.classifier_lr)
    encoder_optimizer = AdamW(encoder_parameters, lr=args.pretrain_model_lr)
    max_valid_map = 0
    max_valid_epoch = 0
    max_valid_f1 = 0
    for epoch in range(args.pretrain_epoch_num):
        print('Pretrain epoch ', epoch)
        model.train()
        pretrain_map = 0
        pretrain_loss = 0
        pretrain_f1 = 0
        for sememes_t, definition_words_t, sememes, mask_t, idx, idx_sememes_t, idx_sememes, idx_mask, image_tensor in tqdm(dataloaders['train']):
            optimizer.zero_grad()
            encoder_optimizer.zero_grad()
            loss, score, indices = model('pretrain',device = device, x=definition_words_t, y=idx_sememes_t, mask = mask_t, index = idx, index_mask = idx_mask)
            loss.backward()
            optimizer.step()
            encoder_optimizer.step()
            predicted = indices.detach().cpu().numpy().tolist()
            score = score.detach().cpu().numpy()
            for i in range(len(idx_sememes)):
                m, f = evaluate(idx_sememes[i], predicted[i], score[i], args.threshold)
                pretrain_map += m
                pretrain_f1 += f
            pretrain_loss += loss.item()
        model.eval()
        prevalid_map = 0
        prevalid_loss = 0
        prevalid_f1 = 0
        for sememes_t, definition_words_t, sememes, mask_t, idx, idx_sememes_t, idx_sememes, idx_mask, image_tensor in tqdm(dataloaders['valid']):
            loss, score, indices = model('pretrain',device = device, x=definition_words_t, y=idx_sememes_t, mask = mask_t, index = idx, index_mask = idx_mask)
            predicted = indices.detach().cpu().numpy().tolist()
            score = score.detach().cpu().numpy()
            for i in range(len(idx_sememes)):
                m, f = evaluate(idx_sememes[i], predicted[i], score[i], args.threshold)
                prevalid_map += m
                prevalid_f1 += f
            prevalid_loss += loss.item()
        print(f'pretrain loss {pretrain_loss / train_data_num}, pretrain map {pretrain_map / train_data_num}, pretrain f1 {pretrain_f1 / train_data_num}')
        print(f'prevalid loss {prevalid_loss / valid_data_num}, prevalid map {prevalid_map / valid_data_num}, prevalid f1 {prevalid_f1 / valid_data_num}')
        
        if prevalid_map / valid_data_num > max_valid_map:
            max_valid_epoch = epoch
            max_valid_map = prevalid_map / valid_data_num
            max_valid_f1 = prevalid_f1 / valid_data_num
            torch.save(model.state_dict(), os.path.join('output', args.result))
    print(f'pretrain max valid map {max_valid_map}, pretrain max valid f1 {max_valid_f1}')
    model.load_state_dict(torch.load(os.path.join('output', args.result)))

    max_valid_map = 0
    max_valid_epoch = 0
    max_valid_f1 = 0

    for epoch in range(args.epoch_num):
        print('Train epoch', epoch)
        train_map = 0
        train_loss = 0
        train_f1 = 0
        model.train()
        for sememes_t, definition_words_t, sememes, mask_t, idx, idx_sememes_t, idx_sememes, idx_mask, image_tensor in tqdm(dataloaders['train']):
            optimizer.zero_grad()
            encoder_optimizer.zero_grad()
            loss, score, indices = model('train',device = device, x=definition_words_t, y=idx_sememes_t, mask = mask_t, index = idx, index_mask = idx_mask)
            loss.backward()
            optimizer.step()
            encoder_optimizer.step()
            predicted = indices.detach().cpu().numpy().tolist()
            score = score.detach().cpu().numpy()
            for i in range(len(sememes)):
                m, f = evaluate(sememes[i], predicted[i], score[i], args.threshold)
                train_map += m
                train_f1 += f
            train_loss += loss.item()
        
        valid_map = 0
        valid_loss = 0
        valid_f1 = 0
        model.eval()
        for sememes_t, definition_words_t, sememes, mask_t, idx, idx_sememes_t, idx_sememes, idx_mask, image_tensor in tqdm(dataloaders['valid']):
            loss, score, indices = model('train',device = device, x=definition_words_t, y=idx_sememes_t, mask = mask_t, index = idx, index_mask = idx_mask)
            predicted = indices.detach().cpu().numpy().tolist()
            score = score.detach().cpu().numpy()
            for i in range(len(sememes)):
                m, f  = evaluate(sememes[i], predicted[i], score[i], args.threshold)
                valid_map += m
                valid_f1 += f
            valid_loss += loss.item()
        print(f'train loss {train_loss / train_data_num}, train map {train_map / train_data_num}, train f1 {train_f1 / train_data_num}')
        print(f'valid loss {valid_loss / valid_data_num}, valid map {valid_map /valid_data_num}, valid f1 {valid_f1 / valid_data_num}')
        if valid_map / valid_data_num > max_valid_map:
            max_valid_epoch = epoch
            max_valid_map = valid_map / valid_data_num
            max_valid_f1 = valid_f1 / valid_data_num
            torch.save(model.state_dict(), os.path.join('output', args.result))
    
    print(f'train max valid map {max_valid_map}, train max valid map epoch {max_valid_epoch}, train max valid f1 {max_valid_f1}')
    model.load_state_dict(torch.load(os.path.join('output', args.result)))

    test_map = 0
    test_loss = 0
    test_f1 = 0
    for sememes_t, definition_words_t, sememes, mask_t, idx, idx_sememes_t, idx_sememes, idx_mask, image_tensor in tqdm(dataloaders['test']):
        loss, score, indices = model('train',device = device, x=definition_words_t, y=idx_sememes_t, mask = mask_t, index = idx, index_mask = idx_mask)
        score = score.detach().cpu().numpy().tolist()
        predicted = indices.detach().cpu().numpy().tolist()
        for i in range(len(sememes)):
            m, f  = evaluate(sememes[i], predicted[i], score[i], args.threshold)
            test_map += m
            test_f1 += f
        test_loss += loss.item()
    print(f'test loss {test_loss / test_data_num}, test map {test_map / test_data_num}, test f1 {test_f1 / test_data_num}')

def test(args):
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
    print("Data initializing...")
    datasets = {
        'test':MultiSrcDataset('./data_new/test_synset_image_dic.json','./data/babel_data.json','/data2/private/lvchuancheng/babel_images',tokenizer, transform['test'])
    }
    for i in range(datasets['test'].__len__()):
        sequence_length = len(datasets['test'].__getitem__(i)['di'])
        idx = datasets['test'].__getitem__(i)['si']
        for idx_pair, sememe in idx:
            for i in idx_pair:
                if i > sequence_length:
                    print(datasets['test'].__getitem__(i))
                    print(sequence_length)
                    return



if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type = str, default = './data/ecf_data/')
    parser.add_argument("--sememe_number", type = int, default = sememe_number)
    parser.add_argument("--batch_size", type = int, default = 4)
    parser.add_argument("--hidden_size", type =int ,default = 768)
    parser.add_argument("--epoch_num", type = int, default = 10)
    parser.add_argument("--pretrain_epoch_num", type = int, default = 30)
    parser.add_argument("--mix_train", type = int, default = 1)
    parser.add_argument("--result", type = str, default = 'model')
    parser.add_argument("--threshold", type = int, default = -1)
    parser.add_argument("--pretrain_model_lr", type = float, default = 1e-5)
    parser.add_argument("--classifier_lr", type = float, default = 0.001)
    args = parser.parse_args()
    print(args)
    print('Training start...')
    train(args)
    # test(args)
    print('Training completed!')