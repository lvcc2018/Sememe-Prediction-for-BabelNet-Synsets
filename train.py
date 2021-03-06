import torch
import os
from tqdm import tqdm
from model import *
from utils.data_utils import *
from utils.train_utils import *

import argparse
import pickle
import logging
from transformers import XLMRobertaModel, XLMRobertaTokenizer, AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import logging as log
log.set_verbosity_error()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()

    model_name = get_model_name(args)

    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    data_processer = DataProcesser(
        'data/clean_data', 'data/sememe_idx', tokenizer)

    data = pickle.load(open('data_set/'+args.data_set, 'rb'))
    data_list = {i: data[i] for i in ['train', 'valid', 'test']}

    if args.training_mode == 'pretrain':
        data_set = {i: data_processer.create_pretrain_dataset(
            data_list[i], en_lang=args.en, zh_lang=args.zh, gloss=args.gloss, word=args.word) for i in ['train', 'valid', 'test']}
        data_loader = {i: data_processer.create_dataloader(
            data_set[i], args.batch_size, True, data_processer.pretrain_text_collate_fn) for i in ['train', 'valid', 'test']}
        logger.info("Data Initialization Succeeded!")
        model = TextForSememePrediction(
            args.pretrain_model, SEMEME_NUM, args.hidden_size, args.dropout)
        logger.info("Model Initialization Succeeded!")
        device = args.device
        model.to(device)
        optimizer = AdamW([
            {'params': [p for n, p in model.classification_head.named_parameters(
            ) if p.requires_grad], 'lr':args.classifier_learning_rate},
            {'params': [p for n, p in model.text_encoder.named_parameters(
            ) if p.requires_grad], 'lr':args.encoder_learning_rate, 'momentum':0.9, 'weight_decay':1e-2}
        ])
        if args.do_train:
            logger.info("***** Running training *****")
            logger.info("  Num batches = %d", len(data_loader['train']))
            logger.info("  Batch size = %d", args.batch_size)

            best_val_MAP = 0.0
            best_val_f1 = 0.0
            for epoch in range(args.pretrain_num_epochs):
                tr_loss = 0
                model.train()
                tbar = tqdm(data_loader['train'], desc="Iteration")
                for step, batch in enumerate(tbar):
                    optimizer.zero_grad()
                    batch = tuple(t.to(device) for t in batch)
                    ids, masks, labels, mask_index = batch
                    loss, output, indice = model(mode='pretrain',
                                                 input_ids=ids, input_mask=masks, labels=labels, mask_idx=mask_index)
                    loss.backward()
                    optimizer.step()
                    tr_loss += loss.item()
                    tbar.set_description('Epoch %d Loss = %.4f' %
                                         (epoch, tr_loss / (step+1)))
                Loss, MAP, f1 = evaluate(
                    args.training_mode, model, data_loader['valid'], device)
                logger.info("Loss=%.4f, MAP=%.4f on valid set." % (Loss, MAP))
                if MAP > best_val_MAP:
                    best_val_MAP = MAP
                    torch.save(model.state_dict(), open(
                        os.path.join('output/further_pretrained_model', model_name), 'wb'))
        else:
            state_dict = torch.load(
                open(os.path.join('output/further_pretrained_model', model_name), 'rb'))
            model.load_state_dict(state_dict)
            logger.info("Loaded saved model")

        if args.do_eval:
            logger.info("***** Running evaluation *****")
            logger.info("  Num examples = %d", len(data_loader['test']))
            logger.info("  Batch size = %d", args.batch_size)

            Loss, MAP, f1 = evaluate(
                args.training_mode, model, data_loader['test'], device)
            logger.info("***** Test evaluation completed *****")
            logger.info("MAP=%.4f, f1=%.4f" % (MAP, f1))
            logger.info("***** Writing results to file *****")
            logger.info("Done.")

    elif args.training_mode == 'train':
        data_set = {i: data_processer.create_dataset(
            data_list[i], en_lang=args.en, zh_lang=args.zh, fr_lang=args.fr, gloss=args.gloss, word=args.word) for i in ['train', 'valid', 'test']}
        data_loader = {i: data_processer.create_dataloader(
            data_set[i], args.batch_size, True, data_processer.text_collate_fn) for i in ['train', 'valid', 'test']}

        logger.info("Data Initialization Succeeded!")

        model = TextForSememePrediction(
            args.pretrain_model, SEMEME_NUM, args.hidden_size, args.dropout)

        if args.load_model != 'NOT_LOAD':
            logger.info("Loading Model from {}".format(args.load_model))
            state_dict = torch.load(
                open(args.load_model, 'rb'))
            model.load_state_dict(state_dict)

        logger.info("Model Initialization Succeeded!")

        device = args.device
        model.to(device)

        optimizer = AdamW([
            {'params': [p for n, p in model.classification_head.named_parameters(
            ) if p.requires_grad], 'lr':args.classifier_learning_rate},
            {'params': [p for n, p in model.text_encoder.named_parameters(
            ) if p.requires_grad], 'lr':args.encoder_learning_rate, 'momentum':0.9, 'weight_decay':1e-2}
        ])

        if args.do_train:
            logger.info("***** Running training *****")
            logger.info("  Num batches = %d", len(data_loader['train']))
            logger.info("  Batch size = %d", args.batch_size)

            best_val_MAP = 0.0
            best_val_f1 = 0.0
            for epoch in range(args.num_epochs):
                tr_loss = 0
                model.train()
                tbar = tqdm(data_loader['train'], desc="Iteration")
                for step, batch in enumerate(tbar):
                    optimizer.zero_grad()
                    batch = tuple(t.to(device) for t in batch)
                    ids, masks, labels = batch
                    loss, output, indice = model('train',
                                                 input_ids=ids, input_mask=masks, labels=labels)
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    tr_loss += loss.item()
                    tbar.set_description('Epoch %d Loss = %.4f' %
                                         (epoch, tr_loss / (step+1)))

                Loss, MAP, f1 = evaluate(
                    'train', model, data_loader['valid'], device)
                logger.info("Loss=%.4f, MAP=%.4f on valid set." % (Loss, MAP))
                if MAP > best_val_MAP:
                    best_val_MAP = MAP
                    torch.save(model.state_dict(), open(
                        os.path.join('output', model_name), 'wb'))

        if args.do_eval:
            logger.info("Loaded saved model")
            logger.info("***** Running evaluation *****")
            logger.info("  Num examples = %d", len(data_loader['test']))
            logger.info("  Batch size = %d", args.batch_size)

            Loss, MAP, f1 = evaluate(
                'train', model, data_loader['test'], device)
            logger.info("***** Test evaluation completed *****")
            logger.info("MAP=%.4f, f1=%.4f" % (MAP, f1))
            logger.info("***** Writing results to file *****")
            logger.info("Done.")


def train_img():
    data_processer = ImgDataProcesser()
    data = pickle.load(open('data_set/babel_img_data', 'rb'))
    data_list = {i: data[i] for i in ['train', 'valid', 'test']}

    data_set = {i: data_processer.create_dataset(data_list[i]) for i in [
        'train', 'valid', 'test']}
    data_loader = {i: data_processer.create_dataloader(
        data_set[i], 8, True, data_processer.img_collate_fn) for i in ['train', 'valid', 'test']}

    logger.info("Data Initialization Succeeded!")
    model = ImgForSememePrediction(SEMEME_NUM, 1000)
    logger.info("Model Initialization Succeeded!")
    device = 'cuda:4'
    model.to(device)
    optimizer = AdamW([
        {'params': [p for n, p in model.classification_head.named_parameters(
        ) if p.requires_grad], 'lr':0.0002},
    ])
    logger.info("***** Running training *****")
    logger.info("  Num batches = %d", len(data_loader['train']))
    logger.info("  Batch size = %d", 8)
    best_val_MAP = 0.0
    best_val_f1 = 0.0
    for epoch in range(100):
        tr_loss = 0
        model.train()
        tbar = tqdm(data_loader['train'], desc="Iteration")
        for step, batch in enumerate(tbar):
            optimizer.zero_grad()
            batch = tuple(t.to(device) for t in batch)
            ids, labels = batch
            loss, output, indice = model(input_ids=ids, labels=labels)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            tbar.set_description('Epoch %d Loss = %.4f' %
                                 (epoch, tr_loss / (step+1)))
        Loss, MAP, f1 = img_evaluate(
            model, data_loader['valid'], device)
        logger.info("Loss=%.4f, MAP=%.4f, F1=%.4f on valid set." %
                    (Loss, MAP, f1))
        if MAP > best_val_MAP:
            best_val_MAP = MAP
            torch.save(model.state_dict(), open(
                os.path.join('output', 'img_single'), 'wb'))
    Loss, MAP, f1 = img_evaluate(
        model, data_loader['test'], device)
    logger.info("Loss=%.4f, MAP=%.4f, F1=%.4f on test set." % (Loss, MAP, f1))


def multi_source_train():
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    print(args)
    model_name = get_model_name(args)
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    data_processer = MultiSourceDataProcesser(
        'data/clean_data', 'data/sememe_idx', tokenizer)
    data_list = pickle.load(open('data_set/'+args.data_set, 'rb'))


    if args.training_mode == 'pretrain':
        data_set = {i: data_processer.create_pretrain_dataset(
            data_list[i], en_lang=args.en, zh_lang=args.zh, gloss=args.gloss, word=args.word) for i in ['train', 'valid', 'test']}
        data_loader = {i: data_processer.create_dataloader(
            data_set[i], args.batch_size, True, data_processer.pretrain_text_collate_fn) for i in ['train', 'valid', 'test']}
        logger.info("Data Initialization Succeeded!")
        model = MultiSourceForSememePrediction(
            args.pretrain_model, SEMEME_NUM, args.hidden_size, args.img_hidden_size, args.dropout)
        if args.load_model != 'NOT_LOAD':
            logger.info("Loading Model from {}".format(args.load_model))
            state_dict = torch.load(
                open('multi_source_output/further_pretrained_model/'+args.load_model, 'rb'))
            model.load_state_dict(state_dict)
        logger.info("Model Initialization Succeeded!")
        device = args.device
        model.to(device)
        optimizer = AdamW([
            {'params': [p for n, p in model.text_pretrain_classification_head.named_parameters(
            ) if p.requires_grad], 'lr':args.classifier_learning_rate},
            {'params': [p for n, p in model.text_encoder.named_parameters(
            ) if p.requires_grad], 'lr':args.encoder_learning_rate, 'momentum':0.9, 'weight_decay':1e-2}
        ])
        if args.do_train:
            logger.info("***** Running training *****")
            logger.info("  Num batches = %d", len(data_loader['train']))
            logger.info("  Batch size = %d", args.batch_size)

            best_val_MAP = 0.0
            best_val_f1 = 0.0
            for epoch in range(args.pretrain_num_epochs):
                tr_loss = 0
                model.train()
                tbar = tqdm(data_loader['train'], desc="Iteration")
                for step, batch in enumerate(tbar):
                    optimizer.zero_grad()
                    batch = tuple(t.to(device) for t in batch)
                    ids, masks, labels, mask_index = batch
                    loss, output, indice = model(mode='pretrain',
                                                 text_ids=ids, text_mask=masks, labels=labels, mask_idx=mask_index)
                    loss.backward()
                    optimizer.step()
                    tr_loss += loss.item()
                    tbar.set_description('Epoch %d Loss = %.4f' %
                                         (epoch, tr_loss / (step+1)))
                Loss, MAP, f1 = multi_source_evaluate(
                    args.training_mode, model, data_loader['valid'], device)
                logger.info("Loss=%.4f, MAP=%.4f on valid set." % (Loss, MAP))
                if MAP > best_val_MAP:
                    best_val_MAP = MAP
                    torch.save(model.state_dict(), open(
                        os.path.join('multi_source_output/further_pretrained_model', model_name), 'wb'))

        if args.do_eval:
            logger.info("***** Running evaluation *****")
            logger.info("  Num examples = %d", len(data_loader['test']))
            logger.info("  Batch size = %d", args.batch_size)

            Loss, MAP, f1 = multi_source_evaluate(
                args.training_mode, model, data_loader['test'], device)
            logger.info("***** Test evaluation completed *****")
            logger.info("MAP=%.4f, f1=%.4f" % (MAP, f1))
            logger.info("***** Writing results to file *****")
            logger.info("Done.")

    else:
        data_set = {i: data_processer.create_dataset(
            args.data_feature, data_list[i]) for i in ['train', 'valid', 'test']}
        data_loader = {i: data_processer.create_dataloader(
            data_set[i], args.batch_size, True, data_processer.ms_collate_fn) for i in ['train', 'valid', 'test']}
        logger.info("Data Initialization Succeeded!")
        model = MultiSourceForSememePrediction(
            args.pretrain_model, SEMEME_NUM, args.hidden_size, args.img_hidden_size, args.dropout)
        if args.load_model != 'NOT_LOAD':
            logger.info("Loading Model from {}".format(args.load_model))
            state_dict = torch.load(
                open('multi_source_output/'+args.load_model, 'rb'))
            model.load_state_dict(state_dict)
        logger.info("Model Initialization Succeeded!")
        device = args.device
        model.to(device)
        optimizer = {
            'train_img': AdamW([
                {'params': [p for n, p in model.img_classification_head.named_parameters(
                ) if p.requires_grad], 'lr':args.classifier_learning_rate},
            ]),
            'train_text_with_pooler_output': AdamW([
                {'params': [p for n, p in model.text_pooler_classification_head.named_parameters(
                ) if p.requires_grad], 'lr':args.classifier_learning_rate},
                {'params': [p for n, p in model.text_encoder.named_parameters(
                ) if p.requires_grad], 'lr':args.encoder_learning_rate, 'momentum':0.9, 'weight_decay':1e-2}
            ]),
            'train_text_with_last_hidden_state': AdamW([
                {'params': [p for n, p in model.text_max_classification_head.named_parameters(
                ) if p.requires_grad], 'lr':args.classifier_learning_rate},
                {'params': [p for n, p in model.text_encoder.named_parameters(
                ) if p.requires_grad], 'lr':args.encoder_learning_rate, 'momentum':0.9, 'weight_decay':1e-2}
            ]),
            'train_with_multi_source': AdamW([
                {'params': [p for n, p in model.classification_head.named_parameters(
                ) if p.requires_grad], 'lr':args.classifier_learning_rate},
                {'params': [p for n, p in model.text_encoder.named_parameters(
                ) if p.requires_grad], 'lr':args.encoder_learning_rate, 'momentum':0.9, 'weight_decay':1e-2}
            ]),
            'train_with_multi_source_pro': AdamW([
                {'params': [p for n, p in model.classification_head.named_parameters(
                ) if p.requires_grad], 'lr':args.classifier_learning_rate},
                {'params': [p for n, p in model.img_encoder_classification_head.named_parameters(
                ) if p.requires_grad], 'lr':args.classifier_learning_rate},
                {'params': [p for n, p in model.text_encoder.named_parameters(
                ) if p.requires_grad], 'lr':args.encoder_learning_rate, 'momentum':0.9, 'weight_decay':1e-2}
            ]),
        }[args.training_mode]

        if args.do_train:
            logger.info("***** Running training *****")
            logger.info("  Num batches = %d", len(data_loader['train']))
            logger.info("  Batch size = %d", args.batch_size)

            best_val_MAP = 0.0
            best_val_f1 = 0.0
            for epoch in range(args.num_epochs):
                tr_loss = 0
                model.train()
                tbar = tqdm(data_loader['train'], desc="Iteration")
                for step, batch in enumerate(tbar):
                    optimizer.zero_grad()
                    batch = tuple(t.to(device) for t in batch)
                    text_ids, text_mask, img_ids, labels = batch
                    loss, output, indice = model(mode=args.training_mode,
                                                 text_ids=text_ids, text_mask=text_mask, img_ids = img_ids, labels=labels)
                    loss.backward()
                    optimizer.step()
                    tr_loss += loss.item()
                    tbar.set_description('Epoch %d Loss = %.4f' %
                                         (epoch, tr_loss / (step+1)))
                Loss, MAP, f1 = multi_source_evaluate(
                    args.training_mode, model, data_loader['valid'], device)
                logger.info("Loss=%.4f, MAP=%.4f on valid set." % (Loss, MAP))
                if MAP > best_val_MAP:
                    best_val_MAP = MAP
                    torch.save(model.state_dict(), open(
                        os.path.join('multi_source_output', model_name), 'wb'))

        if args.do_eval:
            logger.info("***** Running evaluation *****")
            logger.info("  Num examples = %d", len(data_loader['test']))
            logger.info("  Batch size = %d", args.batch_size)

            Loss, MAP, f1 = multi_source_evaluate(
                args.training_mode, model, data_loader['test'], device)
            logger.info("***** Test evaluation completed *****")
            logger.info("MAP=%.4f, f1=%.4f" % (MAP, f1))
            logger.info("***** Writing results to file *****")
            logger.info("Done.")


if __name__ == '__main__':
    multi_source_train()
