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

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()

    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    data_processer = DataProcesser('data/clean_data', 'data/sememe_idx', tokenizer)

    data = pickle.load(open('data_set/'+args.data_set, 'rb'))
    data_list = {i: data[i] for i in ['train', 'valid', 'test']}
    data_set = {i: data_processer.create_dataset(data_list[i],en_lang=True, zh_lang=True, fr_lang=True, gloss=True, word=True) for i in [
        'train', 'valid', 'test']}
    data_loader = {i: data_processer.create_dataloader(
        data_set[i], args.batch_size, True, data_processer.text_collate_fn) for i in ['train', 'valid', 'test']}

    logger.info("Data Initialization Succeeded!")

    model = MultiModalForSememePrediction(SEMEME_NUM, 768, 0.3)
    logger.info("Model Initialization Succeeded!")

    device = args.device
    model.to(device)

    # no_decay = ['bias', 'final_layer_norm.weight']
    # params = list(model.named_parameters())
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in params if not any(
    #         nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in params if any(
    #         nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]
    no_decay = ['bias', 'final_layer_norm.weight']

    classification_head_params = [
        para for name, para in model.classification_head.named_parameters() if para.requires_grad]
    encoder_params = params = list(model.text_encoder.named_parameters())

    optimizer = AdamW([
        {'params': [p for n, p in encoder_params if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01, 'lr':2e-5},
        {'params': classification_head_params, 'lr': 1e-3},
        {'params': [p for n, p in encoder_params if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr':2e-5}
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
                loss, indice = model(
                    input_ids=ids, input_mask=masks, labels=labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                tr_loss += loss.item()
                tbar.set_description('Epoch %d Loss = %.4f' %
                                     (epoch, tr_loss / (step+1)))

            logger.info("Testing on validation set...")
            MAP, f1 = evaluate(model, data_loader['train'], device)
            logger.info("MAP=%.4f on train set." % (MAP))
            MAP, f1 = evaluate(model, data_loader['valid'], device)
            logger.info("MAP=%.4f on valid set." % (MAP))
            if MAP > best_val_MAP:
                best_val_MAP = MAP
                torch.save(model.state_dict(), open(
                    os.path.join('output', 'model_pro.pt'), 'wb'))
    else:
        state_dict = torch.load(open(os.path.join('output', 'model_pro.pt'), 'rb'))
        model.load_state_dict(state_dict)
        logger.info("Loaded saved model")

    if args.do_eval:
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(data_loader['test']))
        logger.info("  Batch size = %d", args.batch_size)

        MAP, f1 = evaluate(model, data_loader['test'], device)
        output_eval_file = os.path.join('output', "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Test evaluation completed *****")
            logger.info("MAP=%.4f, f1=%.4f" % (MAP, f1))
            logger.info("***** Writing results to file *****")
            writer.write("MAP=%.4f, f1=%.4f" % (MAP, f1))
            logger.info("Done.")


if __name__ == '__main__':
    main()
