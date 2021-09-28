from tqdm import tqdm
import numpy as np
import torch


def add_args(parser):
    parser.add_argument("--data_set",
                        default='all_data',
                        type=str)
    parser.add_argument("--batch_size",
                        default=8,
                        type=int)
    parser.add_argument("--hidden_size",
                        default=768,
                        type=int)
    parser.add_argument("--classifier_learning_rate",
                        default=1e-3,
                        type=float,
                        help="The initial learning rate for classifier.")
    parser.add_argument("--encoder_learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for encoder.")
    parser.add_argument("--dropout",
                        default=0.3,
                        type=float)
    parser.add_argument("--num_epochs",
                        default=100,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--pretrain_num_epochs",
                        default=100,
                        type=int,
                        help="Total number of pretraining epochs to perform.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval or not.")
    parser.add_argument("--device",
                        default='cuda:4',
                        type=str)
    parser.add_argument("--en",
                        action='store_true')
    parser.add_argument("--zh",
                        action='store_true')
    parser.add_argument("--fr",
                        action='store_true')
    parser.add_argument("--gloss",
                        action='store_true')
    parser.add_argument("--word",
                        action='store_true')

    return parser


def calculate_MAP_f1(output, indice, ground_truth, threshold):
    temp = []
    for i in range(len(ground_truth)):
        if ground_truth[i] == 1:
            temp.append(i)
    ground_truth = temp
    index = 1
    correct = 0
    MAP = 0
    for predicted_sememe in indice:
        if predicted_sememe in ground_truth:
            correct += 1
            MAP += (correct / index)
        index += 1
    MAP /= len(ground_truth)
    real_prediction = []
    for i in range(len(output)):
        if output[i] > threshold:
            real_prediction.append(i)
    prediction = real_prediction
    if len(list(set(prediction) & set(ground_truth))) == 0:
        f1 = 0
    else:
        recall = len(list(set(prediction) & set(ground_truth))) / \
            len(ground_truth)
        precision = len(list(set(prediction) & set(
            ground_truth))) / len(prediction)
        f1 = 2*recall*precision/(recall + precision)
    return MAP, f1


def evaluate(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []
    all_MAP = 0.0
    all_f1 = 0.0
    all_loss = 0.0
    for ids, masks, labels in tqdm(dataloader):
        ids = ids.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            loss, output, indice = model(
                input_ids=ids, input_mask=masks, labels=labels)
        all_loss += loss.item()
        output = output.detach().cpu().numpy().tolist()
        indice = indice.detach().cpu().numpy().tolist()
        labels = labels.cpu().numpy().tolist()
        for i in range(len(output)):
            MAP, f1 = calculate_MAP_f1(output[i], indice[i], labels[i], 0.3)
            all_MAP += MAP/len(output)
            all_f1 += f1/len(output)
    all_loss /= len(dataloader)
    all_MAP /= len(dataloader)
    all_f1 /= len(dataloader)
    return all_loss, all_MAP, all_f1


def get_model_name(args):
    model_name = []
    if args.word:
        model_name.append('word')
    if args.gloss:
        model_name.append('gloss')
    if args.en:
        model_name.append('en')
    if args.zh:
        model_name.append('zh')
    if args.fr:
        model_name.append('fr')
    return '_'.join(model_name)+'.pt'
