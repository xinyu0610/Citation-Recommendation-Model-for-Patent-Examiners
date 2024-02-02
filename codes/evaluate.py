# -*- encoding:utf-8 -*-
"""
  This script provides an k-BERT exmaple for classification.
"""
import os
import sys
import torch
import json
import random
import argparse
import collections
import torch.nn as nn
import torch.nn.functional as F
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils.tokenizer import *
from uer.model_builder import build_model
from uer.utils.optimizers import BertAdam
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from brain import KnowledgeGraph
from multiprocessing import Process, Pool
import numpy as np


# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

class BertClassifier(nn.Module):
    def __init__(self, args, model):
        super(BertClassifier, self).__init__()
        self.embedding = model.embedding
        self.encoder = model.encoder
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, args.labels_num)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()
        self.use_vm = False if args.no_vm else True
        print("[BertClassifier] use visible_matrix: {}".format(self.use_vm))

    def forward(self, src, label, mask, pos=None, vm=None):
        """
        Args:
            src: [batch_size x seq_length]
            label: [batch_size]
            mask: [batch_size x seq_length]
        """
        # Embedding.
        emb = self.embedding(src, mask, pos)
        # Encoder.
        if not self.use_vm:
            vm = None
        output = self.encoder(emb, mask, vm)
        # Target.
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        loss = self.criterion(self.softmax(logits.view(-1, self.labels_num)), label.view(-1))
        return loss, logits


class BertGRU(nn.Module):
    def __init__(self, args, model):
        # config
        super(BertGRU, self).__init__()
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        # 创建网络结构
        self.embedding = model.embedding
        self.encoder = model.encoder
        self.labels_num = args.labels_num
        # self.pooling = args.pooling

        # 输入形式 (batch, seq, feature)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.output_layer = nn.Linear(args.hidden_size, args.labels_num)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()
        self.use_vm = False if args.no_vm else True
        print("[BertClassifier] use visible_matrix: {}".format(self.use_vm))

    def forward(self, src, label, mask, pos=None, vm=None):
        """
        Args:
            src: [batch_size x seq_length]
            label: [batch_size]
            mask: [batch_size x seq_length]
        """
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
            setattr(self, '_flattened', True)
        # Embedding.
        emb = self.embedding(src, mask, pos)
        # Encoder.
        if not self.use_vm:
            vm = None
        output = self.encoder(emb, mask, vm)  # [32 x 512 x 768]
        # Target.
        # if self.pooling == "mean":
        #     output = torch.mean(output, dim=1)
        # elif self.pooling == "max":
        #     output = torch.max(output, dim=1)[0]
        # elif self.pooling == "last":
        #     output = output[:, -1, :]
        # else:
        #     output = output[:, 0, :]
        gru_out, _ = self.gru(output)  # [32 x 512 x 768]
        gru_out = gru_out[:, -1, :]  # [32 x 768]
        logits = self.output_layer(gru_out)  # [32 x 2]
        loss = self.criterion(self.softmax(logits.view(-1, self.labels_num)), label.view(-1))
        return loss, logits


class BertCNN(nn.Module):
    def __init__(self, args, model):
        # config
        super(BertCNN, self).__init__()
        self.layers_num = args.layers_num
        self.hidden_size = args.hidden_size
        self.emb_size = args.emb_size
        self.dropout = args.dropout
        # 创建网络结构
        self.embedding = model.embedding
        self.encoder = model.encoder
        self.labels_num = args.labels_num
        # self.pooling = args.pooling
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=args.hidden_size, kernel_size=(k, args.hidden_size)) for k in
             (2, 3, 4)])
        self.output_layer = nn.Linear(args.hidden_size * 3, args.labels_num)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()
        self.use_vm = False if args.no_vm else True
        print("[BertClassifier] use visible_matrix: {}".format(self.use_vm))

    def conv_and_pool(self, x, conv):
        x = conv(x)
        x = F.relu(x)
        x = x.squeeze(3)
        size = x.size(2)
        x = F.max_pool1d(x, size)
        x = x.squeeze(2)
        return x

    def forward(self, src, label, mask, pos=None, vm=None):
        """
        Args:
            src: [batch_size x seq_length]
            label: [batch_size]
            mask: [batch_size x seq_length]
        """
        # Embedding.
        emb = self.embedding(src, mask, pos)
        # Encoder.
        if not self.use_vm:
            vm = None
        output = self.encoder(emb, mask, vm)  # [32 x 512 x 768]
        # Target.
        # if self.pooling == "mean":
        #     output = torch.mean(output, dim=1)
        # elif self.pooling == "max":
        #     output = torch.max(output, dim=1)[0]
        # elif self.pooling == "last":
        #     output = output[:, -1, :]
        # else:
        #     output = output[:, 0, :]
        output = torch.unsqueeze(output, 1)  # [32 x 1 x 512 x 768]
        cnn_out = torch.cat([self.conv_and_pool(output, conv) for conv in self.convs], 1)
        logits = self.output_layer(cnn_out)  # [32 x 2]
        loss = self.criterion(self.softmax(logits.view(-1, self.labels_num)), label.view(-1))
        return loss, logits


def add_knowledge_worker(params):
    p_id, sentences, columns, kg, vocab, args = params

    sentences_num = len(sentences)
    dataset = []
    for line_id, line in enumerate(sentences):
        if line_id % 10000 == 0:
            print("Progress of process {}: {}/{}".format(p_id, line_id, sentences_num))
            sys.stdout.flush()
        line = line.strip().split('\t')
        try:
            if len(line) == 2:
                label = int(line[columns["label"]])
                text = CLS_TOKEN + line[columns["text_a"]]

                tokens, pos, vm, _ = kg.add_knowledge_with_vm([text], add_pad=True, max_length=args.seq_length)
                tokens = tokens[0]
                pos = pos[0]
                vm = vm[0].astype("bool")

                token_ids = [vocab.get(t) for t in tokens]
                mask = [1 if t != PAD_TOKEN else 0 for t in tokens]

                dataset.append((token_ids, label, mask, pos, vm))

            elif len(line) == 3:
                label = int(line[columns["label"]])
                text = CLS_TOKEN + line[columns["text_a"]] + SEP_TOKEN + line[columns["text_b"]] + SEP_TOKEN

                tokens, pos, vm, _ = kg.add_knowledge_with_vm([text], add_pad=True, max_length=args.seq_length)
                tokens = tokens[0]
                pos = pos[0]
                vm = vm[0].astype("bool")

                token_ids = [vocab.get(t) for t in tokens]
                mask = []
                seg_tag = 1
                for t in tokens:
                    if t == PAD_TOKEN:
                        mask.append(0)
                    else:
                        mask.append(seg_tag)
                    if t == SEP_TOKEN:
                        seg_tag += 1

                dataset.append((token_ids, label, mask, pos, vm))

            elif len(line) == 4:  # for dbqa
                qid = int(line[columns["qid"]])
                label = int(line[columns["label"]])
                text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
                text = CLS_TOKEN + text_a + SEP_TOKEN + text_b + SEP_TOKEN

                tokens, pos, vm, _ = kg.add_knowledge_with_vm([text], add_pad=True, max_length=args.seq_length)
                tokens = tokens[0]
                pos = pos[0]
                vm = vm[0].astype("bool")

                token_ids = [vocab.get(t) for t in tokens]
                mask = []
                seg_tag = 1
                for t in tokens:
                    if t == PAD_TOKEN:
                        mask.append(0)
                    else:
                        mask.append(seg_tag)
                    if t == SEP_TOKEN:
                        seg_tag += 1

                dataset.append((token_ids, label, mask, pos, vm, qid))
            else:
                pass

        except:
            print("Error line: ", line)
    return dataset


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--trained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--test_path", type=str, required=True,
                        help="Path of the predict set.")
    # parser.add_argument("--predict_output_path", type=str, default="./outputs/bertsim.txt",
    #                     help="Path of the predict output.")
    parser.add_argument("--config_path", default="./models/google_config.json", type=str,
                        help="Path of the config file.")
    parser.add_argument("--vocab_path", default="./models/google_vocab.txt", type=str,
                        help="Path of the vocabulary file.")

    # Model options.
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=512,
                        help="Sequence length.")
    parser.add_argument("--embedding", choices=["bert", "word"], default="bert",
                        help="Emebdding type.")
    parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
                                              "cnn", "gatedcnn", "attn", \
                                              "rcnn", "crnn", "gpt", "bilstm"], \
                        default="bert", help="Encoder type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    # Subword options.
    parser.add_argument("--subword_type", choices=["none", "char"], default="none",
                        help="Subword feature type.")
    parser.add_argument("--sub_vocab_path", type=str, default="models/sub_vocab.txt",
                        help="Path of the subword vocabulary file.")
    parser.add_argument("--subencoder", choices=["avg", "lstm", "gru", "cnn"], default="avg",
                        help="Subencoder type.")
    parser.add_argument("--sub_layers_num", type=int, default=2, help="The number of subencoder layers.")

    # Tokenizer options.
    parser.add_argument("--tokenizer", choices=["bert", "char", "word", "space"], default="bert",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Word tokenizer supports online word segmentation based on jieba segmentor."
                             "Space tokenizer segments sentences into words according to space."
                        )

    # Optimizer options.
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")

    # Training options.
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=20,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")

    # Evaluation options.
    parser.add_argument("--mean_reciprocal_rank", action="store_true", help="Evaluation metrics for DBQA dataset.")

    # kg
    parser.add_argument("--kg_name", required=True, help="KG name or path")
    parser.add_argument("--workers_num", type=int, default=1, help="number of process for loading dataset")
    parser.add_argument("--no_vm", action="store_true", help="Disable the visible_matrix")

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Count the number of labels.
    labels_set = set()
    columns = {}
    with open(args.test_path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            try:
                line = line.strip().split("\t")
                if line_id == 0:
                    for i, column_name in enumerate(line):
                        columns[column_name] = i
                    continue
                label = int(line[columns["label"]])
                labels_set.add(label)
            except:
                pass
    args.labels_num = len(labels_set)

    # Load vocabulary.
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    # Build bert model.
    # A pseudo target is added.
    args.target = "bert"
    model = build_model(args)

    # Build classification model.
    model = BertClassifier(args, model)
    # model = BertGRU(args, model)
    # model = BertCNN(args, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device('cpu')
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=[0,1,2,3,4,5,6])
    model = model.to(device)

    # 加载训练好的模型参数
    if hasattr(model, 'module') and torch.cuda.device_count() > 1:
        model.module.load_state_dict(torch.load(args.trained_model_path, map_location='cpu'))
    else:
        model.load_state_dict(torch.load(args.trained_model_path, map_location='cpu'))

    # Datset loader.
    def batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms):
        instances_num = input_ids.size()[0]
        for i in range(instances_num // batch_size):
            input_ids_batch = input_ids[i * batch_size: (i + 1) * batch_size, :]
            label_ids_batch = label_ids[i * batch_size: (i + 1) * batch_size]
            mask_ids_batch = mask_ids[i * batch_size: (i + 1) * batch_size, :]
            pos_ids_batch = pos_ids[i * batch_size: (i + 1) * batch_size, :]
            vms_batch = vms[i * batch_size: (i + 1) * batch_size]
            yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch
        if instances_num > instances_num // batch_size * batch_size:
            input_ids_batch = input_ids[instances_num // batch_size * batch_size:, :]
            label_ids_batch = label_ids[instances_num // batch_size * batch_size:]
            mask_ids_batch = mask_ids[instances_num // batch_size * batch_size:, :]
            pos_ids_batch = pos_ids[instances_num // batch_size * batch_size:, :]
            vms_batch = vms[instances_num // batch_size * batch_size:]

            yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch

    # Build knowledge graph.
    if args.kg_name == 'none':
        spo_files = []
    else:
        spo_files = [args.kg_name]
    kg = KnowledgeGraph(spo_files=spo_files, predicate=True)

    def read_dataset(path, workers_num=1):

        print("Loading sentences from {}".format(path))
        sentences = []
        with open(path, mode='r', encoding="utf-8") as f:
            for line_id, line in enumerate(f):
                if line_id == 0:
                    continue
                sentences.append(line)
        sentence_num = len(sentences)

        print("There are {} sentence in total. We use {} processes to inject knowledge into sentences.".format(
            sentence_num, workers_num))
        if workers_num > 1:
            params = []
            sentence_per_block = int(sentence_num / workers_num) + 1
            for i in range(workers_num):
                params.append(
                    (i, sentences[i * sentence_per_block: (i + 1) * sentence_per_block], columns, kg, vocab, args))
            pool = Pool(workers_num)
            res = pool.map(add_knowledge_worker, params)
            pool.close()
            pool.join()
            dataset = [sample for block in res for sample in block]
        else:
            params = (0, sentences, columns, kg, vocab, args)
            dataset = add_knowledge_worker(params)

        return dataset

    # Evaluation function.
    def evaluate(args, is_test, metrics='Acc'):
        if is_test:
            dataset = read_dataset(args.test_path, workers_num=args.workers_num)
        else:
            dataset = read_dataset(args.dev_path, workers_num=args.workers_num)

        input_ids = torch.LongTensor([sample[0] for sample in dataset])
        label_ids = torch.LongTensor([sample[1] for sample in dataset])
        mask_ids = torch.LongTensor([sample[2] for sample in dataset])
        pos_ids = torch.LongTensor([example[3] for example in dataset])
        vms = [example[4] for example in dataset]

        batch_size = args.batch_size
        instances_num = input_ids.size()[0]
        if is_test:
            print("The number of evaluation instances: ", instances_num)

        correct = 0
        # Confusion matrix.
        confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

        model.eval()

        if not args.mean_reciprocal_rank:
            for i, (input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch) in enumerate(
                    batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms)):

                # vms_batch = vms_batch.long()
                vms_batch = torch.LongTensor(vms_batch)

                input_ids_batch = input_ids_batch.to(device)
                label_ids_batch = label_ids_batch.to(device)
                mask_ids_batch = mask_ids_batch.to(device)
                pos_ids_batch = pos_ids_batch.to(device)
                vms_batch = vms_batch.to(device)

                with torch.no_grad():
                    try:
                        loss, logits = model(input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch)
                    except:
                        print(input_ids_batch)
                        print(input_ids_batch.size())
                        print(vms_batch)
                        print(vms_batch.size())

                logits = nn.Softmax(dim=1)(logits)
                pred = torch.argmax(logits, dim=1)
                gold = label_ids_batch
                for j in range(pred.size()[0]):
                    confusion[pred[j], gold[j]] += 1
                correct += torch.sum(pred == gold).item()

            if is_test:
                print("Confusion matrix:")
                print(confusion)
                print("Report precision, recall, and f1:")

            for i in range(confusion.size()[0]):
                if confusion[i, :].sum().item() == 0:
                    p = 0
                else:
                    p = confusion[i, i].item() / confusion[i, :].sum().item()
                if confusion[:, i].sum().item() == 0:
                    r = 0
                else:
                    r = confusion[i, i].item() / confusion[:, i].sum().item()
                if p + r == 0:
                    f1 = 0
                else:
                    f1 = 2 * p * r / (p + r)
                if i == 1:
                    label_1_f1 = f1
                print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i, p, r, f1))
            print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / len(dataset), correct, len(dataset)))
            if metrics == 'Acc':
                return correct / len(dataset)
            elif metrics == 'f1':
                return label_1_f1
            else:
                return correct / len(dataset)
        else:
            for i, (input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch) in enumerate(
                    batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms)):

                vms_batch = torch.LongTensor(vms_batch)

                input_ids_batch = input_ids_batch.to(device)
                label_ids_batch = label_ids_batch.to(device)
                mask_ids_batch = mask_ids_batch.to(device)
                pos_ids_batch = pos_ids_batch.to(device)
                vms_batch = vms_batch.to(device)

                with torch.no_grad():
                    loss, logits = model(input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch)
                logits = nn.Softmax(dim=1)(logits)
                if i == 0:
                    logits_all = logits
                if i >= 1:
                    logits_all = torch.cat((logits_all, logits), 0)

            order = -1
            gold = []
            for i in range(len(dataset)):
                qid = dataset[i][-1]
                label = dataset[i][1]
                if (qid == order).any:
                    j += 1
                    if label == 1:
                        gold.append((qid, j))
                else:
                    order = qid
                    j = 0
                    if label == 1:
                        gold.append((qid, j))

            label_order = []
            order = -1
            for i in range(len(gold)):
                if gold[i][0] == order:
                    templist.append(gold[i][1])
                elif gold[i][0] != order:
                    order = gold[i][0]
                    if i > 0:
                        label_order.append(templist)
                    templist = []
                    templist.append(gold[i][1])
            label_order.append(templist)

            order = -1
            score_list = []
            for i in range(len(logits_all)):
                score = float(logits_all[i][1])
                qid = int(dataset[i][-1])
                if qid == order:
                    templist.append(score)
                else:
                    order = qid
                    if i > 0:
                        score_list.append(templist)
                    templist = []
                    templist.append(score)
            score_list.append(templist)

            rank = []
            pred = []
            print(len(score_list))
            print(len(label_order))
            for i in range(len(score_list)):
                if len(label_order[i]) == 1:
                    if label_order[i][0] < len(score_list[i]):
                        true_score = score_list[i][label_order[i][0]]
                        score_list[i].sort(reverse=True)
                        for j in range(len(score_list[i])):
                            if score_list[i][j] == true_score:
                                rank.append(1 / (j + 1))
                    else:
                        rank.append(0)

                else:
                    true_rank = len(score_list[i])
                    for k in range(len(label_order[i])):
                        if label_order[i][k] < len(score_list[i]):
                            true_score = score_list[i][label_order[i][k]]
                            temp = sorted(score_list[i], reverse=True)
                            for j in range(len(temp)):
                                if temp[j] == true_score:
                                    if j < true_rank:
                                        true_rank = j
                    if true_rank < len(score_list[i]):
                        rank.append(1 / (true_rank + 1))
                    else:
                        rank.append(0)
            MRR = sum(rank) / len(rank)
            print("MRR", MRR)
            return MRR

    print("Start evaluation on test dataset.")
    evaluate(args, True)


    # def predict(args):
    #     dataset = read_dataset(args.test_path, workers_num=args.workers_num)
    #
    #     input_ids = torch.LongTensor([sample[0] for sample in dataset])
    #     label_ids = torch.LongTensor([sample[1] for sample in dataset])
    #     mask_ids = torch.LongTensor([sample[2] for sample in dataset])
    #     pos_ids = torch.LongTensor([example[3] for example in dataset])
    #     vms = [example[4] for example in dataset]
    #
    #     batch_size = args.batch_size
    #     instances_num = input_ids.size()[0]
    #     print("The number of evaluation instances: ", instances_num)
    #
    #     # calculate similarity
    #     similarity = torch.FloatTensor([]).to(device)
    #     model.eval()
    #
    #     for i, (input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch) in enumerate(
    #             batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms)):
    #
    #         # vms_batch = vms_batch.long()
    #         vms_batch = torch.LongTensor(vms_batch)
    #
    #         input_ids_batch = input_ids_batch.to(device)
    #         label_ids_batch = label_ids_batch.to(device)
    #         mask_ids_batch = mask_ids_batch.to(device)
    #         pos_ids_batch = pos_ids_batch.to(device)
    #         vms_batch = vms_batch.to(device)
    #
    #         with torch.no_grad():
    #             try:
    #                 loss, logits = model(input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch)
    #             except Exception as e:
    #                 print(e)
    #                 print(input_ids_batch)
    #                 print(input_ids_batch.size())
    #                 print(vms_batch)
    #                 print(vms_batch.size())
    #
    #         logits = nn.Softmax(dim=1)(logits)
    #         # weights = torch.Tensor([0, 0.5, 1])
    #         # sim = torch.sum(torch.mul(weights, logits), dim=1)
    #         # similarity = torch.cat([similarity, sim])
    #     return logits


    # predict phase.
    # print("predicting")
    # result = predict(args)
    # # if torch.cuda.is_available():
    # # gpu tensor to python list
    # # result_list = result.cpu().numpy().tolist()
    # # else:
    # # cpu tensor to python list
    # result_list = result.numpy().tolist()
    # lines = []
    # print('Similarity prediction finished. writing to %s' % args.predict_output_path)
    # with open(args.predict_output_path, 'w', newline='') as f:
    #     fr = open('./data_NLP_bal/dataset/sim.tsv', 'r', encoding='utf-8')
    #     for line in fr:
    #         lines.append(line.replace('\t', ' '))
    #     for i in range(len(result_list)):
    #         f.write(str(lines[i].replace('\n', '')) + ' ' + str(result_list[i]) + '\n')
    # print('writing finished')

if __name__ == "__main__":
    main()
