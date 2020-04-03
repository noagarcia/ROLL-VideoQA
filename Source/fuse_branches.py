# coding=utf-8

import argparse
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
import pandas as pd
from Source import utils
import torch.utils.data as data
import time
from pytorch_transformers.tokenization_bert import BertTokenizer

import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='Data/', type=str)
    parser.add_argument("--dataset", default='knowit', type=str, help='knowit or tvqa')
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str)
    parser.add_argument("--do_lower_case", default=True)
    parser.add_argument('--seed', type=int, default=181)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--device", default='cuda', type=str, help="cuda, cpu")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--workers", default=8)
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--nepochs', default=100, help='Number of epochs', type=int)
    parser.add_argument('--patience', default=15, type=int)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--weight_loss_read', default=0.06, type=float)
    parser.add_argument('--weight_loss_observe', default=0.06, type=float)
    parser.add_argument('--weight_loss_recall', default=0.08, type=float)
    parser.add_argument('--weight_loss_final', default=0.80, type=float)
    parser.add_argument("--train_name", default='FusionMW', type=str)
    args, unknown = parser.parse_known_args()
    return args


class LanguageData(object):
    def __init__(self, id_q, question, subtitles, answer1, answer2, answer3, answer4, kg, label, vision = None):
        self.id_q = id_q
        self.question = question
        self.subtitles = subtitles
        self.kg = kg
        self.label = label
        self.vision = vision
        self.answers = [
            answer1,
            answer2,
            answer3,
            answer4,
        ]


class FusionDataloader(data.Dataset):

    def __init__(self, args, split, mode, tokenizer = None):
        # mode: scores, features, both

        self.mode = mode
        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length

        # Load Data
        if split == 'train':
            input_file = os.path.join(args.data_dir, args.csvtrain)
        elif split == 'val':
            input_file = os.path.join(args.data_dir, args.csvval)
        elif split == 'test':
            input_file = os.path.join(args.data_dir, args.csvtest)

        df = pd.read_csv(input_file, delimiter='\t')
        self.labels = (df['idxCorrect'] - 1).to_list()
        self.read_scores = utils.load_obj('data/branches_features/read_scores_%s.pckl' % split)
        self.observe_scores = utils.load_obj('data/branches_features/observe_scores_%s.pckl' % split)
        # self.recall_scores = utils.load_obj('data/branches_features/recall_scores_%s.pckl' % split)
        self.recall_scores = utils.load_obj('data/branches_features/recall_humankg_scores_%s.pckl' % split)

        read_features = utils.load_obj('data/branches_features/read_features_%s.pckl' % split)
        observe_features = utils.load_obj('data/branches_features/observe_features_%s.pckl' % split)
        # recall_features = utils.load_obj('data/branches_features/recall_features_%s.pckl' % split)
        recall_features = utils.load_obj('data/branches_features/recall_humankg_features_%s.pckl' % split)
        qa_features = utils.load_obj('data/branches_features/qa_features_%s.pckl' % split)
        self.read_features = np.reshape(read_features, (int(read_features.shape[0]/4),4,768))
        self.observe_features = np.reshape(observe_features, (int(observe_features.shape[0]/4),4,768))
        # self.recall_features = np.reshape(recall_features[0], (int(recall_features[0].shape[0]/4),5,4,768))
        self.recall_features = np.reshape(recall_features, (int(recall_features.shape[0]/4),4,768))
        self.recall_logits_slice = recall_features[1]
        self.qa_features = np.reshape(qa_features, (int(qa_features.shape[0] / 4), 4, 768))

        self.num_samples = len(self.labels)

        logger.info('Dataloader with %d samples' % self.num_samples)


    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):

        label = self.labels[index]
        outputs = [label, index]
        inputs = []

        if self.mode == 'scores' or self.mode == 'both':
            in_read_scores = self.read_scores[index,:]
            in_obs_scores = self.observe_scores[index,:]
            in_recall_scores = self.recall_scores[index,:]
            inputs.extend([in_read_scores, in_obs_scores, in_recall_scores])

        if not self.mode == 'scores':
            in_read_feat = self.read_features[index,:]
            in_obs_feat = self.observe_features[index,:]
            # recall_slices = self.recall_features[index,:]
            # recall_logits_slice = self.recall_logits_slice[index,:]
            # idx_slice, _ = np.unravel_index(recall_logits_slice.argmax(), recall_logits_slice.shape)
            # in_recall_feat = recall_slices[idx_slice,:]
            in_recall_feat = self.recall_features[index,:]
            inputs.extend([in_read_feat, in_obs_feat, in_recall_feat])

        if self.mode == 'features+qa':

            sample = self.samples[index]
            question_tokens = self.tokenizer.tokenize(sample.question)
            choice_features = []
            for answer_index, answer in enumerate(sample.answers):
                start_tokens = question_tokens[:]
                ending_tokens = self.tokenizer.tokenize(answer)
                _truncate_seq_pair_inv(start_tokens, ending_tokens, self.max_seq_length - 3)
                tokens = [self.tokenizer.cls_token] + start_tokens + [self.tokenizer.sep_token] + ending_tokens + [self.tokenizer.sep_token]
                segment_ids = [0] * (len(start_tokens) + 2) + [1] * (len(ending_tokens) + 1)
                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)
                padding = [self.tokenizer.pad_token_id] * (self.max_seq_length - len(input_ids))
                input_ids += padding
                input_mask += padding
                segment_ids += padding
                choice_features.append((tokens, input_ids, input_mask, segment_ids))

            input_ids = torch.tensor([data[1] for data in choice_features], dtype=torch.long)
            input_mask = torch.tensor([data[2] for data in choice_features], dtype=torch.long)
            segment_ids = torch.tensor([data[3] for data in choice_features], dtype=torch.long)

            inputs.extend([input_ids, segment_ids, input_mask])

        elif self.mode == 'features+qapre':
            in_qa_feat = self.qa_features[index,:]
            inputs.extend([in_qa_feat])

        return inputs, outputs


def trainEpoch(args, train_loader, model, criterion, optimizer, epoch, val_loader = None, num_batches = 0):

    read_losses = utils.AverageMeter()
    obs_losses = utils.AverageMeter()
    recall_losses = utils.AverageMeter()
    final_losses = utils.AverageMeter()
    losses = utils.AverageMeter()
    model.train()
    for batch_idx, (input, target) in enumerate(train_loader):

        # Inputs to Variable type
        input_var = list()
        for j in range(len(input)):
            input_var.append(torch.autograd.Variable(input[j]).cuda())

        # Targets to Variable type
        target_var = list()
        for j in range(len(target)):
            target[j] = target[j].cuda(async=True)
            target_var.append(torch.autograd.Variable(target[j]))

        # Output of the model
        output = model(*input_var)

        # Compute loss
        read_loss = criterion(output[0], target_var[0])
        obs_loss = criterion(output[1], target_var[0])
        recall_loss = criterion(output[2], target_var[0])
        final_loss = criterion(output[3], target_var[0])
        train_loss = args.weight_loss_read * read_loss + \
                     args.weight_loss_observe * obs_loss + \
                     args.weight_loss_recall * recall_loss + \
                     args.weight_loss_final * final_loss

        read_losses.update(read_loss.data.cpu().numpy(), input[0].size(0))
        obs_losses.update(obs_loss.data.cpu().numpy(), input[0].size(0))
        recall_losses.update(recall_loss.data.cpu().numpy(), input[0].size(0))
        final_losses.update(final_loss.data.cpu().numpy(), input[0].size(0))

        losses.update(train_loss.data.cpu().numpy(), input[0].size(0))

        # Backpropagate loss and update weights
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Print info
        logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
            epoch, batch_idx, len(train_loader), 100. * batch_idx / len(train_loader), loss=losses))

        # acc = valEpoch(args, val_loader, model, criterion, (epoch*num_batches)+batch_idx)


    # Plot loss after all mini-batches have finished
    plotter.plot('loss', 'train', 'Class Loss', epoch, losses.avg)
    # return acc


def valEpoch(args, val_loader, model, criterion, epoch):

    losses = utils.AverageMeter()
    model.eval()
    for batch_idx, (input, target) in enumerate(val_loader):

        # Inputs to Variable type
        input_var = list()
        for j in range(len(input)):
            input_var.append(torch.autograd.Variable(input[j]).cuda())

        # Targets to Variable type
        target_var = list()
        for j in range(len(target)):
            target[j] = target[j].cuda(async=True)
            target_var.append(torch.autograd.Variable(target[j]))

        # Output of the model
        with torch.no_grad():
            output = model(*input_var)

        # Compute loss
        _, predicted = torch.max(output[3], 1)
        _, p_read = torch.max(output[0], 1)
        _, p_obs = torch.max(output[1], 1)
        _, p_recall = torch.max(output[2], 1)
        read_loss = criterion(output[0], target_var[0])
        obs_loss = criterion(output[1], target_var[0])
        recall_loss = criterion(output[2], target_var[0])
        final_loss = criterion(output[3], target_var[0])
        train_loss = args.weight_loss_read * read_loss + \
                     args.weight_loss_observe * obs_loss + \
                     args.weight_loss_recall * recall_loss + \
                     args.weight_loss_final * final_loss

        losses.update(train_loss.data.cpu().numpy(), input[0].size(0))

        # Save predictions to compute accuracy
        if batch_idx == 0:
            out = predicted.data.cpu().numpy()
            out_r = p_read.data.cpu().numpy()
            out_o = p_obs.data.cpu().numpy()
            out_ll = p_recall.data.cpu().numpy()
            label = target[0].cpu().numpy()
        else:
            out = np.concatenate((out,predicted.data.cpu().numpy()),axis=0)
            out_r = np.concatenate((out_r, p_read.data.cpu().numpy()), axis=0)
            out_o = np.concatenate((out_o, p_obs.data.cpu().numpy()), axis=0)
            out_ll = np.concatenate((out_ll, p_recall.data.cpu().numpy()), axis=0)
            label = np.concatenate((label,target[0].cpu().numpy()),axis=0)

    # Accuracy
    acc = np.sum(out == label) / len(out)
    logger.info('Validation set: Average loss: {:.4f}\t'
          'Accuracy {acc}'.format(losses.avg, acc=acc))
    plotter.plot('loss', 'val', 'Class Loss', epoch, losses.avg)
    plotter.plot('acc', 'val', 'Class Accuracy', epoch, acc)

    acc_read = np.sum(out_r == label) / len(out)
    acc_osb = np.sum(out_o == label) / len(out)
    acc_recall = np.sum(out_ll == label) / len(out)
    plotter.plot('readacc', 'val', 'Read Accuracy', epoch, acc_read)
    plotter.plot('obsacc', 'val', 'Obs Accuracy', epoch, acc_osb)
    plotter.plot('recallacc', 'val', 'Recall Accuracy', epoch, acc_recall)

    return acc


def train(args, outdir):

    # Set GPU
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(args.device, n_gpu))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Create training directory
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Model
    tokenizer = None
    if args.model == 'multitask_qaatt':
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
        model = FusionFCftMultitaskAtt(args)
        mode = 'features+qa'
    elif args.model == 'multitask_qaattpre':
        model = FusionFCftMultitaskAttPre(args)
        mode = 'features+qapre'
    elif args.model == 'multitask':
        model = FusionFCftMultitask()
        mode = 'features'

    if args.device == "cuda":
        model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    class_loss = nn.CrossEntropyLoss().cuda()

    # Data
    trainDataObject = FusionDataloader(args, split='train', mode=mode, tokenizer=tokenizer)
    valDataObject = FusionDataloader(args, split='val', mode=mode, tokenizer=tokenizer)
    train_loader = torch.utils.data.DataLoader(trainDataObject, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(valDataObject, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)
    num_batches = train_loader.__len__()

    # Now, let's start the training process!
    logger.info('Training loader with %d samples' % train_loader.__len__())
    logger.info('Validation loader with %d samples' % val_loader.__len__())
    logger.info('Training...')
    pattrack = 0
    best_val = 0
    for epoch in range(0, args.nepochs):

        # Epoch
        trainEpoch(args, train_loader, model, class_loss, optimizer, epoch, val_loader, num_batches)
        current_val = valEpoch(args, val_loader, model, class_loss, epoch)

        # Check patience
        is_best = current_val > best_val
        best_val = max(current_val, best_val)
        if not is_best:
            pattrack += 1
        else:
            pattrack = 0
        if pattrack >= args.patience:
            break

        logger.info('** Validation information: %f (this accuracy) - %f (best accuracy) - %d (patience valtrack)' % (current_val, best_val, pattrack))

        # Save
        state = {'state_dict': model.state_dict(),
                 'best_val': best_val,
                 'optimizer': optimizer.state_dict(),
                 'pattrack': pattrack,
                 'curr_val': current_val}
        filename = os.path.join(outdir, 'model_latest.pth.tar')
        torch.save(state, filename)
        if is_best:
            filename = os.path.join(outdir, 'model_best.pth.tar')
            torch.save(state, filename)


def evaluate(args, modeldir):

    # Model
    tokenizer = None
    if args.model == 'multitask_qaatt':
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
        model = FusionFCftMultitaskAtt(args)
        mode = 'features+qa'
    elif args.model == 'multitask_qaattpre':
        model = FusionFCftMultitaskAttPre(args)
        mode = 'features+qapre'
    elif args.model == 'multitask':
        model = FusionFCftMultitask()
        mode = 'features'


    if args.device == "cuda":
        model.cuda()

    # Load best model
    logger.info("=> loading checkpoint from '{}'".format(modeldir))
    checkpoint = torch.load(os.path.join(modeldir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])

    # Data
    evalDataObject = FusionDataloader(args, split='test', mode=mode, tokenizer=tokenizer)
    test_loader = torch.utils.data.DataLoader(evalDataObject, batch_size=args.batch_size, shuffle=False, pin_memory=(not args.no_cuda), num_workers=args.workers)
    logger.info('Evaluation loader with %d samples' % test_loader.__len__())

    # Switch to evaluation mode & compute test samples embeddings
    batch_time = utils.AverageMeter()
    end = time.time()
    model.eval()
    for i, (input, target) in enumerate(test_loader):

        # Inputs to Variable type
        input_var = list()
        for j in range(len(input)):
            input_var.append(torch.autograd.Variable(input[j]).cuda())

        # Targets to Variable type
        target_var = list()
        for j in range(len(target)):
            target[j] = target[j].cuda(async=True)
            target_var.append(torch.autograd.Variable(target[j]))

        # Output of the model
        with torch.no_grad():
            output = model(*input_var)
        _, predicted = torch.max(output[3], 1)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Store outpputs
        if i==0:
            out = predicted.data.cpu().numpy()
            label = target[0].cpu().numpy()
            index = target[1].cpu().numpy()

            scores_read = output[0].data.cpu().numpy()
            scores_observe = output[1].data.cpu().numpy()
            scores_recall = output[2].data.cpu().numpy()
            scores_final = output[3].data.cpu().numpy()
        else:
            out = np.concatenate((out,predicted.data.cpu().numpy()),axis=0)
            label = np.concatenate((label,target[0].cpu().numpy()),axis=0)
            index = np.concatenate((index, target[1].cpu().numpy()), axis=0)

            scores_read = np.concatenate((scores_read, output[0].cpu().numpy()), axis=0)
            scores_observe = np.concatenate((scores_observe, output[1].cpu().numpy()), axis=0)
            scores_recall = np.concatenate((scores_recall, output[2].cpu().numpy()), axis=0)
            scores_final = np.concatenate((scores_final, output[3].cpu().numpy()), axis=0)

    # Compute Accuracy
    print('************* %s' % args.model)
    df = pd.read_csv('data/knowit_data_test.csv', delimiter='\t')
    logger.info('Average time per sample %.02f ms for %d samples' % (batch_time.sum / evalDataObject.num_samples * 1000, evalDataObject.num_samples))
    print_acc(df, out, label, index)

    utils.save_obj(label, os.path.join(outdir, 'test_labels.pckl'))
    utils.save_obj(out, os.path.join(outdir, 'test_predicted.pckl'))
    utils.save_obj(index, os.path.join(outdir, 'test_indices.pckl'))

    utils.save_obj(scores_read, os.path.join(outdir, 'test_scores_read.pckl'))
    utils.save_obj(scores_observe, os.path.join(outdir, 'test_scores_observe.pckl'))
    utils.save_obj(scores_recall, os.path.join(outdir, 'test_scores_recall.pckl'))
    utils.save_obj(scores_final, os.path.join(outdir, 'test_scores_final.pckl'))


if __name__ == "__main__":

    args = get_params()

    if args.model not in accepted_models:
        logger.error("Model not recognised")
    else:
        train_name = 'Fusion_humankgretrieved_%s_%d_%d_%d_%d_seed%d' % (args.model, args.weight_loss_read*100, args.weight_loss_observe*100,
                                                args.weight_loss_recall*100, args.weight_loss_final*100, args.seed)
        outdir = 'Training/Fusion/Multitask/' + train_name + '/'

        if not os.path.isfile(os.path.join(outdir, 'model_best.pth.tar')):
            global plotter
            plotter = utils.VisdomLinePlotter(env_name=train_name)
            train(args, outdir)

        evaluate(args, outdir)
