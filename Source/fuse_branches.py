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
import time

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
    parser.add_argument("--workers", default=8)
    parser.add_argument("--device", default='cuda', type=str, help="cuda, cpu")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--nepochs', default=100, help='Number of epochs', type=int)
    parser.add_argument('--patience', default=15, type=int)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--weight_loss_read', default=0.06, type=float)
    parser.add_argument('--weight_loss_observe', default=0.06, type=float)
    parser.add_argument('--weight_loss_recall', default=0.08, type=float)
    parser.add_argument('--weight_loss_final', default=0.80, type=float)
    parser.add_argument('--use_read', action='store_true')
    parser.add_argument('--use_observe', action='store_true')
    parser.add_argument('--use_recall', action='store_true')
    parser.add_argument("--train_name", default='FusionMW', type=str)
    args, unknown = parser.parse_known_args()
    return args


class FusionMW(nn.Module):
    def __init__(self):
        super(FusionMW, self).__init__()
        self.fc_read = nn.Sequential(nn.Linear(768, 1))
        self.fc_obs = nn.Sequential(nn.Linear(768, 1))
        self.fc_recall = nn.Sequential(nn.Linear(768, 1))
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Sequential(nn.Linear(3, 1))

    def forward(self, in_read_feat, in_obs_feat, in_recall_feat):

        num_choices = in_read_feat.shape[1]

        # R, O, LL features
        flat_in_read_feat = in_read_feat.view(-1, in_read_feat.size(-1))
        flat_in_obs_feat = in_obs_feat.view(-1, in_obs_feat.size(-1))
        flat_in_recall_feat = in_recall_feat.view(-1, in_recall_feat.size(-1))
        flat_in_read_feat = self.dropout(flat_in_read_feat)
        flat_in_obs_feat = self.dropout(flat_in_obs_feat)
        flat_in_recall_feat = self.dropout(flat_in_recall_feat)

        # R, O, LL scores
        read_scores = self.fc_read(flat_in_read_feat)
        obs_scores = self.fc_obs(flat_in_obs_feat)
        recall_scores = self.fc_recall(flat_in_recall_feat)
        reshaped_read_scores = read_scores.view(-1, num_choices)
        reshaped_obs_scores = obs_scores.view(-1, num_choices)
        reshaped_recall_scores = recall_scores.view(-1, num_choices)

        # Final score
        all_feat = torch.squeeze(torch.cat([read_scores, obs_scores, recall_scores], 1), 1)
        final_scores = self.classifier(all_feat)
        reshaped_final_scores = final_scores.view(-1, num_choices)
        return [reshaped_read_scores, reshaped_obs_scores, reshaped_recall_scores, reshaped_final_scores]


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


def trainEpoch(args, train_loader, model, criterion, optimizer, epoch):

    read_losses, obs_losses, recall_losses = utils.AverageMeter(), utils.AverageMeter(), utils.AverageMeter()
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

        # Track loss
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

    # Plot loss after all mini-batches have finished
    plotter.plot('loss', 'train', 'Class Loss', epoch, losses.avg)


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


def train(args, modeldir):

    # Set GPU
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(args.device, n_gpu))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Create training directory
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)

    # Model, optimizer and loss
    model = FusionMW()
    if args.device == "cuda":
        model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    class_loss = nn.CrossEntropyLoss().cuda()

    # Data
    trainDataObject = FusionDataloader(args, split='train')
    valDataObject = FusionDataloader(args, split='val')
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
        trainEpoch(args, train_loader, model, class_loss, optimizer, epoch)
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
        filename = os.path.join(modeldir, 'model_latest.pth.tar')
        torch.save(state, filename)
        if is_best:
            filename = os.path.join(modeldir, 'model_best.pth.tar')
            torch.save(state, filename)


def evaluate(args, modeldir):

    # Model
    model = FusionMW()
    if args.device == "cuda":
        model.cuda()
    class_loss = nn.CrossEntropyLoss().cuda()
    logger.info("=> loading checkpoint from '{}'".format(modeldir))
    checkpoint = torch.load(os.path.join(modeldir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])

    # Data
    evalDataObject = FusionDataloader(args, split='test')
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

    # Print accuracy
    df = pd.read_csv(os.path.join(args.data_dir, 'knowit_data/knowit_data_test.csv'), delimiter='\t')
    utils.accuracy(df, out, label, index)


if __name__ == "__main__":

    args = get_params()

    assert args.dataset in ['knowit', 'tvqa']

    if args.dataset == 'knowit':
        from Source.dataloader_knowit import FusionDataloader
        args.descriptions_file = 'Data/knowit_observe/scenes_descriptions.csv'
    elif args.dataset == 'tvqa':
        # from Source.dataloader_tvqa import FusionDataloader
        logger.error('Sorry, TVQA+ dataset not implemented yet.')
        import sys
        sys.exit(0)

    # Create training and data directories
    modeldir = os.path.join('Training', args.train_name)
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)

    outdatadir = os.path.join(args.data_dir, args.dataset)
    if not os.path.exists(outdatadir):
        os.makedirs(outdatadir)

    # Train if model does not exist
    if not os.path.isfile(os.path.join(modeldir, 'model_best.pth.tar')):
        global plotter
        plotter = utils.VisdomLinePlotter(env_name=args.train_name)
        train(args, modeldir)

    # Evaluation
    evaluate(args, modeldir)
