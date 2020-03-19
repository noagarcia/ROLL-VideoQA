# coding=utf-8

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import pandas as pd
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
from Source import utils
import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

np.set_printoptions(threshold=sys.maxsize)


def train_epoch(args, model, train_dataloader, optimizer, max_grad_norm, scheduler, n_gpu, epoch):
    losses = utils.AverageMeter()
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader, desc="Train iter")):
        batch = tuple(t.to(args.device) for t in batch)
        input_ids, input_mask, segment_ids, qid, truelabel = batch[:5]
        outputs = model(input_ids, segment_ids, input_mask, truelabel)
        loss = outputs[0]
        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        losses.update(loss.item(), input_ids.shape[0])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    if args.do_plot:
        plotter.plot('loss', 'train', 'Loss', epoch, losses.avg)


def val_epoch(args, model, val_dataloader, n_gpu, epoch):
    losses = utils.AverageMeter()
    model.eval()
    for step, batch in enumerate(tqdm(val_dataloader, desc="Val iter")):
        batch = tuple(t.to(args.device) for t in batch)
        input_ids, input_mask, segment_ids, qid, truelabel = batch[:5]
        with torch.no_grad():
            outputs = model(input_ids, segment_ids, input_mask, truelabel)
            loss, logits = outputs[:2]
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
        logits = logits.detach().cpu().numpy()
        truelabel = truelabel.detach().cpu()
        outputs = np.argmax(logits, axis=1)
        losses.update(loss.item(), input_ids.shape[0])
        if step == 0:
            label = truelabel.numpy()
            out = outputs
        else:
            label = np.concatenate((label, truelabel.numpy()), axis=0)
            out = np.concatenate((out, outputs), axis=0)
    acc =  np.sum(out == label)/len(label)
    if args.do_plot:
        plotter.plot('loss', 'val', 'Loss', epoch, losses.avg)
        plotter.plot('acc', 'val', 'Accuracy', epoch, acc)
    return acc


def branch_training(args, model, modeldir, n_gpu, trainDataObject, valDataObject):

    # For visualization
    if args.do_plot:
        global plotter
        plotter = utils.VisdomLinePlotter(env_name=args.train_name)

    # Load data
    train_dataloader = torch.utils.data.DataLoader(trainDataObject, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)
    val_dataloader = torch.utils.data.DataLoader(valDataObject, batch_size=args.eval_batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    num_train_optimization_steps = int(trainDataObject.num_samples / args.batch_size) * args.num_train_epochs

    # Optimizer
    num_warmup_steps = float(args.warmup_proportion) * float(num_train_optimization_steps)
    max_grad_norm = 1.0
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps,
                                     t_total=num_train_optimization_steps)  # PyTorch scheduler

    # Start training
    logger.info("Num examples = %d", train_dataloader.__len__())
    logger.info("Batch size = %d", args.batch_size)
    logger.info("Num steps = %d", num_train_optimization_steps)
    pattrack = 0
    best_val = 0
    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        train_epoch(args, model, train_dataloader, optimizer, max_grad_norm, scheduler, n_gpu, epoch)
        current_val = val_epoch(args, model, val_dataloader, n_gpu, epoch)

        # Check patience
        is_best = current_val > best_val
        best_val = max(current_val, best_val)
        if not is_best:
            pattrack += 1
        else:
            pattrack = 0
        if pattrack >= args.patience:
            break

        # Save a trained model and the associated configuration
        if is_best:
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(modeldir)


def branch_embeddings(args, model, outdatadir, evalDataObject, split, branch_name):

    # Load data
    eval_dataloader = torch.utils.data.DataLoader(evalDataObject, batch_size=args.eval_batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)

    # Extract embeddings
    logger.info("Data split : %s", split)
    logger.info("Num examples = %d", eval_dataloader.__len__())
    logger.info("Batch size = %d", args.eval_batch_size)
    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader, desc="Iteration")):
        batch = tuple(t.to(args.device) for t in batch)
        input_ids, input_mask, segment_ids, qid, truelabel = batch[:5]
        with torch.no_grad():
            outputs = model(input_ids, segment_ids, input_mask, labels=truelabel)

        if branch_name == 'recall':
            loss, logits, cls_out, logits_slice = outputs[:4]
            logits_slice = logits_slice.detach().cpu().numpy()
            if step == 0:
                branch_logits_slice = logits_slice
            else:
                branch_logits_slice = np.concatenate((branch_logits_slice, logits_slice), axis=0)
        else:
            loss, logits, cls_out = outputs[:3]

        qid = qid.detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        cls_out = cls_out.detach().cpu().numpy()
        truelabel = truelabel.detach().cpu()
        outputs = np.argmax(logits, axis=1)

        if step == 0:
            label = truelabel.numpy()
            out = outputs
            index = qid
            branch_scores = logits
            branch_embeddings = cls_out
        else:
            label = np.concatenate((label, truelabel.numpy()), axis=0)
            out = np.concatenate((out, outputs), axis=0)
            index = np.concatenate((index, qid), axis=0)
            branch_scores = np.concatenate((branch_scores, logits), axis=0)
            branch_embeddings = np.concatenate((branch_embeddings, cls_out), axis=0)

    # Save embeddings
    if branch_name == 'recall':
        branch_embeddings = (branch_embeddings, branch_logits_slice)
    logger.info('Saving %s embeddings for branch... %s' % (split, branch_name))
    utils.save_obj(branch_scores, os.path.join(outdatadir, '%s_branch_scores_%s.pckl' % (branch_name, split)))
    utils.save_obj(branch_embeddings, os.path.join(outdatadir, '%s_branch_embeddings_%s.pckl' % (branch_name, split)))

    # Print accuracy on the test set
    if split == 'test':
        df = pd.read_csv(os.path.join(args.data_dir, 'knowit_data/knowit_data_test.csv'), delimiter='\t')
        utils.accuracy(df, out, label, index)

