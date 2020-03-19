# coding=utf-8

import argparse
import os
import random
import sys
import numpy as np
import torch
from torch import nn
from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel
from pytorch_transformers.tokenization_bert import BertTokenizer
from Source.train_branch import branch_training, branch_embeddings
import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

np.set_printoptions(threshold=sys.maxsize)


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='data/', type=str)
    parser.add_argument("--dataset", default='knowit', type=str, help='knowit or tvqa')
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str)
    parser.add_argument("--do_lower_case", default=True)
    parser.add_argument('--seed', type=int, default=181)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--num_train_epochs", default=10.0, type=float)
    parser.add_argument("--patience", default=3.0, type=float)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--device", default='cuda', type=str, help="cuda, cpu")
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=32, type=int)
    parser.add_argument("--max_seq_length", default=200, type=int)
    parser.add_argument("--seq_stride", default=100, type=int)
    parser.add_argument("--num_max_slices", default=5, type=int)
    parser.add_argument("--workers", default=8)
    parser.add_argument("--train_name", default='RecallBranch', type=str)
    args, unknown = parser.parse_known_args()
    return args


class RecallTransformer(BertPreTrainedModel):

    def __init__(self, config):
        super(RecallTransformer, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.hidden_size = config.hidden_size
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        num_choices = input_ids.shape[2]
        num_slices = input_ids.shape[1]
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        outputs = self.bert(flat_input_ids, position_ids=flat_position_ids, token_type_ids=flat_token_type_ids, attention_mask=flat_attention_mask, head_mask=head_mask)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_slices, num_choices)
        pooled_reshaped_logits = torch.max(reshaped_logits, dim=1)[0]

        pooled_output_slices = pooled_output.view(-1, num_slices, self.hidden_size)
        outputs = (pooled_reshaped_logits,) + (pooled_output_slices,) + (reshaped_logits,)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(pooled_reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


def pretrain_recall_branch(args):

    # Create training directory
    modeldir = os.path.join('Training', args.train_name)
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
    outdatadir = os.path.join(args.data_dir, args.dataset + ('_embeddings'))
    if not os.path.exists(outdatadir):
        os.makedirs(outdatadir)

    # Prepare GPUs
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(args.device, n_gpu))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Load BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # Do training if there is not already a model in modeldir
    if not os.path.isfile(os.path.join(modeldir, 'pytorch_model.bin')):

        # Prepare model
        model = RecallTransformer.from_pretrained(args.bert_model, cache_dir=os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, 'distributed_{}'.format(-1)))
        model.to(args.device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Load training data
        trainDataObject = RecallBranchData(args, split='train', tokenizer=tokenizer)
        valDataObject = RecallBranchData(args, split='val', tokenizer=tokenizer)

        # Start training
        logger.info('*** Read branch training ***')
        branch_training(args, model, modeldir, n_gpu, trainDataObject, valDataObject)


    # For inference, load trained weights
    model = RecallTransformer.from_pretrained(modeldir)
    model.to(args.device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Get Read Branch embeddings for each dataset split
    logger.info('*** Get read branch embeddgins for each data split ***')
    trainDataObject = RecallBranchData(args, split='train', tokenizer=tokenizer)
    valDataObject = RecallBranchData(args, split='val', tokenizer=tokenizer)
    testDataObject = RecallBranchData(args, split='test', tokenizer=tokenizer)
    branch_embeddings(args, model, outdatadir, trainDataObject, split='train', branch_name='read')
    branch_embeddings(args, model, outdatadir, valDataObject, split='val', branch_name='read')
    branch_embeddings(args, model, outdatadir, testDataObject, split='test', branch_name='read')
    logger.info('*** Pretraining read branch done!')


if __name__ == "__main__":

    args = get_params()

    # Check dataset
    assert args.dataset in ['knowit', 'tvqa']
    if args.dataset == 'knowit':
        from Source.dataloader_knowit import RecallBranchData
    elif args.dataset == 'tvqa':
        # from Source.dataloader_tvqa import RecallBranchData
        logger.error('Sorry, TVQA+ dataset not implemented yet.')
        import sys
        sys.exit(0)

    pretrain_recall_branch(args)