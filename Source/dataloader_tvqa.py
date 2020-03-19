import torch.utils.data as data
import torch
import pandas as pd
import os
import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


# Truncate pair of sequences if longer than max_length
def _truncate_seq_pair_inv(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)
        else:
            tokens_b.pop()


# Dataloader for the Read Branch
class ReadBranchData(data.Dataset):

    def __init__(self, args, split, tokenizer):
        self.tokenizer = tokenizer

        # TODO


    def __len__(self):

        # TODO

        return 0


    def __getitem__(self, index):

        # TODO

        return None
