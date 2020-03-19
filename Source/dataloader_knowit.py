import torch.utils.data as data
import torch
import pandas as pd
import os
import re
import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext


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


# Class to contain a single instance of the dataset
class DataSample(object):
    def __init__(self,
                 qid,
                 question,
                 answer1,
                 answer2,
                 answer3,
                 answer4,
                 subtitles,
                 vision,
                 knowledge,
                 label):
        self.qid = qid
        self.question = question
        self.subtitles = subtitles
        self.kg = knowledge
        self.label = label
        self.vision = vision
        self.answers = [
            answer1,
            answer2,
            answer3,
            answer4,
        ]


# Load KnowIT VQA data
def load_knowit_data(args, split_name):
    input_file = ''
    if split_name == 'train':
        input_file = os.path.join(args.data_dir, 'knowit_data/knowit_data_train.csv')
    elif split_name == 'val':
        input_file = os.path.join(args.data_dir, 'knowit_data/knowit_data_val.csv')
    elif split_name == 'test':
        input_file = os.path.join(args.data_dir, 'knowit_data/knowit_data_test.csv')
    df = pd.read_csv(input_file, delimiter='\t')
    logger.info('Loaded file %s.' % (input_file))
    return df


# Dataloader for the Read Branch
class ReadBranchData(data.Dataset):

    def __init__(self, args, split, tokenizer):
        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length
        df = load_knowit_data(args, split)
        self.samples = self.get_data(df)
        self.num_samples = len(self.samples)
        logger.info('ReadData branch ready with %d samples' % self.num_samples)


    # Load data into list of DataSamples
    def get_data(self, df):
        samples = []
        for index, row in df.iterrows():
            question = row['question']
            answer1 = row['answer1']
            answer2 = row['answer2']
            answer3 = row['answer3']
            answer4 = row['answer4']
            subtitles = cleanhtml(row['subtitle'].replace('<br />', ' ').replace(' - ', ' '))
            label = int(df['idxCorrect'].iloc[index] - 1)
            samples.append(DataSample(qid=index,
                                      question=question,
                                      answer1=answer1,
                                      answer2=answer2,
                                      answer3=answer3,
                                      answer4=answer4,
                                      subtitles = subtitles,
                                      vision=None,
                                      knowledge=None,
                                      label=label))
        return samples


    def __len__(self):
        return self.num_samples


    def __getitem__(self, index):
        # Convert each sample into 4 BERT input sequences as:
        # [CLS] + subtitles + question + [SEP] + answer1 + [SEP]
        # [CLS] + subtitles + question + [SEP] + answer2 + [SEP]
        # [CLS] + subtitles + question + [SEP] + answer3 + [SEP]
        # [CLS] + subtitles + question + [SEP] + answer4 + [SEP]

        sample = self.samples[index]
        subtitle_tokens = self.tokenizer.tokenize(sample.subtitles)
        question_tokens = self.tokenizer.tokenize(sample.question)
        choice_features = []
        for answer_index, answer in enumerate(sample.answers):

            start_tokens = subtitle_tokens[:] + question_tokens[:]
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

            assert len(input_ids) == self.max_seq_length
            assert len(input_mask) == self.max_seq_length
            assert len(segment_ids) == self.max_seq_length

            choice_features.append((tokens, input_ids, input_mask, segment_ids))

        input_ids = torch.tensor([data[1] for data in choice_features], dtype=torch.long)
        input_mask = torch.tensor([data[2] for data in choice_features], dtype=torch.long)
        segment_ids = torch.tensor([data[3] for data in choice_features], dtype=torch.long)
        qid = torch.tensor(sample.qid, dtype=torch.long)
        label = torch.tensor(sample.label, dtype=torch.long)

        return input_ids, input_mask, segment_ids, qid, label
