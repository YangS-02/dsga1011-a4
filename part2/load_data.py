import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):
    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.

        NOTES:
            * Why do we need beginning of sentence token for decoder?
                - Decoder predicts SQL queries one token at a time. And it makes the next token prediction at each input position.
                - So, at the start, it has no SQL tokens to predict the next, so it needs something to start with.
        '''
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.questions_tokenized, self.sqls_tokenized  = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        """
        :param data_folder: path to directory containing the questions and SQL queries files.
        :param split: one of "train", "dev", or "test"
        :param tokenizer: set to t5-small in the constructor.
        """
        tokenizer_options = {# Shared options for encoder/questions in all splits and decoder/SQL in only train/dev splits
            "add_special_tokens": True,  # add </s> the end of sequence token for T5 to end of questions and sql queries
            "padding": False, # Make the choice explicit (padding is handled in the collate function)
            "truncation": False,  # Keep all tokens
        }
        questions = load_lines(os.path.join(data_folder, f"{split}.nl"))
        tokenized_question = tokenizer(questions, **tokenizer_options)["input_ids"] # "input_ids" gets the lists of token ids for each question.
        if split == "test":  # Since only questions for test set.
            return tokenized_question, None
        queries = load_lines(os.path.join(data_folder, f"{split}.sql"))
        tokenized_queries = tokenizer(queries, **tokenizer_options)["input_ids"]
        return tokenized_question, tokenized_queries

    def __len__(self):
        """
        Tells DataLoader how many examples in the split.
        """
        return len(self.questions_tokenized)

    def __getitem__(self, idx):
        """
        Tells DataLoader what to return for one example.
        """
        question_ids = torch.tensor(self.questions_tokenized[idx], dtype=torch.long)
        if self.sqls_tokenized is None:  # test set
            return question_ids
        return question_ids, torch.tensor(self.sqls_tokenized[idx], dtype=torch.long)


def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    # NOTES:
        - __getitem__ returns one (question, SQL) pair (already tokenized tho and in a tuple) for each example.
          And the DataLoader packages pairs of these examples into a batch.
        - Decoder_inputs vs decoder_targets:
            - [<pad> ,  SELECT      ,    airline_code]
            - [SELECT,  airline_code,    </s>      ]
            - So, use <pad> and encoder output to predict SELECT; use <pad> and SELECT to predict airline_code;
              use <pad> and SELECT and airline_code to predict </s>.
            - And we get input from the target: Take all but the last token of target. And then add <pad> to the start.


    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    questions, sqls = zip(*batch)  # Unpack the batch
    encoder_ids = pad_sequence(questions, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()

    decoder_targets = pad_sequence(sqls, batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = torch.full((len(batch), 1), PAD_IDX, dtype=torch.long)
    decoder_inputs = torch.cat((initial_decoder_inputs, decoder_targets[:, :-1]), dim=1)
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = pad_sequence(batch, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    initial_decoder_inputs = torch.full((len(batch), 1), PAD_IDX, dtype=torch.long)
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    return train_x, train_y, dev_x, dev_y, test_x