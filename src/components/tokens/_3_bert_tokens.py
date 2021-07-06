import logging
import numpy as np
import pandas as pd

from transformers import AutoTokenizer

from src.common.path_resolvers import resolve_interval_raw_text_path


logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)


COL_WORD = 'word'

tokenizer_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

def interval_bert_tokens(interval_id):
    text = extract_text(interval_id)
    df_bert_tokens = get_df_bert_tokens(text)
    return df_bert_tokens


def extract_text(interval_id):
    # raw.csv
    raw_text_path = resolve_interval_raw_text_path(interval_id)
    df_raw = pd.read_csv(raw_text_path)
    words = df_raw[COL_WORD].tolist()
    text = ' '.join(words)
    return text


def get_df_bert_tokens(text):
    tokenized_text, tokenized_line, offset_mapping = tokenize_text(text)
    df_bert_tokens = pd.DataFrame({
        'bert_token': tokenized_text,
        'token_id': tokenized_line,
        'offset_start': offset_mapping[:,0],
        'offset_end': offset_mapping[:,1],
    })
    return df_bert_tokens


def tokenize_text(text):
    tokenized_text = tokenizer.tokenize(text, add_special_tokens=False)
    tokenized_output = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    tokenized_line = tokenized_output['input_ids']
    offset_mapping = np.array(tokenized_output['offset_mapping'])
    #       bert input ids     ==     word pieces     ==     word piece bounds
    assert len(tokenized_line) == len(tokenized_text) == len(offset_mapping)
    return tokenized_text, tokenized_line, offset_mapping





def create_bert_tokens(df_interval_texts):
    interval_ids = df_interval_texts['interval_id'].tolist()
    interval_texts = df_interval_texts['text'].tolist()
    lst_df_word_pieces = []
    all_bert_tokens = [interval_bert_tokens(interval_id, text) for interval_id, text in tqdm(zip(interval_ids, interval_texts))]
    df_bert_tokens = pd.concat(all_bert_tokens)
    return df_bert_tokens

