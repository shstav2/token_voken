import logging
import numpy as np
import pandas as pd

from transformers import AutoTokenizer

from src.common.path_resolvers import resolve_interval_raw_text_path
from src.common.display_utils import V
from src.common.debug import one_percent_chance, ten_percent_chance


logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)


TAG = 'BertTokenization'


COL_WORD                = 'word'
# Characters
COL_WORD_CHARS_LEN_P_1  = 'word_len_plus_1'
COL_WORD_CHAR_START     = 'start_char'
COL_WORD_CHAR_END       = 'end_char'
# Tokenization
COL_OFFSET_START        = 'offset_start'
COL_BERT_TOKEN          = 'bert_token'
COL_BERT_TOKEN_ID       = 'token_id'
# Frames
COL_WORD_FRAME_START    = 'start_frame'
COL_WORD_FRAME_END      = 'end_frame'
COL_WORD_FRAMES_COUNT   = 'frames_count'
COL_WORD_FRAME_SELECTED = 'selected_frame'


COL_OUTPUT_WORD_TOKEN = [
    # word / token
    COL_WORD, COL_BERT_TOKEN, COL_BERT_TOKEN_ID,
    # frame info
    COL_WORD_FRAME_SELECTED, COL_WORD_FRAME_START, COL_WORD_FRAME_END,
    # character indices
    COL_WORD_CHAR_START, COL_WORD_CHAR_END,
]

# >>> df_raw_words
#             word  start_char  end_char
# 0         having           1         7
# 1           your           8        12
# 2         mother          13        19
COL_OUTPUT_WORDS = [
    COL_WORD, COL_WORD_CHAR_START, COL_WORD_CHAR_END
]

tokenizer_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)


class BertTokens:
    def __init__(self, interval_id):
        #  '214436'
        self.interval_id = interval_id
        # '/home/stav/Data/PATS_DATA/Videos/oliver/DRauXXz6t0Y/214436/Text/Raw.csv'
        self.text_path = resolve_interval_raw_text_path(self.interval_id)
        #   >>> df_raw_words
        #        word  start_char  end_char
        # 0    having           0         6
        # 1      your           7        11
        # 2    mother          12        18
        # 3     catch          19        24
        # 4       you          25        28
        self.df_words = pd.read_csv(self.text_path)
        # 'having your mother catch you..'
        self.text = ' '.join(self.df_words[COL_WORD])
        self.debug = one_percent_chance()
    def save_df_tokens(self):
        df_words, df_bert_tokens, df_words_and_tokens = self.get_df_tokens()


    def get_df_tokens(self):
        self._annotate_words()
        tokenized_text, tokenized_line, offset_mapping = BertTokens._tokenize_text(self.text)
        df_bert_tokens = BertTokens._as_dataframe(tokenized_text, tokenized_line, offset_mapping)
        df_words_and_tokens = map_bert_token_to_original_word(df_bert_tokens, self.df_words)
        return (
            self.df_words[COL_OUTPUT_WORDS],
            df_bert_tokens,
            df_words_and_tokens[COL_OUTPUT_WORD_TOKEN])
    def _extract_text(self):
        # raw.csv
        raw_text_path = resolve_interval_raw_text_path(self.interval_id)
        df_raw = pd.read_csv(raw_text_path)
        words = df_raw[COL_WORD].tolist()
        text = ' '.join(words)
        return df_raw, text
    @staticmethod
    def _tokenize_text(text):
        tokenized_text = tokenizer.tokenize(text, add_special_tokens=False)
        tokenized_output = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        tokenized_line = tokenized_output['input_ids']
        offset_mapping = np.array(tokenized_output['offset_mapping'])
        #       bert input ids     ==     word pieces     ==     word piece bounds
        assert len(tokenized_line) == len(tokenized_text) == len(offset_mapping)
        return tokenized_text, tokenized_line, offset_mapping
    @staticmethod
    def _as_dataframe(tokenized_text, tokenized_line, offset_mapping):
        df_bert_tokens = pd.DataFrame({
            COL_BERT_TOKEN: tokenized_text,
            'token_id': tokenized_line,
            'offset_start': offset_mapping[:, 0],
            'offset_end': offset_mapping[:, 1],
        })
        return df_bert_tokens
    def _annotate_words(self):
        # Word to frame
        self.df_words[COL_WORD_FRAME_SELECTED] = self.df_words[[COL_WORD_FRAME_START, COL_WORD_FRAME_END]].mean(axis=1).astype(int)
        # Word to characters mapping
        self.df_words[COL_WORD_CHARS_LEN_P_1] = self.df_words[COL_WORD].str.len() + 1
        self.df_words[COL_WORD_CHAR_END] = self.df_words[COL_WORD_CHARS_LEN_P_1].transform(pd.Series.cumsum)
        self.df_words[COL_WORD_CHAR_START] = self.df_words[COL_WORD_CHAR_END] - self.df_words[COL_WORD_CHARS_LEN_P_1]
        self.df_words[COL_WORD_CHAR_END] = self.df_words[COL_WORD_CHAR_END] - 1
        self._check_annotations()
    def _check_annotations(self):
        if ten_percent_chance():
            word, start_char, end_char = self.df_words.sample().iloc[0]
            assert self.text[start_char:end_char] == word
            if self.debug:
                logging.info(f'{TAG} {V} Word Annotation {word} = text[{start_char}:{end_char}]')



# https://stackoverflow.com/questions/44367672/best-way-to-join-merge-by-range-in-pandas
def map_bert_token_to_original_word(df_tokens, df_words):
    # offset mask
    a = df_tokens[COL_OFFSET_START].values
    word_end_idx = df_words[COL_WORD_CHAR_END].values
    word_start_idx = df_words[COL_WORD_CHAR_START].values
    mask_offset = (a[:, None] >= word_start_idx) & (a[:, None] < word_end_idx)
    # style_value_counts(mask_offset, 'Offset Mask')
    # interval mask
    # a2 = A.interval_id.values
    # b2 = B.interval_id.values
    # mask_interval = ((a2[:, None] == b2))
    # # style_value_counts(mask_interval, 'Interval Mask')
    # # combine masks
    # mask_combined = mask_offset & mask_interval
    # i, j = np.where(mask_combined)
    # style_value_counts(mask_combined, 'Combined Mask')
    i, j = np.where(mask_offset)
    cols_all = df_tokens.columns.append(df_words.columns)
    df = pd.DataFrame(
        np.column_stack([df_tokens.values[i], df_words.values[j]]),
        columns=cols_all
    )
    cols_no_dups = list(df.columns)
    for i, col_name in enumerate(df.columns):
        if col_name in df.columns[:i]:
            dup_i = cols_no_dups.index(col_name)
            assert df[cols_no_dups[i]].equals(df[cols_no_dups[dup_i]])
            print(f'cols_no_dups[i={i}]=toDrop')
            cols_no_dups[i] = "toDROP"
    if 'toDrop' in cols_no_dups:
        df.columns = cols_no_dups
        df = df.drop("toDROP", 1)
    df['token_id'] = df['token_id'].astype(int)
    # display(df_with_caption(df.head(n=10), 'Original Word - Bert Token Mapping'))
    return df

#
# interval_id = '214436'
# df_raw_words, df_bert_tokens, df_words_and_tokens = BertTokens(interval_id).get_df_tokens()