import os
import logging
import pandas as pd

from src.common.constants import FRAME_RATE

logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)



COL_WORD                = 'word'
# Frames
COL_WORD_FRAME_START    = 'start_frame'
COL_WORD_FRAME_END      = 'end_frame'
COL_WORD_FRAMES_COUNT   = 'frames_count'
COL_WORD_FRAME_SELECTED = 'selected_frame'

COL_WORD_TIME           = 'time'

# # Characters
# COL_WORD_CHARS_LEN_P_1  = 'word_len_plus_1'
# COL_WORD_CHAR_START     = 'start_char'
# COL_WORD_CHAR_END       = 'end_char'

# COL
# df_token_voken['word_end'] = df_token_voken.groupby('interval_id')['word_len_plus_1'].transform(pd.Series.cumsum)
# df_token_voken['word_start'] = df_token_voken['word_end'] - df_token_voken['word_len_plus_1'] + 1

def annotate_word(df_words):
    # Word frame annotations
    df_words[COL_WORD_FRAME_SELECTED] = df_words[[COL_WORD_FRAME_START, COL_WORD_FRAME_END]].mean(axis=1).astype(int)

    df_words[COL_WORD_TIME] = round(df_words[COL_WORD_FRAME_SELECTED] / FRAME_RATE, 1)

    # Word to characters mapping
    df_words[COL_WORD_CHARS_LEN_P_1] = df_words[COL_WORD].str.len() + 1
    df_words[COL_WORD_CHAR_END] = df_words[COL_WORD_CHARS_LEN_P_1].transform(pd.Series.cumsum)
    df_words[COL_WORD_CHAR_START] = df_words[COL_WORD_CHAR_END] - df_words[COL_WORD_CHARS_LEN_P_1] + 1

    return df_words


def save_df_word(interval_id):
    raw_text_path = resolve_interval_raw_text_path(interval_id)
    df_words.to_csv(raw_text_path, index=False, header=True)
