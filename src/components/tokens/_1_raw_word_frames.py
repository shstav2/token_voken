import os
import logging
import pandas as pd

from src.common.debug import one_percent_chance
from src.common.file_utils import save_csv
from src.common.path_resolvers import resolve_interval_local_raw_text_path
from src.common.display_utils import ARR_R


logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)


SPEAKER_NAME = 'noah'
SPEAKER_PATS_DIR = '/Users/staveshemesh/Projects/PATS_DATA/Processed/oliver/data/processed/oliver'
PATS = os.path.join(SPEAKER_PATS_DIR, SPEAKER_NAME)


interval_id = '216509'

COL_WORD_INPUT          = 'Word'
COL_WORD_OUTPUT         = 'word'

# Frames
COL_WORD_FRAME_START    = 'start_frame'
COL_WORD_FRAME_END      = 'end_frame'
COL_WORD_FRAMES_COUNT   = 'frames_count'


TAG = '[Text|Raw]'


def save_raw_text(interval_id):
    df_words = extract_interval_text(interval_id)
    save_df_words(df_words, interval_id)


def extract_interval_text(interval_id):
    interval_text_path = os.path.join(PATS, f'{interval_id}.h5')
    df_words = pd.read_hdf(interval_text_path)
    df_words.rename(columns={COL_WORD_INPUT: COL_WORD_OUTPUT}, inplace=True)
    # Word to frames mapping
    df_words[COL_WORD_FRAME_START] = df_words[COL_WORD_FRAME_START].astype(int)
    df_words[COL_WORD_FRAME_END] = df_words[COL_WORD_FRAME_END].astype(int)
    df_words[COL_WORD_FRAMES_COUNT] = df_words[COL_WORD_FRAME_END] - df_words[COL_WORD_FRAME_START]
    return df_words


def save_df_words(df_words, interval_id):
    raw_text_path = resolve_interval_local_raw_text_path(interval_id)
    save_csv(df_words, raw_text_path)

# def save_df_words(df_words, interval_id):
#     raw_text_path = resolve_interval_local_raw_text_path(interval_id)
#     text_dir = os.path.dirname(raw_text_path)
#     if not os.path.exists(text_dir):
#         os.mkdir(text_dir)
#     df_words.to_csv(raw_text_path, index=False, header=True)
#     if one_percent_chance():
#         cols = df_words.columns.tolist()
#         logger.info(f'{TAG} Saving {cols} {df_words.shape[0]:,} {ARR_R}  {raw_text_path}')
#

