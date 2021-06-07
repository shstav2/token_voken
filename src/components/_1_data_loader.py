import pandas as pd
from src.common.constants import INTERVALS_PATH

import logging
logger = logging.getLogger(__name__)


def load_intervals():
    logger.info(f'Loading intervals from {INTERVALS_PATH}..')
    df_intervals = pd.read_csv(INTERVALS_PATH, dtype={'interval_id': object})
    logger.info(f'Fetch {df_intervals.shape} shape dataframe.')
    return df_intervals


def load_videos(df_intervals):
    df_videos = pd.DataFrame({'video_id': df_intervals['video_id'].unique()})
    return df_videos


def load_valid_intervals():
    df_intervals = load_intervals()
    return df_intervals[df_intervals['valid']].copy()