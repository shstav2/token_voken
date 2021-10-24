import pandas as pd
import logging
logger = logging.getLogger(__name__)


def load_intervals(path):
    logger.info(f'Loading intervals from {path}..')
    df_intervals = pd.read_csv(path, dtype={'interval_id': object})
    return df_intervals


def load_videos(df_intervals):
    df_videos = pd.DataFrame({'video_id': df_intervals['video_id'].unique()})
    return df_videos


def load_valid_intervals(path):
    df_intervals = load_intervals(path)
    return df_intervals[df_intervals['valid']].copy()