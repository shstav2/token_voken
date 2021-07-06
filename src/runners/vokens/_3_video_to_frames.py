import numpy as np
import logging

from src.monitoring.status import status_interval_video_frames_dir
from src.common.data_loader import load_valid_intervals
from src.common.path_resolvers import resolve_interval_video_path, resolve_interval_frames_dir
from src.components.vokens._3_video_to_frames import video_to_frames


NUMBER_OF_BATCHES = 43


logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)


def extract_interval_video_to_frames(df_intervals):
    # Interval video frames status
    logging.info('-------- Interval Video ➜ Frames -----------')
    df_intervals['status_interval_frames_dir'] = df_intervals['interval_id'].apply(status_interval_video_frames_dir)
    logger.info('[Status] Interval video frames:\n' \
                f"{df_intervals['status_interval_frames_dir'].value_counts()}")
    # Extract frames from video
    logger.info('Extract interval frames...:')
    df_intervals_pending = df_intervals[~df_intervals['status_interval_frames_dir']]
    df_intervals_pending.sort_values(by=['video_id', 'interval_id'], inplace=True)
    for i, chunk in enumerate(np.array_split(df_intervals_pending, NUMBER_OF_BATCHES), 1):
        logger.info(f'\tExtract interval video frame, batch #{i} sized {chunk.shape}...:')
        for _, row in chunk.iterrows():
            interval_id = row['interval_id']
            interval_video_path = resolve_interval_video_path(interval_id)
            interval_frames_dir = resolve_interval_frames_dir(interval_id)
            if i == 1:
                logger.info(f'\t{interval_video_path} → {interval_frames_dir}')
            video_to_frames(interval_video_path, interval_frames_dir)

def run():
    # Interval video into jpg frames
    df_intervals = load_valid_intervals()
    extract_interval_video_to_frames(df_intervals)


if __name__ == '__main__':
    run()
