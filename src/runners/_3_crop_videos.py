import math
import numpy as np
import logging

from src.common.status import status_interval_video_downloaded
from src.components._1_data_loader import load_valid_intervals
from src.components._3_video_crop import crop_tool


BATCH_SIZE = 20


logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)


def interval_video_status_and_crop(df_intervals):
    # Interval video status
    logging.info('-------- Interval Video Crop -----------')
    df_intervals['status_interval_video_file'] = df_intervals['interval_id'].apply(lambda video_id:\
        status_interval_video_downloaded(df_intervals, video_id)
    )
    logger.info('[Status] Interval video download:\n' \
                f"{df_intervals['status_interval_video_file'].value_counts()}")
    # Crop videos in batches
    df_intervals_pending = df_intervals[~df_intervals['status_interval_video_file']]
    df_intervals_pending.sort_values(by='video_id', inplace=True)
    pending_count = df_intervals_pending.shape[0]
    logger.info(f'Crop {pending_count} interval videos...:')
    number_of_batches = math.ceil(pending_count / BATCH_SIZE)
    if 0 < pending_count:
        for i, chunk in enumerate(np.array_split(df_intervals_pending, number_of_batches), 1):
            logger.info(f'\tDownloading videos batch #{i} sized {chunk.shape}...:')
            for _, row in chunk.iterrows():
                crop_tool(row)

def run():
    # Interval video mp4 file
    df_intervals = load_valid_intervals()
    logger.info(f'Fetch intervals, {df_intervals.shape} shape dataframe.')
    interval_video_status_and_crop(df_intervals)


if __name__ == '__main__':
    run()
