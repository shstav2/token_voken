import numpy as np
import pandas as pd
import logging

from common.status import status_interval_video_downloaded, status_video_downloaded
from pipeline._1_data_loader import load_intervals
from pipeline._2_video_downloader import youtube_downloader

CHUNK_SIZE = 5


logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)


def video_status_and_download(df_intervals):
    logging.info('-------- Video Download -----------')
    # Video download status
    df_videos = pd.DataFrame({'video_id': df_intervals['video_id'].unique()})
    df_videos['status_download'] = df_videos['video_id'].apply(status_video_downloaded)
    logger.info('[Status] Video download:')
    logger.info(df_videos['status_download'].value_counts())
    # Download videos in batches
    logger.info('Downloading videos...:')
    df_videos_pending = df_videos[~df_videos['status_download']]
    for i, chunk in enumerate(np.array_split(df_videos_pending['video_id'], CHUNK_SIZE)):
        logger.info(f'\tDownloading videos batch #{i} sized {len(chunk)}...:')
        for video_id in chunk:
            youtube_downloader(video_id)

def run():
    df_intervals = load_intervals()
    video_status_and_download(df_intervals)
    # Video mp4 file


    # # Interval video mp4 file
    # df_intervals['status_interval_video_downloaded'] = df_intervals['interval_id'].apply(lambda video_id:\
    #     status_interval_video_downloaded(df_intervals, video_id)
    # )
    # logger.info('[Status] Interval video download:')
    # logger.info(df_intervals['status_interval_video_downloaded'].value_counts())


if __name__ == '__main__':
    run()
