import numpy as np
import logging

from src.monitoring.status import status_video_downloaded
from src.common.data_loader import load_intervals, load_videos
from src.common.constants import DF_INTERVALS_NOAH
from src.components.vokens._1_video_downloader import youtube_downloader

CHUNK_SIZE = 5


logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)


def video_status_and_download(df_videos):
    logging.info('-------- Video Download -----------')
    # Video download status
    df_videos['status_download'] = df_videos['video_id'].apply(status_video_downloaded)
    logger.info('[Status] Video download:')
    logger.info(df_videos['status_download'].value_counts())
    # Download videos in batches
    logger.info('Downloading videos...:')
    df_videos_pending = df_videos[~df_videos['status_download']]
    df_videos_pending = df_videos_pending.iloc[::-1] # reverse
    for i, chunk in enumerate(np.array_split(df_videos_pending['video_id'], CHUNK_SIZE)):
        logger.info(f'\tDownloading videos batch #{i} sized {len(chunk)}...:')
        for video_id in chunk:
            youtube_downloader(video_id)

def run():
    # Video mp4 file
    df_intervals = load_intervals(DF_INTERVALS_NOAH)
    df_videos = load_videos(df_intervals)
    video_status_and_download(df_videos)


if __name__ == '__main__':
    run()
