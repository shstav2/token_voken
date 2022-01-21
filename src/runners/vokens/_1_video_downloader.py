import logging
from src.monitoring.status import status_video_downloaded
from src.common.data_loader import load_intervals, load_videos
from src.common.constants import DF_INTERVALS_NOAH
from src.components.vokens._1_video_downloader import youtube_downloader

logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)


LOG_EVERY = 3

HALF_SIZE = 100


def video_status_and_download(df_videos):
    logging.info('-------- Video Download -----------')
    # Video download status
    df_videos['status_download'] = df_videos['video_id'].apply(status_video_downloaded)
    logger.info('[Status] Video download:')
    logger.info(df_videos['status_download'].value_counts())
    # Download videos
    df_videos_pending = df_videos[~df_videos['status_download']]
    # df_videos_pending = df_videos_pending.iloc[::-1] # reverse
    video_ids = df_videos_pending['video_id'].tolist()
    logger.info(f'Downloading last {HALF_SIZE} videos...:')
    for i, video_id in enumerate(video_ids[HALF_SIZE:]):
        if i % LOG_EVERY == 0:
            logger.info(f'Video {i}/{HALF_SIZE} {video_id}')
        youtube_downloader(video_id)

def run():
    df_intervals = load_intervals(DF_INTERVALS_NOAH)
    df_videos = load_videos(df_intervals)
    video_status_and_download(df_videos)


if __name__ == '__main__':
    run()
