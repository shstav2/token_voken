import logging
from src.monitoring.status import status_interval_video_downloaded, status_video_downloaded, status_detected_faces_dir
from src.common.data_loader import load_valid_intervals
from src.common.constants import DF_INTERVALS_NOAH
from src.components.vokens._2_video_crop import crop_tool


logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)


LOG_EVERY = 3


def interval_video_status_and_crop(df_intervals):
    # Interval video status
    logging.info('-------- Interval Video Crop -----------')
    df_intervals['status_interval_video_file'] = df_intervals['interval_id'].apply(status_interval_video_downloaded)
    logger.info('[Status] Interval video download:\n' \
                f"{df_intervals['status_interval_video_file'].value_counts()}")
    # Crop videos in batches
    df_intervals['status_full_video_downloaded'] = df_intervals['video_id'].apply(status_video_downloaded)
    df_intervals['status_interval_faces_dir'] = df_intervals['interval_id'].apply(status_detected_faces_dir)
    df_intervals_pending = df_intervals[
        (df_intervals['status_full_video_downloaded']) & (~df_intervals['status_interval_video_file']) & (~df_intervals['status_interval_faces_dir'])
    ]
    df_intervals_pending.sort_values(by='video_id', inplace=True)
    pending_count = df_intervals_pending.shape[0]
    logger.info(f'!Crop {pending_count} interval videos...:')
    if 0 < pending_count:
        for i, row in df_intervals_pending.iterrows():
            if i % LOG_EVERY == 0:
                logger.info(f'{i} / {pending_count} {row["interval_id"]}')
            crop_tool(row)

def run():
    # Interval video mp4 file
    df_intervals = load_valid_intervals(DF_INTERVALS_NOAH)
    logger.info(f'Fetch intervals, {df_intervals.shape} shape dataframe.')
    interval_video_status_and_crop(df_intervals)


if __name__ == '__main__':
    run()
