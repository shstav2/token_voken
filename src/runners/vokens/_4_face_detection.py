import numpy as np
import logging

from src.common.data_loader import load_valid_intervals
from src.common.constants import DF_INTERVALS_NOAH
from src.monitoring.status import status_detected_faces_dir, status_interval_video_frames_dir
from src.components.vokens._4_face_detection import interval_extract_faces


logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)


def detect_faces_in_frames(df_intervals):
    # Interval faces
    logging.info('-------- Frames ➜ Faces -----------')
    df_intervals['status_interval_faces_dir'] = df_intervals['interval_id'].apply(status_detected_faces_dir)
    logger.info('[Status] Interval detected faces:\n' \
                f"{df_intervals['status_interval_faces_dir'].value_counts()}")
    # Make sure pervious step finished successfully
    df_intervals['status_interval_frames_dir'] = df_intervals['interval_id'].apply(status_interval_video_frames_dir)

    # Extract faces
    df_intervals_pending = df_intervals[(df_intervals['status_interval_frames_dir']) & (~df_intervals['status_interval_faces_dir'])].copy()
    df_intervals_pending.sort_values(by=['video_id', 'interval_id'], ascending=False, inplace=True)
    pending_count = df_intervals_pending.shape[0]
    logger.info(f'Extract faces from frames for {pending_count} intervals...:')
    interval_ids = df_intervals_pending['interval_id'].tolist()
    for interval_id in interval_ids:
        interval_extract_faces(interval_id)


def run():
    # Detect faces in frames
    df_intervals = load_valid_intervals(DF_INTERVALS_NOAH)
    detect_faces_in_frames(df_intervals)


if __name__ == '__main__':
    run()
