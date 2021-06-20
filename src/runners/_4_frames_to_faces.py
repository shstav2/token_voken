import math
import numpy as np
import logging

from src.monitoring.status import status_detected_faces_dir
from src.common.data_loader import load_valid_intervals
from src.components._4_frames_to_faces import create_face_images

BATCH_SIZE = 10


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

    # Extract faces
    df_intervals_pending = df_intervals[~df_intervals['status_interval_faces_dir']].copy()
    df_intervals_pending.sort_values(by=['video_id', 'interval_id'], ascending=False, inplace=True)
    pending_count = df_intervals_pending.shape[0]
    number_of_batches = math.ceil(pending_count / BATCH_SIZE)
    logger.info(f'Extract faces from frames for {pending_count} intervals...:')
    for i, chunk in enumerate(np.array_split(df_intervals_pending, number_of_batches), 1):
        logger.info(f'\tExtract interval video frame, batch #{i} sized {chunk.shape}...:')
        for _, row in chunk.iterrows():
            interval_id = row['interval_id']
            create_face_images(interval_id)


def run():
    # Detect faces in frames
    df_intervals = load_valid_intervals()
    detect_faces_in_frames(df_intervals)


if __name__ == '__main__':
    run()
