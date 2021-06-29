import logging
from tqdm import tqdm

from src.common.data_loader import load_valid_intervals
from src.components._6_face_recognition import find_face_recognition_errors


logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)


def find_and_fix_face_recognition_errors(df_intervals):
    logging.info('-------- Faces ➜ Relevant Face -----------')
    interval_ids = df_intervals['interval_id'].tolist()
    # Find recognition errors
    for interval_id in tqdm(interval_ids):
        mislabeled_frames, _ = find_face_recognition_errors(interval_id)
    # Fix recognition errors
    for
    _copy_recognized_face()


def run():
    df_intervals = load_valid_intervals()
    find_and_fix_face_recognition_errors(df_intervals)


if __name__ == '__main__':
    run()
