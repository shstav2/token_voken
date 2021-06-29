import logging
from tqdm import tqdm

from src.monitoring.status import status_facial_fecnet_embeddings_dir
from src.common.data_loader import load_valid_intervals
from src.components._5_faces_to_vectors import extract_and_save_interval_facial_embeddings


logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)


def extract_facial_embeddings(df_intervals):
    logging.info('-------- Faces ➜ Embeddings -----------')
    df_intervals['status_face_embeddings_dir'] = df_intervals['interval_id'].apply(status_facial_fecnet_embeddings_dir)
    logger.info(f'[Status] Interval face embeddings:\n {df_intervals["status_face_embeddings_dir"].value_counts()}')

    df_intervals_pending = df_intervals[~df_intervals['status_face_embeddings_dir']].copy()
    df_intervals_pending.sort_values(by=['video_id', 'interval_id'], ascending=False, inplace=True)
    pending_count = df_intervals_pending.shape[0]
    logger.info(f'Extract embeddings from faces for {pending_count} intervals...:')
    interval_ids = df_intervals_pending['interval_id'].tolist()
    for interval_id in tqdm(interval_ids):
        extract_and_save_interval_facial_embeddings(interval_id)


def run():
    df_intervals = load_valid_intervals()
    extract_facial_embeddings(df_intervals)


if __name__ == '__main__':
    run()
