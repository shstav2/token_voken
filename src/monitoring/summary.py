import logging
import pandas as pd
from src.common.data_loader import load_valid_intervals
from src.common.display_utils import print_value_counts
from src.monitoring.status import status_interval_video_downloaded, \
    status_interval_video_frames_dir, status_detected_faces_dir
from src.monitoring.validations import validate_faces_count_eq_frames_count, \
    validate_embeddings_count_eq_frames_count

logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_status_dataframe():
    df_intervals = load_valid_intervals()
    return status_summary_intervals(df_intervals)

def status_summary_intervals(df_intervals):
    df_intervals.sort_values(by=['video_id', 'interval_id'], inplace=True)
    interval_ids = df_intervals['interval_id'].to_list()
    status_records = []
    for interval_id in interval_ids:
        status = status_summary_interval(interval_id)
        status_records.append([interval_id] + status)
    df_status = pd.DataFrame(status_records,
        columns=['interval_id',
                 'status_video', 'status_frames', 'status_faces',
                 'valid_faces_eq_frames', 'valid_embeddings_eq_frames'])
    df_status.set_index('interval_id', inplace=True)
    print_value_counts(df_status, logger)
    return df_status

def status_summary_interval(interval_id, debug=False):
    # status
    status_video_file = status_interval_video_downloaded(interval_id, debug=debug)
    status_frames_dir = status_interval_video_frames_dir(interval_id, debug=debug)
    status_faces_dir = status_detected_faces_dir(interval_id, debug=debug)
    # validation
    valid_faces_eq_frames = validate_faces_count_eq_frames_count(interval_id)
    valid_embeddings_eq_frames = validate_embeddings_count_eq_frames_count(interval_id)

    return [
        status_video_file, status_frames_dir, status_faces_dir,
        valid_faces_eq_frames, valid_embeddings_eq_frames
    ]
