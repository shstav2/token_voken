import os
import logging

from src.common.path_resolvers import resolve_video_file_path, \
    resolve_interval_video_path, resolve_interval_frames_dir, resolve_interval_all_faces_dir, \
    resolve_interval_facial_embeddings_dir, resolve_interval_resnet_embeddings_dir, \
    resolve_interval_raw_text_path, resolve_interval_text_tokenized_path
from src.common.file_utils import exists_and_has_content

logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Interval Video
VIDEO_FILE_SIZE_THRESHOLD = 9999
# Frames
SINGLE_FRAME_SIZE_APPROX_BYTES = 130 * 1000 # 130K
FRAMES_DIR_SIZE_THRESHOLD =  3 * SINGLE_FRAME_SIZE_APPROX_BYTES
# DETECTED_FACES_COUNTER_THRESHOLD_BYTES = 3 * IMAGE_SIZE_APPROX_BYTES
# Embeddings
SINGLE_EMBEDDING_FILE = 192
FACIAL_EMBEDDINGS_DIR_SIZE_THRESHOLD = 10 * SINGLE_EMBEDDING_FILE

FRAME_RATE = 15


# ------- Step 1: Full Video (mp4)

def status_video_downloaded(video_id):
    video_path = resolve_video_file_path(video_id)
    return os.path.exists(video_path)


# ------- Step 2: Cropped Interval Videos (mp4)

def status_interval_video_downloaded(interval_id, debug=False):
    video_path = resolve_interval_video_path(interval_id)
    return exists_and_has_content(video_path, VIDEO_FILE_SIZE_THRESHOLD, debug=debug)


# ------- Step 3: Interval Videos → Frames (jpg)

# [Frames] Videos/oliver/0Rnq1NpHdmw/101462/Frames
def status_interval_video_frames_dir(interval_id, debug=False):
    interval_frames_dir = resolve_interval_frames_dir(interval_id, create=False)
    return exists_and_has_content(interval_frames_dir, FRAMES_DIR_SIZE_THRESHOLD, debug=debug)


# ------- Step 4: Frames → Faces (224x224 jpg)

# [FacesAll] Videos/oliver/0Rnq1NpHdmw/101462/FacesAll/00012/
def status_detected_faces_dir(interval_id, debug=False):
    interval_faces_dir = resolve_interval_all_faces_dir(interval_id, create=False)
    return exists_and_has_content(interval_faces_dir, FRAMES_DIR_SIZE_THRESHOLD, debug=debug)


# ------- Step 5: Frames → ResNet Embeddings (512 dim)

# [ResNet] Videos/oliver/0Rnq1NpHdmw/101462/ResNet
def status_facial_resnet_embeddings_dir(interval_id, debug=False):
    interval_resnet_dir = resolve_interval_resnet_embeddings_dir(interval_id, create=False)
    return exists_and_has_content(interval_resnet_dir, FACIAL_EMBEDDINGS_DIR_SIZE_THRESHOLD, debug=debug)


# ------- Step 7: Frames → FECNet Embeddings (16 dim)

# [FECNet] Videos/oliver/0Rnq1NpHdmw/101462/FECNet
def status_facial_fecnet_embeddings_dir(interval_id, debug=False):
    interval_facial_embeddings_dir = resolve_interval_facial_embeddings_dir(interval_id, create=False)
    return exists_and_has_content(interval_facial_embeddings_dir, FACIAL_EMBEDDINGS_DIR_SIZE_THRESHOLD, debug=debug)



# ------- Step 1: pats .h5 → raw words dataframe

# [Text/Raw] Videos/oliver/0Rnq1NpHdmw/101462/Text/Raw.csv
def status_text_raw_csv(interval_id):
    raw_text_csv_path = resolve_interval_raw_text_path(interval_id)
    return os.path.exists(raw_text_csv_path)


# ------- Step 2: raw words dataframe  → bert tokens dataframe

# [Text/Raw] Videos/oliver/0Rnq1NpHdmw/101462/Text/Raw.csv
def status_text_tokens_csv(interval_id):
    tokens_text_csv_path = resolve_interval_text_tokenized_path(interval_id)
    return os.path.exists(tokens_text_csv_path)



def status_frames(df_intervals):
    # 1 Video file
    df_intervals['status_interval_video_downloaded'] = df_intervals['interval_id'].apply(
        lambda i: status_interval_video_downloaded(df_intervals, i))
    # 2 Frames dir
    df_intervals['frames_dir_exists'] = df_intervals['interval_frames_dir'].apply(os.path.exists)
    df_intervals['frames_count'] = df_intervals['interval_frames_dir'].apply(
        lambda frame_dir: len(os.listdir(frame_dir)) if os.path.exists(frame_dir) else -1)
    df_intervals['supposed_frames_count'] = (df_intervals['duration'] * FRAME_RATE).astype(int)
    df_intervals['missing_frames_count'] = (df_intervals['supposed_frames_count'] - df_intervals['frames_count']).abs()
    df_intervals['has_completed_frames'] = df_intervals['missing_frames_count'] < 20
    # 3 Frames dir content
    df_intervals['frames_dir_content_size'] = df_intervals['interval_id'].apply(
        lambda i: status_frames_dir_content_size(df_intervals, i))
    df_intervals['has_detected_faces'] = df_intervals['interval_id'].apply(
        lambda i: status_detected_faces_directories_exist(df_intervals, i))
    df_intervals['need_to_extract_frames'] = (~df_intervals['has_completed_frames']) & (~df_intervals['has_detected_faces'])
