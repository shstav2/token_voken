import os
import logging
from pathlib import Path

from src.common.path_resolvers import resolve_video_file_path, \
    resolve_interval_video_path, resolve_interval_frames_dir, resolve_interval_all_faces_dir
from src.common.debug import one_percent_chance
from src.common.display_utils import bool_to_symbol


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
FRAME_RATE = 15


# ----------- Utils

def get_total_size(path):
    path_object = Path(path)
    if path_object.is_file():
        return os.path.getsize(path)
    root_listdir = path_object.glob('**/*')
    return sum(f.stat().st_size for f in root_listdir if f.is_file())

def get_number_of_directories(path):
    dir_count = 0
    for root, dirs, files in os.walk(path):
        dir_count += len(dirs)
    return dir_count

def human_readable_size(size_bytes, decimal_places=2):
    size = size_bytes
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']:
        if size < 1024.0 or unit == 'PiB':
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"

def exists_and_has_content(file_or_directory_path, size_threshold_bytes):
    path_exists = os.path.exists(file_or_directory_path)
    path_size_bytes = get_total_size(file_or_directory_path) if path_exists else -1
    path_suffient_size = size_threshold_bytes < path_size_bytes
    path_exists_and_has_content = path_exists and path_suffient_size
    if one_percent_chance():
        symbol = bool_to_symbol(path_exists_and_has_content)
        logger.info(f'\t[Status] {symbol} {file_or_directory_path} (size: {human_readable_size(path_size_bytes)})')
    return path_exists_and_has_content


# ------- Step 2 Full Video (mp4)

def status_video_downloaded(video_id):
    video_path = resolve_video_file_path(video_id)
    return os.path.exists(video_path)


# ------- Step 3 Cropped Interval Videos (mp4)

def status_interval_video_downloaded(df_intervals, interval_id):
    video_path = resolve_interval_video_path(df_intervals, interval_id)
    return exists_and_has_content(video_path, VIDEO_FILE_SIZE_THRESHOLD)


# ------- Step 4 Interval Videos → Frames(jpg)

def status_interval_video_frames_dir(df_intervals, interval_id):
    interval_frames_dir = resolve_interval_frames_dir(df_intervals, interval_id, create=False)
    return exists_and_has_content(interval_frames_dir, FRAMES_DIR_SIZE_THRESHOLD)


# ------- Step 5 Frames → Faces

def status_detected_faces_dir_exist(df_intervals, interval_id):
    interval_faces_dir = resolve_interval_all_faces_dir(df_intervals, interval_id, create=False)
    return exists_and_has_content(interval_faces_dir, FRAMES_DIR_SIZE_THRESHOLD)



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
