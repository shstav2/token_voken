import os
import logging
from pathlib import Path

from src.common.path_resolvers import resolve_video_file_path, \
    resolve_interval_video_path, resolve_interval_frames_dir, resolve_interval_faces_dir
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
    root_directory = Path(path)
    root_listdir = root_directory.glob('**/*')
    return sum(f.stat().st_size for f in root_listdir if f.is_file())


def get_number_of_directories(path):
    dir_count = 0
    for root, dirs, files in os.walk(path):
        dir_count += len(dirs)
    return dir_count


# ------- Step 2 Full Video (mp4)

def status_video_downloaded(video_id):
    video_path = resolve_video_file_path(video_id)
    return os.path.exists(video_path)


# ------- Step 3 Cropped Interval Videos (mp4)

def status_interval_video_downloaded(df_intervals, interval_id):
    video_path = resolve_interval_video_path(df_intervals, interval_id)
    interval_video_exists = os.path.exists(video_path)
    video_size_kilobytes = os.path.getsize(video_path)
    valid_video_size = VIDEO_FILE_SIZE_THRESHOLD < video_size_kilobytes
    video_status_ok = interval_video_exists and valid_video_size
    if one_percent_chance():
        symbol = bool_to_symbol(video_status_ok)
        logger.info(f'\t[Status] {symbol} Interval video {video_path} (size: {video_size_kilobytes:,} KB)')
    return video_status_ok


# ------- Step 4 Interval Videos → Frames(jpg)

def status_interval_video_frames_dir(df_intervals, interval_id):
    interval_frames_dir = resolve_interval_frames_dir(df_intervals, interval_id, create=False)
    interval_frames_dir_exists = os.path.exists(interval_frames_dir)
    interval_frames_dir_size_bytes = get_total_size(interval_frames_dir) if interval_frames_dir_exists else -1
    interval_frames_dir_size_kilo_bytes = interval_frames_dir_size_bytes // 1000
    interval_frames_dir_contains_files = FRAMES_DIR_SIZE_THRESHOLD < interval_frames_dir_size_bytes
    interval_frames_exists = interval_frames_dir_exists and interval_frames_dir_contains_files
    if one_percent_chance():
        symbol = bool_to_symbol(interval_frames_exists)
        logger.info(f'\t[Status] {symbol} Interval video {interval_frames_dir} '\
                        f'(size: {interval_frames_dir_size_kilo_bytes:,} KB)')
    return interval_frames_exists


# ------- Step 5 Frames → Faces

def status_detected_faces_directories_exist(df_intervals, interval_id):
    interval_faces_dir = resolve_interval_faces_dir(df_intervals, interval_id, create=False)
    interval_faces_dir_exists = os.path.exists(interval_faces_dir)
    if one_percent_chance():
        symbol = bool_to_symbol(interval_faces_dir_exists)
        logger.info(f'\t[Status] {symbol} Interval faces {interval_faces_dir_exists}')
    return interval_faces_dir
    # and \
    #        DETECTED_FACES_COUNTER_THRESHOLD_BYTES < get_number_of_directories(interval_face_annot_224)
#


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
