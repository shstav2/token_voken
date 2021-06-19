import os
import logging
import pandas as pd

from src.common.path_resolvers import resolve_video_file_path, \
    resolve_interval_video_path, resolve_interval_frames_dir, resolve_interval_all_faces_dir
from src.monitoring.utils import exists_and_has_content

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


# ------- Step 1 Full Video (mp4)

def status_video_downloaded(video_id):
    video_path = resolve_video_file_path(video_id)
    return os.path.exists(video_path)


# ------- Step 2 Cropped Interval Videos (mp4)

def status_interval_video_downloaded(df_intervals, interval_id, debug=False):
    video_path = resolve_interval_video_path(df_intervals, interval_id)
    return exists_and_has_content(video_path, VIDEO_FILE_SIZE_THRESHOLD, debug=debug)


# ------- Step 3 Interval Videos → Frames(jpg)

def status_interval_video_frames_dir(df_intervals, interval_id, debug=False):
    interval_frames_dir = resolve_interval_frames_dir(df_intervals, interval_id, create=False)
    return exists_and_has_content(interval_frames_dir, FRAMES_DIR_SIZE_THRESHOLD, debug=debug)


# ------- Step 5 Frames → Faces

def status_detected_faces_dir_exist(df_intervals, interval_id, debug=False):
    interval_faces_dir = resolve_interval_all_faces_dir(df_intervals, interval_id, create=False)
    return exists_and_has_content(interval_faces_dir, FRAMES_DIR_SIZE_THRESHOLD, debug=debug)



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
