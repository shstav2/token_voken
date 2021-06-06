import os

from src.common.path_resolvers import *

DETECTED_FACES_COUNTER_THRESHOLD = 10
FRAMES_DIR_SIZE_THRESHOLD = 100
FRAME_RATE = 15

# ----------- Utils

def get_number_of_directories(path):
    dir_count = 0
    for root, dirs, files in os.walk(path):
        dir_count += len(dirs)
    return dir_count


def status_detected_faces_directories_exist(df_intervals, interval_id):
    # /vokens/face_annot_224/  ({000}/detected_face.png)
    interval_face_annot_224 = resolve_interval_face_annot_224_dir(df_intervals, interval_id, create=False)
    interval_face_annot_224_exists = os.path.exists(interval_face_annot_224)
    return interval_face_annot_224_exists and \
           DETECTED_FACES_COUNTER_THRESHOLD < get_number_of_directories(interval_face_annot_224)


# ------- Video (mp4)

def status_video_downloaded(video_id):
    video_path = resolve_video_file_path(video_id)
    return os.path.exists(video_path)


def status_interval_video_downloaded(df_intervals, interval_id):
    video_path = resolve_interval_video_path(df_intervals, interval_id)
    return os.path.exists(video_path)



def status_frames_dir_content_size(df_intervals, interval_id):
    interval_frames_dir = resolve_interval_frames_dir(df_intervals, interval_id, create=False)
    interval_frames_dir_exists = os.path.exists(interval_frames_dir)
    return interval_frames_dir_exists and \
        FRAMES_DIR_SIZE_THRESHOLD < os.path.getsize(interval_frames_dir)


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
