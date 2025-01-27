# coding=utf-8
import os
import pandas as pd

from common.constants import PATS_SPEAKER_VIZ_DIR, VIDEO_FRAMES_DIR_NAME

# Data/PATS_DATA/
# └── Videos
#     └── oliver
#         ├── 0Rnq1NpHdmw
#         │   ├── 0Rnq1NpHdmw.mp4
#         │   ├── 101462
#         │   │   └── 101462.mp4
#         │   ├── 101463

# ------- Utils

def get_interval_row(df_intervals, interval_id):
    row = df_intervals[df_intervals['interval_id'] == interval_id].iloc[0]
    return row

def get_video_id(df_intervals, interval_id):
    row = get_interval_row(df_intervals, interval_id)
    return row['video_id']

def get_duration(df_intervals, interval_id):
    row = get_interval_row(df_intervals, interval_id)
    return row['delta_time']

def get_frame_count(interval_id):
    return read_text(interval_id).iloc[-1].end_frame


# ------- 2) Full Video (mp4)

def resolve_video_dir_path(video_id):
    video_dir = os.path.join(PATS_SPEAKER_VIZ_DIR, video_id)
    return video_dir

def resolve_video_file_path(video_id):
    video_dir = resolve_video_dir_path(video_id)
    interval_path = os.path.join(video_dir, f'{video_id}.mp4')
    return interval_path


# ------- 3) Interval Video (mp4)

def resolve_interval_video_path(df_intervals, interval_id):
    video_id = get_video_id(df_intervals, interval_id)
    video_dir = resolve_video_dir_path(video_id)
    interval_path = os.path.join(video_dir, interval_id, f'{interval_id}.mp4')
    return interval_path

def resolve_interval_dir(df_intervals, interval_id):
    video_id = get_video_id(df_intervals, interval_id)
    video_dir = resolve_video_dir_path(video_id)
    interval_path = os.path.join(video_dir, interval_id)
    return interval_path


# ------- 4) Interval Frames (jpg)

def resolve_interval_frames_dir(df_intervals, interval_id, create=False):
    interval_video_dir = resolve_interval_dir(df_intervals, interval_id)
    inetrval_frames_dir = os.path.join(interval_video_dir, VIDEO_FRAMES_DIR_NAME)
    if create and not os.path.exists(inetrval_frames_dir):
        os.makedirs(inetrval_frames_dir)
    return inetrval_frames_dir

def resolve_speaker_intervals_text_dir():
    # '/Users/staveshemesh/Projects/PATS_DATA/Processed/oliver/data/', 'processed/oliver'
    return os.path.join(PATS_SPEAKER_DATA_DIR, 'processed', SPEAKER_NAME)

def resolve_interval_text_path(interval_id):
    speaker_intervals_texts = resolve_speaker_intervals_text_dir()
    interval_text_path = os.path.join(speaker_intervals_texts, f'{interval_id}.h5')
    return interval_text_path

def read_text(interval_id, debug=False):
    interval_text_path = resolve_interval_text_path(interval_id)
    if debug:
        print('resolve_interval_text_path: ', interval_text_path)
    df_token_frames_interval = pd.read_hdf(interval_text_path)
    df_token_frames_interval['start_frame'] = df_token_frames_interval['start_frame'].astype(int)
    df_token_frames_interval['end_frame'] = df_token_frames_interval['end_frame'].astype(int)
    df_token_frames_interval['frames_count'] = df_token_frames_interval['end_frame'] - df_token_frames_interval['start_frame']
    return df_token_frames_interval

def resolve_interval_face_annot_224_dir(df_intervals, interval_id, create=False):
    interval_video_path = resolve_interval_video_path(df_intervals, interval_id)
    interval_video_dir = os.path.dirname(interval_video_path)
    inetrval_face_annot_dir = os.path.join(interval_video_dir, FACE_IMAGE_DIR_NAME)
    if create and not os.path.exists(inetrval_face_annot_dir):
        os.makedirs(inetrval_face_annot_dir)
    return inetrval_face_annot_dir

def resolve_224_voken_path(df_intervals, interval_id, frame_id):
    single_frame_face_annot_dir = resolve_interval_face_annot_224_dir(df_intervals, interval_id, frame_id)
    detected_face_frame_path = os.path.join(single_frame_face_annot_dir, f'{frame_id:05d}.{FRAME_EXTENSION}').format(frame_id)
    return detected_face_frame_path

def resolve_frame_face_annot_dir(df_intervals, interval_id, frame_id, create=False):
    face_annot_dir = resolve_interval_face_annot_224_dir(df_intervals, interval_id)
    single_frame_face_annot_dir = os.path.join(face_annot_dir, f"{frame_id:05d}")
    if create and not os.path.exists(single_frame_face_annot_dir):
        os.makedirs(single_frame_face_annot_dir)
    return single_frame_face_annot_dir

def resolve_detected_face_path(df_intervals, interval_id, frame_id, face_id, create):
    single_frame_face_annot_dir = resolve_frame_face_annot_dir(df_intervals, interval_id, frame_id, create)
    detected_face_frame_path = os.path.join(single_frame_face_annot_dir, 'detected_face_{}.png').format(face_id)
    return detected_face_frame_path

def resolve_annot_faces_path(df_intervals, interval_id, frame_id):
    single_frame_face_annot_dir = resolve_frame_face_annot_dir(df_intervals, interval_id, frame_id)
    return os.path.join(single_frame_face_annot_dir, 'annotated_faces.png')
