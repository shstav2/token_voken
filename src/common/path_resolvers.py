# coding=utf-8
import os

from src.common.constants import \
    PATS_VIDEOS_DIR, PATS_DATA_ROOT, LOCAL_PATS_DATA_ROOT, \
    VIDEO_FRAMES_DIR_NAME,\
    ALL_FACES_IMAGE_DIR_NAME, FACES_IMAGE_DIR_NAME, \
    RESNET_EMBEDDING_DIR_NAME, FECNET_EMBEDDING_DIR_NAME, \
    TEXT_DIR_NAME, TEXT_RAW_FILENAME, TEXT_TOKENS_FILENAME, \
    \
    FRAME_EXTENSION, EMBEDDING_EXTENSION, TEXT_EXTENSION, \
    TOKEN_VOKEN_EXTENSTION, \
    \
    DATASETS_VOKENIZATION, \
    DF_TOKEN_VOKEN_CSV_FILENAME, DF_TOKEN_VOKEN_PKL_FILENAME, \
    TOKENS_DATA_FILENAME, VOKENS_DATA_FILENAME
from src.data.interval_to_video.all import INTERVAL_TO_VIDEO, video_id_to_speaker

# Data/PATS_DATA/
# └── Videos
#     └── oliver
#         └── 0Rnq1NpHdmw
#             ├── 0Rnq1NpHdmw.mp4
#             ├── 101462
#             │   └── 101462.mp4
#             │   └── Frames
#             │   |   ├── 00000.jpg
#             │   |   ├── 00001.jpg
#             │   |   └── 00002.jpg
#             │   └── Faces
#             │   |   ├── 00000.jpg
#             │   |   ├── 00001.jpg
#             │   |   └── 00002.jpg
#             │   └── FacesAll
#             |   |   ├── 00000
#             |   |   │   ├── annotated_faces.jpg
#             |   |   │   ├── face_0.jpg
#             |   |   │   └── face_1.jpg
#             |   |   └── 00001
#             │   |       ├── annotated_faces.jpg
#             │   |       └── face_0.jpg
#             │   └── FECNet


# ------- Utils

def get_video_id(interval_id):
    return INTERVAL_TO_VIDEO[interval_id]

def get_interval_row(df_intervals, interval_id):
    row = df_intervals[df_intervals['interval_id'] == interval_id].iloc[0]
    return row

def get_duration(df_intervals, interval_id):
    row = get_interval_row(df_intervals, interval_id)
    return row['delta_time']

def get_frame_count(interval_id):
    return read_text(interval_id).iloc[-1].end_frame


def localize_path(remote_path):
    return remote_path.replace(PATS_DATA_ROOT, LOCAL_PATS_DATA_ROOT)


# ------- 1) Full Video (mp4)

def resolve_video_dir_path(video_id):
    speaker_name = video_id_to_speaker(video_id)
    video_dir = os.path.join(PATS_VIDEOS_DIR, speaker_name, video_id)
    return video_dir

def resolve_video_file_path(video_id):
    video_dir = resolve_video_dir_path(video_id)
    interval_path = os.path.join(video_dir, f'{video_id}.mp4')
    return interval_path


# ------- 2) Interval Video (mp4)

# Videos/oliver/0Rnq1NpHdmw/101462/101462.mp4
def resolve_interval_video_path(interval_id):
    video_id = get_video_id(interval_id)
    video_dir = resolve_video_dir_path(video_id)
    interval_path = os.path.join(video_dir, interval_id, f'{interval_id}.mp4')
    return interval_path

# Videos/oliver/0Rnq1NpHdmw/101462/
def resolve_interval_dir(interval_id):
    video_id = get_video_id(interval_id)
    video_dir = resolve_video_dir_path(video_id)
    interval_path = os.path.join(video_dir, interval_id)
    return interval_path


# ------- 3) Interval Frames (jpg)

# Videos/oliver/0Rnq1NpHdmw/101462/Frames
def resolve_interval_frames_dir(interval_id, create=False):
    interval_video_dir = resolve_interval_dir(interval_id)
    inetrval_frames_dir = os.path.join(interval_video_dir, VIDEO_FRAMES_DIR_NAME)
    if create and not os.path.exists(inetrval_frames_dir):
        os.makedirs(inetrval_frames_dir)
    return inetrval_frames_dir

def resolve_interval_frame_path(interval_id, frame_id, create=False):
    inetrval_frames_dir = resolve_interval_frames_dir(interval_id, create)
    single_frame_path = os.path.join(inetrval_frames_dir, f"{frame_id:05d}.{FRAME_EXTENSION}")
    return single_frame_path


# ------- 4) FacesAll

# [FacesAll] Videos/oliver/0Rnq1NpHdmw/101462/FacesAll
def resolve_interval_all_faces_dir(interval_id, create=False):
    interval_video_path = resolve_interval_dir(interval_id)
    interval_face_annot_dir = os.path.join(interval_video_path, ALL_FACES_IMAGE_DIR_NAME)
    if create and not os.path.exists(interval_face_annot_dir):
        os.makedirs(interval_face_annot_dir)
    return interval_face_annot_dir

# [FacesAll] Videos/oliver/0Rnq1NpHdmw/101462/FacesAll/00012
def resolve_single_frame_faces_dir(interval_id, frame_id, create=False):
    face_annot_dir = resolve_interval_all_faces_dir(interval_id)
    single_frame_face_annot_dir = os.path.join(face_annot_dir, f"{frame_id:05d}")
    if create and not os.path.exists(single_frame_face_annot_dir):
        os.makedirs(single_frame_face_annot_dir)
    return single_frame_face_annot_dir

# [FacesAll] Videos/oliver/0Rnq1NpHdmw/101462/FacesAll/00012/face_0.jpg
def resolve_detected_face_path(interval_id, frame_id, face_id, create=False):
    single_frame_faces_dir = resolve_single_frame_faces_dir(interval_id, frame_id, create)
    detected_face_frame_path = os.path.join(single_frame_faces_dir, f'face_{face_id}.{FRAME_EXTENSION}')
    return detected_face_frame_path

# [FacesAll] Videos/oliver/0Rnq1NpHdmw/101462/FacesAll/00012/annotated_faces.jpg
def resolve_annot_faces_path(interval_id, frame_id, create=False):
    single_frame_face_annot_dir = resolve_single_frame_faces_dir(interval_id, frame_id, create)
    return os.path.join(single_frame_face_annot_dir, f'annotated_faces.{FRAME_EXTENSION}')


# ------- 5) ResNet Face Image Embeddings

# [ResNet] Videos/oliver/0Rnq1NpHdmw/101462/ResNet
def resolve_interval_resnet_embeddings_dir(interval_id, create=False):
    interval_video_path = resolve_interval_dir(interval_id)
    interval_face_embeddings_dir = os.path.join(interval_video_path, RESNET_EMBEDDING_DIR_NAME)
    if create and not os.path.exists(interval_face_embeddings_dir):
        os.makedirs(interval_face_embeddings_dir)
    return interval_face_embeddings_dir

# [ResNet] Videos/oliver/0Rnq1NpHdmw/101462/ResNet/00012
def resolve_single_frame_resnet_faces_dir(interval_id, frame_id, create=False):
    face_annot_dir = resolve_interval_resnet_embeddings_dir(interval_id, create)
    single_frame_resnet_faces_dir = os.path.join(face_annot_dir, f"{frame_id:05d}")
    if create and not os.path.exists(single_frame_resnet_faces_dir):
        os.makedirs(single_frame_resnet_faces_dir)
    return single_frame_resnet_faces_dir

# [ResNet] Videos/oliver/0Rnq1NpHdmw/101462/ResNet/00012/face_0.npy
def resolve_face_resnet_embedding_path(interval_id, frame_id, face_id, create=False):
    single_frame_resnet_faces_dir = resolve_single_frame_resnet_faces_dir(interval_id, frame_id, create)
    face_resnet_embedding_path = os.path.join(single_frame_resnet_faces_dir, f'face_{face_id}.{EMBEDDING_EXTENSION}')
    return face_resnet_embedding_path


# ------- 6) Selected Face

# [Faces] /home/stav/Data/PATS_DATA/Videos/oliver/0Rnq1NpHdmw/101462/Faces
def resolve_interval_faces_dir(interval_id, create=False):
    interval_video_path = resolve_interval_dir(interval_id)
    interval_faces_dir = os.path.join(interval_video_path, FACES_IMAGE_DIR_NAME)
    if create and not os.path.exists(interval_faces_dir):
        os.makedirs(interval_faces_dir)
    return interval_faces_dir

# [Faces] [Local] /Users/staveshemesh/Data/Videos/oliver/0Rnq1NpHdmw/101462/Faces
def resolve_interval_local_faces_dir(interval_id):
    interval_remote_faces_dir = resolve_interval_faces_dir(interval_id)
    interval_local_faces_dir = localize_path(interval_remote_faces_dir)
    return interval_local_faces_dir

# [Faces] Videos/oliver/0Rnq1NpHdmw/101462/Faces/00012.jpg
def resolve_frame_face_path(interval_id, frame_id, create=False):
    frame_faces_dir = resolve_interval_faces_dir(interval_id, create)
    frame_face_path = os.path.join(frame_faces_dir, f"{frame_id:05d}.{FRAME_EXTENSION}")
    return frame_face_path


# ------- 7) FECNet Facial Embeddings

# [FECNet] Videos/oliver/0Rnq1NpHdmw/101462/FECNet
def resolve_interval_facial_embeddings_dir(interval_id, create=False):
    interval_video_path = resolve_interval_dir(interval_id)
    interval_face_embeddings_dir = os.path.join(interval_video_path, FECNET_EMBEDDING_DIR_NAME)
    if create and not os.path.exists(interval_face_embeddings_dir):
        os.makedirs(interval_face_embeddings_dir)
    return interval_face_embeddings_dir

# [FECNet] Videos/oliver/0Rnq1NpHdmw/101462/FECNet/00012.npy
def resolve_interval_facial_embedding_path(interval_id, frame_id, create=False):
    facial_embeddings_dir = resolve_interval_facial_embeddings_dir(interval_id, create)
    face_embedding_path = os.path.join(facial_embeddings_dir, f"{frame_id:05d}.{EMBEDDING_EXTENSION}")
    return face_embedding_path


# ------- 1) Raw PATS Text (Words)

# [Text] Videos/oliver/0Rnq1NpHdmw/101462/Text/
def resolve_interval_text_dir(interval_id, create=False):
    interval_path = resolve_interval_dir(interval_id)
    text_dir = os.path.join(interval_path, TEXT_DIR_NAME)
    if create and not os.path.exists(text_dir):
        os.makedirs(text_dir)
    return text_dir

# [Text] [Local] Videos/oliver/0Rnq1NpHdmw/101462/Text/
def resolve_interval_text_dir(interval_id, create=False):
    interval_path = resolve_interval_dir(interval_id)
    text_dir = os.path.join(interval_path, TEXT_DIR_NAME)
    if create and not os.path.exists(text_dir):
        os.makedirs(text_dir)
    return text_dir

# [Text/Raw] Videos/oliver/0Rnq1NpHdmw/101462/Text/Raw.csv
def resolve_interval_raw_text_path(interval_id, create=False):
    interval_text_dir = resolve_interval_text_dir(interval_id, create)
    return os.path.join(interval_text_dir, f'{TEXT_RAW_FILENAME}.{TEXT_EXTENSION}')

# [Text/Raw] [Local] Videos/oliver/0Rnq1NpHdmw/101462/Text/Raw.csv
def resolve_interval_local_raw_text_path(interval_id, create=False):
    interval_remote_faces_dir = resolve_interval_raw_text_path(interval_id, create)
    interval_local_faces_dir = localize_path(interval_remote_faces_dir)
    return interval_local_faces_dir

# ------- 2) Raw PATS Text

# [Text/Tokens] Videos/oliver/0Rnq1NpHdmw/101462/Text/Tokens.csv
def resolve_interval_text_tokenized_path(interval_id, create=False):
    interval_text_dir = resolve_interval_text_dir(interval_id, create)
    return os.path.join(interval_text_dir, f'{TEXT_TOKENS_FILENAME}.{TEXT_EXTENSION}')


# -------- Token-Voken Dataset
"""
(base) stav@eimtest-ESC4000-G4:~/Data/Vokenization/Datasets/Oliver_V3_Noah_V1$ tree .
./home/stav/Data/Vokenization/Datasets/Oliver_V3_Noah_V1
├── df_token_voken.csv
├── df_token_voken_pkl.csv
├── test
│   ├── tokens.hdf5
│   └── vokens.hdf5
├── train
│   ├── tokens.hdf5
│   └── vokens.hdf5
└── vokens.npy

2 directories, 7 files
"""

# [Dataset] /home/stav/Data/Vokenization/Datasets/Oliver_V1
def resolve_dataset_dir(dataset_name):
    return os.path.join(DATASETS_VOKENIZATION, dataset_name)

# [Dataset/Dataframe/pkl] /home/stav/Data/Vokenization/Datasets/Oliver_V1/df_token_voken_pkl.csv
def resolve_dataset_pickle_dataframe(dataset_name):
    dataset_dir = resolve_dataset_dir(dataset_name)
    return os.path.join(dataset_dir, DF_TOKEN_VOKEN_PKL_FILENAME)

# [Dataset/Dataframe/csv] /home/stav/Data/Vokenization/Datasets/Oliver_V1/df_token_voken_partial_cols.csv
def resolve_dataset_csv_dataframe(dataset_name):
    dataset_dir = resolve_dataset_dir(dataset_name)
    return os.path.join(dataset_dir, DF_TOKEN_VOKEN_CSV_FILENAME)

# [Dataset/Train] /home/stav/Data/Vokenization/Datasets/Oliver_V1/train
def resolve_subset_data_dir(dataset_name, subset):
    dataset_dir = resolve_dataset_dir(dataset_name)
    return os.path.join(dataset_dir, subset)

# [Dataset/Train/Tokens] /home/stav/Data/Vokenization/Datasets/Oliver_V1/train/tokens.hdf
def resolve_dataset_tokens_path(dataset_name, subset):
    # /home/stav/Data/Vokenization/Datasets/Oliver_V1/train
    subset_path = resolve_subset_data_dir(dataset_name, subset)
    return os.path.join(subset_path, TOKENS_DATA_FILENAME)

# [Dataset/Train/Vokens] /home/stav/Data/Vokenization/Datasets/Oliver_V1/train/vokens.hdf
def resolve_dataset_vokens_path(dataset_name, subset):
    # /home/stav/Data/Vokenization/Datasets/Oliver_V1/train
    subset_path = resolve_subset_data_dir(dataset_name, subset)
    return os.path.join(subset_path, VOKENS_DATA_FILENAME)

