import os
import pathlib
from src.common.conf import STAV_ROOT, STAV_LOCAL_ROOT

# Base
PATS_DATA_ROOT = os.path.join(STAV_ROOT, 'PATS_DATA')
LOCAL_PATS_DATA_ROOT = STAV_LOCAL_ROOT
SPEAKER_NAME = 'noah'

# Source Channels
VIDEOS                    =  'Videos'    # 105810.mp4
VIDEO_FRAMES_DIR_NAME     =  'Frames'    # 00029.jpg
ALL_FACES_IMAGE_DIR_NAME  =  'FacesAll'  # face_0.jpg
FACES_IMAGE_DIR_NAME      =  'Faces'     # 00029.jpg (224x224)
RESNET_EMBEDDING_DIR_NAME =  'ResNet'    # 512 vector
FECNET_EMBEDDING_DIR_NAME =  'FECNet'    # 8 vector
TEXT_DIR_NAME             =  'Text'
TEXT_RAW_FILENAME         =  'Raw'       # DataFrame [word|start_frame|end_frame|frames_count]
TEXT_TOKENS_FILENAME      =  'Tokens'    # DataFrame

# Interval Parsing
FRAME_EXTENSION           = 'jpg'
EMBEDDING_EXTENSION       = 'npy'
TEXT_EXTENSION            = 'csv'
TOKEN_VOKEN_EXTENSTION    = 'hdf'


# Vokenization Datasets
# /home/stav/Data/Vokenization/Datasets/Oliver_V1
DATASETS_VOKENIZATION    = os.path.join(STAV_ROOT, 'Vokenization/Datasets')
DATASETS_VOKENIZATION_V1 = os.path.join(DATASETS_VOKENIZATION, 'Oliver_V1')


# Magic Numbers
VIDEO_ID_LEN = 11
FRAME_RATE = 15
FACE_IMAGE_SIZE = 224
EMBEDDING_DIM = 16

# Tree
PATS_SPEAKER_VIZ_DIR = os.path.join(PATS_DATA_ROOT, f'{VIDEOS}/{SPEAKER_NAME}')

# PATS_SPEAKER_DATASET_DIR = os.path.join(PATS_DATA_ROOT, f'Datasets/{SPEAKER_NAME}')
# PATS_SPEAKER_DATA_DIR = os.path.join(PATS_DATA_ROOT, f'Processed/{SPEAKER_NAME}/data')
# PATS_SPEAKER_INTERVAL_DIR = os.path.join(PATS_SPEAKER_DATA_DIR, 'processed', SPEAKER_NAME)

# Projects
PROJECT_TOKEN_VOKEN = pathlib.Path(__file__).parent.parent.parent.absolute()
PROJECT_FECNET = '/home/stav/Projects/FECNet'


OLIVER_FACE_PATH = '/home/stav/Data/Sample/oliver/face.jpg'
OLIVER_FACE_RESNET_EMBEDDING_PATH = '/home/stav/Data/Sample/oliver/face.npy'

DF_INTERVALS_NOAH = '/home/stav/Data/PATS_DATA/df_intervals_noah.csv'
