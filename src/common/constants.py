import os
import pathlib
from src.common.conf import STAV_ROOT, STAV_LOCAL_ROOT

# Base
PATS_DATA_ROOT = os.path.join(STAV_ROOT, 'PATS_DATA')
LOCAL_PATS_DATA_ROOT = STAV_LOCAL_ROOT
SPEAKER_NAME = 'oliver'

# Source Channels
VIDEOS                    =  'Videos'    # 105810.mp4
VIDEO_FRAMES_DIR_NAME     =  'Frames'    # 00029.jpg
ALL_FACES_IMAGE_DIR_NAME  =  'FacesAll'  # face_0.jpg
FACES_IMAGE_DIR_NAME      =  'Faces'     # 00029.jpg (224x224)
RESNET_EMBEDDING_DIR_NAME =  'ResNet'    # 512 vector
FECNET_EMBEDDING_DIR_NAME =  'FECNet'    # 8 vector
DIR_NAME_TEXT_PATS        =  'TextPATS'

# Tree
PATS_SPEAKER_VIZ_DIR = os.path.join(PATS_DATA_ROOT, f'{VIDEOS}/{SPEAKER_NAME}')

# PATS_SPEAKER_DATASET_DIR = os.path.join(PATS_DATA_ROOT, f'Datasets/{SPEAKER_NAME}')
# PATS_SPEAKER_DATA_DIR = os.path.join(PATS_DATA_ROOT, f'Processed/{SPEAKER_NAME}/data')
# PATS_SPEAKER_INTERVAL_DIR = os.path.join(PATS_SPEAKER_DATA_DIR, 'processed', SPEAKER_NAME)

# Projects
PROJECT_TOKEN_VOKEN = pathlib.Path(__file__).parent.parent.parent.absolute()
PROJECT_FECNET = '/home/stav/Projects/FECNet'

# Resources
INTERVALS_PATH = os.path.join(PROJECT_TOKEN_VOKEN, 'resources/df_intervals_oliver.csv')

# Interval Parsing
VIDEO_ID_LEN = 11
FRAME_RATE = 15
FRAME_EXTENSION = 'jpg'
FACE_IMAGE_SIZE = 224
EMBEDDING_EXTENSION = 'npy'


OLIVER_FACE_PATH = '/home/stav/Data/Sample/oliver/face.jpg'
OLIVER_FACE_RESNET_EMBEDDING_PATH = '/home/stav/Data/Sample/oliver/face.npy'
