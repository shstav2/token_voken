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
TOKEN_VOKEN_EXTENSTION    = 'hdf5'


# Vokenization Datasets
# /home/stav/Data/Vokenization/Datasets/Oliver_V1
DATASETS_VOKENIZATION        = os.path.join(STAV_ROOT, 'Vokenization/Datasets')
DF_TOKEN_VOKEN_PKL_FILENAME  = 'df_token_voken_pkl.csv'
DF_TOKEN_VOKEN_CSV_FILENAME  = 'df_token_voken_partial_cols.csv'
TOKENS_DATA_FILENAME         = 'tokens.hdf5'
VOKENS_DATA_FILENAME         = 'vokens.hdf5'
INDICES_FILENAME             = 'indices.txt'

# Magic Numbers
VIDEO_ID_LEN = 11
FRAME_RATE = 15
FACE_IMAGE_SIZE = 224
EMBEDDING_DIM = 16
BLOCK_SIZE = 126 # Set by vokenization as the size of each row in the batch


# dataframe columns names
COL_SPEAKER                   = 'speaker'
COL_SET_TYPE                  = 'set_type'
COL_VIDEO_ID                  = 'video_id'
COL_INTERVAL_ID               = 'interval_id'
COL_BERT_TOKEN_ID             = 'token_id'
COL_WORD                      = 'word'
COL_VOKEN                     = 'voken'
COL_VOKEN_ID                  = 'voken_id'
COL_WORD_FRAME_SELECTED       = 'selected_frame'
COL_WORD_FRAME_SELECTED_FIXED = 'selected_frame_fix' # incase the selected frame id does not exist
COL_VOKEN_PATH                = 'voken_path'

# Train/Test split
SPLIT_INDEX = {
    'Oliver_V3':  64713,
    'Noah_V1':   130232
}


# Tree
PATS_VIDEOS_DIR = os.path.join(PATS_DATA_ROOT, VIDEOS)


# Projects
PROJECT_TOKEN_VOKEN = pathlib.Path(__file__).parent.parent.parent.absolute()
PROJECT_FECNET = '/home/stav/Projects/FECNet'


OLIVER_FACE_PATH = '/home/stav/Data/Sample/oliver/face.jpg'
OLIVER_FACE_RESNET_EMBEDDING_PATH = '/home/stav/Data/Sample/oliver/face.npy'

DF_INTERVALS_OLIVER    = os.path.join(PATS_DATA_ROOT, 'DataFrames/df_intervals_all.csv')
DF_INTERVALS_NOAH_V1   = os.path.join(PATS_DATA_ROOT, 'DataFrames/noah/df_intervals_noah_v1.csv') # 2,657 valid
DF_INTERVALS_NOAH_V2   = os.path.join(PATS_DATA_ROOT, 'DataFrames/noah/df_intervals_noah_v2.csv') # 3,744 valid

# set current context
DF_INTERVALS_NOAH       = DF_INTERVALS_NOAH_V2