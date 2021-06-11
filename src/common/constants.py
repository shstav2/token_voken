import pathlib
import os
from common.conf import STAV_ROOT

# ------ PATS Related
# Base
PATS_DATA_ROOT = os.path.join(STAV_ROOT, 'PATS_DATA')
SPEAKER_NAME = 'oliver'
# Sources
VIDEOS = 'Videos'
VIDEO_FRAMES_DIR_NAME = 'Frames'
ALL_FACES_IMAGE_DIR_NAME = 'FacesAll'
FACES_IMAGE_DIR_NAME = 'Faces'

PATS_SPEAKER_VIZ_DIR = os.path.join(PATS_DATA_ROOT, f'{VIDEOS}/{SPEAKER_NAME}')
PATS_SPEAKER_DATASET_DIR = os.path.join(PATS_DATA_ROOT, f'Datasets/{SPEAKER_NAME}')
PATS_SPEAKER_DATA_DIR = os.path.join(PATS_DATA_ROOT, f'Processed/{SPEAKER_NAME}/data')
PATS_SPEAKER_INTERVAL_DIR = os.path.join(PATS_SPEAKER_DATA_DIR, 'processed', SPEAKER_NAME)

# Dataframes
PROJECT_TOKEN_VOKEN = pathlib.Path(__file__).parent.parent.parent.absolute()
INTERVALS_PATH = os.path.join(PROJECT_TOKEN_VOKEN, 'resources/df_intervals_oliver.csv')


# Sys Path Sources
PROJECT_FECNET = '/Users/staveshemesh/Projects/shstav2/FECNet'
# PROJECT_TOKEN_VOKEN = '/Users/staveshemesh/Projects/shstav2/token_voken'

# Interval Parsing
VIDEO_ID_LEN = 11

FRAME_RATE = 15

FRAME_EXTENSION = 'jpg'
# COLS_VIEW = [
#     'speaker', 'interval_id',
#     'duration', 'start_time_string', 'end_time_string',
#     'video_link'
# ]
# COLS_SMALL_FRAME_SIZE = [
#     'video_id', 'interval_id',
#     'duration', 'start_time_string', 'end_time_string',
#     'frames_dir_content_size', 'interval_frames_dir'
# ]