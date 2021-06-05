import os

# PATS related
SPEAKER_NAME = 'oliver'
PATS_DATA_ROOT = '/Users/staveshemesh/Projects/PATS_DATA/'
PATS_SPEAKER_VIZ_DIR = os.path.join(PATS_DATA_ROOT, f'Youtube/{SPEAKER_NAME}')
PATS_SPEAKER_DATASET_DIR = os.path.join(PATS_DATA_ROOT, f'Datasets/{SPEAKER_NAME}')
PATS_SPEAKER_DATA_DIR = os.path.join(PATS_DATA_ROOT, f'Processed/{SPEAKER_NAME}/data')
PATS_SPEAKER_INTERVAL_DIR = os.path.join(PATS_SPEAKER_DATA_DIR, 'processed', SPEAKER_NAME)

# Sys Path Sources
PROJECT_FECNET = '/Users/staveshemesh/Projects/shstav2/FECNet'
PROJECT_TOKEN_VOKEN = '/Users/staveshemesh/Projects/shstav2/token_voken'

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