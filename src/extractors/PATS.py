import os
import logging
import pandas as pd

logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)

PATS = '/Users/staveshemesh/Projects/PATS_DATA/Processed/oliver/data/processed/oliver'

# video_id = 'aw6RsUhw1Q8'
interval_id = '216509'


def extract_interval_text(interval_id, debug=False):
    interval_text_path = os.path.join(PATS, f'{interval_id}.h5')
    if debug:
        logging.info('resolve_interval_text_path: ', interval_text_path)
    df_token_frames_interval = pd.read_hdf(interval_text_path)
    df_token_frames_interval['start_frame'] = df_token_frames_interval['start_frame'].astype(int)
    df_token_frames_interval['end_frame'] = df_token_frames_interval['end_frame'].astype(int)
    df_token_frames_interval['frames_count'] = df_token_frames_interval['end_frame'] - df_token_frames_interval['start_frame']
    return df_token_frames_interval


df = extract_interval_text(interval_id, debug=False)
