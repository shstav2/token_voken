import os
import logging
import pandas as pd

from src.common.debug import one_percent_chance
from src.common.display_utils import bool_to_symbol
from src.common.path_resolvers import resolve_interval_frames_dir, resolve_interval_faces_dir, \
    resolve_interval_dir

logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)



def validate_faces_count_eq_frames_count(interval_id, debug=False):
    # Videos/oliver/0Rnq1NpHdmw/101462/Frames
    interval_frames_dir = resolve_interval_frames_dir(interval_id)
    # Videos/oliver/0Rnq1NpHdmw/101462/Faces
    inerval_faces_dir = resolve_interval_faces_dir(interval_id)

    frames_filenames = os.listdir(interval_frames_dir)
    faces_filenames = os.listdir(inerval_faces_dir)
    detected_faces_for_all_frames = set(frames_filenames) == set(faces_filenames)

    if one_percent_chance() or debug:
        symbol = bool_to_symbol(detected_faces_for_all_frames)
        interval_data_dir = resolve_interval_dir(interval_id)
        first_file, last_file = min(frames_filenames), max(frames_filenames)
        logger.info(f'\t[Validation] {symbol} Frames ↔️  Faces ({first_file} - {last_file}). {interval_data_dir}')

    return detected_faces_for_all_frames
