import os
import logging
import pandas as pd

from src.common.debug import one_percent_chance
from src.common.display_utils import bool_to_symbol
from src.common.path_resolvers import resolve_interval_frames_dir, resolve_interval_faces_dir, \
    resolve_interval_dir, resolve_interval_facial_embeddings_dir

logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)


def validate_dirs_have_the_same_filenames(interval_id, dir1, dir2):
    if not (os.path.exists(dir1) and os.path.exists(dir2)):
        return False

    dir1_filenames = [filename.split('.')[0] for filename in os.listdir(dir1)]
    dir2_filenames = [filename.split('.')[0] for filename in os.listdir(dir2)]
    have_same_filenames = set(dir1_filenames) == set(dir2_filenames)
    if one_percent_chance():
        symbol = bool_to_symbol(have_same_filenames)
        interval_data_dir = resolve_interval_dir(interval_id)
        first_file, last_file = min(dir1_filenames), max(dir2_filenames)
        dir1_name = os.path.basename(dir1)
        dir2_name = os.path.basename(dir2)
        logger.info(f'\t[Validation] {symbol} {dir1_name} ↔️  {dir2_name} ({first_file} - {last_file}). {interval_data_dir}')
    return have_same_filenames


def validate_faces_count_eq_frames_count(interval_id):
    # Videos/oliver/0Rnq1NpHdmw/101462/Frames
    interval_frames_dir = resolve_interval_frames_dir(interval_id)
    # Videos/oliver/0Rnq1NpHdmw/101462/Faces
    interval_faces_dir = resolve_interval_faces_dir(interval_id)
    return validate_dirs_have_the_same_filenames(interval_id, interval_frames_dir, interval_faces_dir)


def validate_embeddings_count_eq_frames_count(interval_id):
    # Videos/oliver/0Rnq1NpHdmw/101462/Frames
    interval_frames_dir = resolve_interval_frames_dir(interval_id)
    # Videos/oliver/0Rnq1NpHdmw/101462/FECNet
    interval_faces_dir = resolve_interval_facial_embeddings_dir(interval_id)
    return validate_dirs_have_the_same_filenames(interval_id, interval_frames_dir, interval_faces_dir)

