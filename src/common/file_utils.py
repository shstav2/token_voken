import os
import glob
from pathlib import Path
import logging

from src.common.debug import one_percent_chance
from src.common.display_utils import bool_to_symbol, ARR_R, SVE, DEL


logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_total_size(path):
    path_object = Path(path)
    if path_object.is_file():
        return os.path.getsize(path)
    root_listdir = path_object.glob('**/*')
    return sum(f.stat().st_size for f in root_listdir if f.is_file())


def get_number_of_directories(path):
    dir_count = 0
    for root, dirs, files in os.walk(path):
        dir_count += len(dirs)
    return dir_count


def human_readable_size(size_bytes, decimal_places=2):
    size = size_bytes
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']:
        if size < 1024.0 or unit == 'PiB':
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"


def exists_and_has_content(file_or_directory_path, size_threshold_bytes, debug=False):
    path_exists = os.path.exists(file_or_directory_path)
    path_size_bytes = get_total_size(file_or_directory_path) if path_exists else -1
    path_suffient_size = size_threshold_bytes < path_size_bytes
    path_exists_and_has_content = path_exists and path_suffient_size
    if one_percent_chance() or debug:
        symbol = bool_to_symbol(path_exists_and_has_content)
        logger.info(f'\t[Status] {symbol} {file_or_directory_path} (size: {human_readable_size(path_size_bytes)})')
    return path_exists_and_has_content


def is_empty_file(path):
    return os.path.getsize(path) == 0


def listdir_nohidden(path):
    return glob.glob(os.path.join(path, '*'))


def create_empty_file(path):
    print('TOUCH ', path)
    Path(path).touch()


def save_csv(df, path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    df.to_csv(path, index=False, header=True)
    if one_percent_chance():
        logger.info(f'{SVE} Saving {df.shape[0]:,} rows  {ARR_R}  {path}'\
                    f'\n\t{df.head(n=2)}')


def delete_file(path):
    path_exists = os.path.exists(path)
    path_size_bytes = get_total_size(path) if path_exists else -1
    logger.info(f'{DEL} Delete file {path} ({path_size_bytes}).')
