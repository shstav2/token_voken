import logging

from src.common.data_loader import load_valid_intervals
from src.components.tokens._1_raw_word_frames import save_raw_text

logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)


def run():
    df_intervals = load_valid_intervals()
    logger.info(f'Extract words from, {df_intervals.shape} intervals..')
    for interval_id in df_intervals['interval_id'].tolist():
        save_raw_text(interval_id)


if __name__ == '__main__':
    run()
