import logging

from src.common.data_loader import load_valid_intervals
from src.common.constants import DF_INTERVALS_NOAH
from src.components.tokens._2_bert_tokens import BertTokens


logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)


def run():
    df_intervals = load_valid_intervals(DF_INTERVALS_NOAH)
    logger.info(f'Extract tokens from {df_intervals.shape} intervals..')
    interval_ids = df_intervals['interval_id'].tolist()
    for interval_id in interval_ids:
        BertTokens(interval_id).save_df_tokens()


if __name__ == '__main__':
    run()
