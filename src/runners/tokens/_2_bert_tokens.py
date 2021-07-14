import logging

from src.common.data_loader import load_valid_intervals
from src.components.tokens._2_bert_tokens import BertTokens


logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)


def run():
    df_intervals = load_valid_intervals()
    logger.info(f'Extract tokens from {df_intervals.shape} intervals..')
    for interval_id in df_intervals['interval_id'].tolist():
        BertTokens(interval_id).save_df_tokens()


if __name__ == '__main__':
    run()
