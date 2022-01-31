import logging

from src.common.data_loader import load_valid_intervals
from src.common.constants import DF_INTERVALS_NOAH, DF_INTERVALS_OLIVER
from src.components.tokens._2_bert_tokens import BertTokens
from src.monitoring.status import status_text_tokens_csv


logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)


def run():
df_intervals = load_valid_intervals(DF_INTERVALS_OLIVER)
df_intervals['status_text_tokens_csv'] = df_intervals['interval_id'].apply(status_text_tokens_csv)
logger.info(df_intervals['status_text_tokens_csv'].value_counts())
df_intevals_pendings = df_intervals[~df_intervals['status_text_tokens_csv']]
logger.info(f'Extract tokens from {df_intevals_pendings.shape} intervals..')
    interval_ids = df_intevals_pendings['interval_id'].tolist()
errors = []
for interval_id in interval_ids:
    try:
        BertTokens(interval_id).save_df_tokens()
    except TypeError:
        errors.append(interval_id)


if __name__ == '__main__':
    run()
