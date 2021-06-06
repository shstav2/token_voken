import pandas as pd
from src.common.constants import INTERVALS_PATH

import logging
logger = logging.getLogger(__name__)


def load_intervals():
    logger.info(f'Loading intervals from {INTERVALS_PATH}..')
    df_intervals = pd.read_csv(INTERVALS_PATH, dtype={'interval_id': object})
    logger.info(f'Fetch {df_intervals.shape} shape dataframe.')
    return df_intervals
