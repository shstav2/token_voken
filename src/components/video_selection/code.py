import time
from tqdm import tqdm
import pandas as pd

tqdm.pandas()

from src.components.video_selection.cols import COL_DURATION
from src.data.interval_to_video.noah import INTERVAL_TO_VIDEO_NOAH
from src.common.path_resolvers import resolve_interval_raw_text_path
from src.common.constants import FRAME_RATE, SPEAKER_NAME


LIST_BULLET = '  ◘ '


def print_df_info(df):
    print(f'{LIST_BULLET}Videos: #{df["video_link"].nunique():,}')
    print(f'{LIST_BULLET}Intervals: #{df["interval_id"].nunique():,}')
    total_duration = df[COL_DURATION].sum()
    total_duration_string = time.strftime('%H hours, %M minutues, %S seconds', time.gmtime(total_duration))
    print(f'{LIST_BULLET}Total Duration: {total_duration_string} ({int(total_duration):,} seconds)')
    all_youtube = df['video_link'].str.contains('youtube').all()
    print(f'{LIST_BULLET}All are Youtube videos: {all_youtube}')

