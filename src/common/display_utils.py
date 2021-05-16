import time
import pandas as pd
from IPython.display import Markdown, display

from common.path_resolvers import resolve_224_voken_path

LIST_BULLET = '  ◘ '
TAB = '&nbsp;&nbsp;&nbsp;&nbsp;'

CAPTION_STYLE = {
    'selector': 'caption',
    'props': [
        ('color', 'blue'),
        ('font-size', '16px')
    ]
}


def printmd(string):
    display(Markdown(string))


def df_with_caption(df, title):
    return df.style.set_caption(title).set_table_styles([CAPTION_STYLE])


def style_value_counts(mask, title):
    df_value_counts = (pd.Series(mask.reshape(-1))).value_counts().to_frame().head()
    df_style = df_with_caption(df_value_counts, title)
    display(df_style)


def display_voken(interval_id, frame_id):
    voken_path = resolve_224_voken_path(interval_id=interval_id, frame_id=frame_id)
    print(voken_path)
    display(IPython.display.Image(voken_path))


def display_df_with_caption(df, title):
    return df.style.set_caption(title).set_table_styles([CAPTION_STYLE])


def display_value_counts(series, title):
    df_value_counts = series.value_counts().to_frame().head()
    df_style = display_df_with_caption(df_value_counts, title)
    display(df_style)


def display_df_info(df_intervals):
    print(f'{LIST_BULLET}Videos: #{df_intervals["video_link"].nunique():,}')
    print(f'{LIST_BULLET}Intervals: #{df_intervals["interval_id"].nunique():,}')
    total_duration = df_intervals["duration"].sum()
    total_duration_string = time.strftime('%H hours, %M minutues, %S seconds', time.gmtime(total_duration))
    print(f'{LIST_BULLET}Total Duration: {total_duration_string} ({int(total_duration):,} seconds)')
    all_youtube = df_intervals['video_link'].str.contains('youtube').all()
    print(f'{LIST_BULLET}All are Youtube videos: {all_youtube}')
