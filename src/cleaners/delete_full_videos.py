from src.common.data_loader import load_intervals, load_videos
from src.common.constants import DF_INTERVALS_NOAH
from src.common.file_utils import delete_file
from src.common.path_resolvers import resolve_video_file_path


df_intervals = load_intervals(DF_INTERVALS_NOAH)
df_videos = load_videos(df_intervals)
video_ids = df_videos['video_id']

for video_id in video_ids:
    path = resolve_video_file_path(video_id)
    if os.path.exists(path):
        # os.remove(path)
        delete_file(path)


