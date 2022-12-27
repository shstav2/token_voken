from src.data.interval_to_video.oliver import INTERVAL_TO_VIDEO_OLIVER
from src.data.interval_to_video.noah import INTERVAL_TO_VIDEO_NOAH


INTERVAL_TO_VIDEO = \
    {
        **INTERVAL_TO_VIDEO_OLIVER,
        **INTERVAL_TO_VIDEO_NOAH
    }


def video_id_to_speaker(video_id):
    if video_id in INTERVAL_TO_VIDEO_OLIVER.values():
        return 'oliver'
    return 'seth'
