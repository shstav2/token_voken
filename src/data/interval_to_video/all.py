from src.data.interval_to_video.oliver import INTERVAL_TO_VIDEO_OLIVER
from src.data.interval_to_video.noah import INTERVAL_TO_VIDEO_NOAH
from src.data.interval_to_video.seth import INTERVAL_TO_VIDEO_SETH


INTERVAL_TO_VIDEO = \
    {
        **INTERVAL_TO_VIDEO_OLIVER,
        **INTERVAL_TO_VIDEO_NOAH,
        **INTERVAL_TO_VIDEO_SETH
    }


def video_id_to_speaker(video_id):
    if video_id in INTERVAL_TO_VIDEO_OLIVER.values():
        return 'oliver'
    if video_id in INTERVAL_TO_VIDEO_NOAH.values():
        return 'noah'
    return 'seth'
