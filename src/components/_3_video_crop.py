import os

from src.common.commands import run_command
from src.common.constants import PATS_SPEAKER_VIZ_DIR


def save_interval(input_fn, start, end, output_fn):
    """
        -strict: Specify how strictly to follow the standards
        -y: Overwrite output files without asking
    """
    command = 'ffmpeg -i "%s" -ss %s -to %s -strict -2 "%s" -y' % (input_fn, start, end, output_fn)
    run_command(command)


def crop_tool(interval_row):
    speaker, video_id, interval_id, start, end = interval_row[
        ['speaker', 'video_id', 'interval_id', 'start_time_string', 'end_time_string']]

    video_dir = os.path.join(PATS_SPEAKER_VIZ_DIR, video_id)
    video_path = os.path.join(video_dir, f'{video_id}.mp4')

    # TODO: use get_interval_video_path
    interval_dir = os.path.join(video_dir, str(interval_id))
    interval_path = os.path.join(interval_dir, f'{interval_id}.mp4')

    if not os.path.exists(os.path.dirname(interval_path)):
        os.makedirs(os.path.dirname(interval_path))
    save_interval(video_path, start, end, interval_path)
