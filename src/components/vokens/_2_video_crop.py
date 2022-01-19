import os

from src.common.commands import run_command
from src.common.path_resolvers import resolve_video_file_path, resolve_interval_video_path


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

    video_path    = resolve_video_file_path(video_id)
    # Videos/oliver/0Rnq1NpHdmw/101462/101462.mp4
    interval_path = resolve_interval_video_path(interval_id)

    if not os.path.exists(os.path.dirname(interval_path)):
        os.makedirs(os.path.dirname(interval_path))
    save_interval(video_path, start, end, interval_path)
