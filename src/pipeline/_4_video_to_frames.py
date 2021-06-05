import os
import shutil
from src.common.commands import run_command
from src.common.constants import FRAME_RATE, FRAME_EXTENSION


START_FRAME = 0
import logging
import sys
logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
                     level=logging.INFO, stream=sys.stdout)


def video_to_frames_and_delete(interval_video_path, interval_frames_dir, override=False, delete=False):
    video_has_frames = os.path.exists(interval_frames_dir) and os.path.getsize(interval_frames_dir) > 0
    if video_has_frames:
        if override:
            frames_to_delete = [frame for frame in os.listdir(interval_frames_dir) if frame.endswith(f'.{FRAME_EXTENSION}')]
            for frame_filename in frames_to_delete:
                frame_fullpath = os.path.join(interval_frames_dir, frame_filename)
                os.remove(frame_fullpath)
        else:
            raise RuntimeError(f'{interval_frames_dir} - not empty')
    video_to_frames(interval_video_path, interval_frames_dir)
    if delete:
        delete_video(interval_video_path)


def video_to_frames(interval_video_path, interval_frames_dir):
    if not os.path.exists(interval_frames_dir):
        os.makedirs(interval_frames_dir)
    command = f'ffmpeg -i "{interval_video_path}" -start_number {START_FRAME} -r {FRAME_RATE} '\
        f'-q:v 2 -qmin 2 -qmax 2 "{interval_frames_dir}/$filename%05d.{FRAME_EXTENSION}"'
    run_command(command)


def delete_video(interval_video_path):
    if os.path.exists(interval_video_path):
        os.remove(interval_video_path)
