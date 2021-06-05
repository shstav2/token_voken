import os

from src.common.commands import run_command
from src.common.constants import PATS_SPEAKER_VIZ_DIR


YOUTUBE_DOWNLOAD_COMMAND = 'youtube-dl -f bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio -o {output_path} {link}'

def youtube_downloader(video_id):
    link = f'https://www.youtube.com/watch?v={video_id}'
    output_dir = os.path.join(PATS_SPEAKER_VIZ_DIR, video_id)
    output_path = os.path.join(output_dir, video_id)
    if not(os.path.exists(os.path.dirname(output_dir))):
        os.makedirs(os.path.dirname(output_dir))
    command = YOUTUBE_DOWNLOAD_COMMAND.format(link=link, output_path=output_path)
    run_command(command)
