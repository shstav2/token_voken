import os

from common.path_resolvers import resolve_video_dir_path, resolve_video_file_path
from src.common.commands import run_command


YOUTUBE_DOWNLOAD_COMMAND = 'youtube-dl -f bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio -o {output_path} {link}'

def youtube_downloader(video_id):
    link = f'https://www.youtube.com/watch?v={video_id}'
    output_dir = resolve_video_dir_path(video_id)
    output_path = resolve_video_file_path(video_id)
    if not(os.path.exists(os.path.dirname(output_dir))):
        os.makedirs(os.path.dirname(output_dir))
    command = YOUTUBE_DOWNLOAD_COMMAND.format(link=link, output_path=output_path)
    run_command(command)
