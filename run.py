print("start")
video_ids = [      'lg9Tu79F4qE', '4FBqT80fVKY', 'ASP0Tad3EMg', 'p_vTyApRF-w','BalEqNEEv2M', '3ezfGuZmZlM', 'urMsr55b25Y', 'N0lA28AsMpo', 'mewW_pRSgEc', 'iaoD6pfFuTo', 'ONxHwmQ9dYo', 'x0xpxMI6mIU', '7a6irhXRudo', 'qKzmJhSFHzc', '-d2JEhzcrOc', '05U0AYfnWgI', 'QGQ5Y_6YTgE', 'RKpjH-O8a6w', 'qWFydl-nFsw', 'Ry96G7qtXjM', 'PCNlRgBTufQ', '0Nyl1yUbVHM', 'H3mX88_tx3Q','dL2fGSMafRM', 'o1ZfBFyrskY', 'RolwE1lXOq8', 's2HpnptGcrU','jxrogLn39Do', '5sBhANSz--k', '0FQfThSU11E', 'Nw-BMNnr1Ko','pVAD70eSWA8', 'm6qubM54PtE', '9TFMg_SSU6k', 'CQyD31t_8Ao',]


import os

from src.common.path_resolvers import resolve_video_dir_path, resolve_video_file_path
from src.common.commands import run_command


YOUTUBE_DOWNLOAD_COMMAND = 'youtube-dl -f bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio -o {output_path} {link} --no-check-certificate'

def youtube_downloader(video_id):
    link = f'https://www.youtube.com/watch?v={video_id}'
    output_dir = resolve_video_dir_path(video_id)
    output_path = resolve_video_file_path(video_id)
    if not(os.path.exists(os.path.dirname(output_dir))):
        os.makedirs(os.path.dirname(output_dir))
    command = YOUTUBE_DOWNLOAD_COMMAND.format(link=link, output_path=output_path)
    run_command(command)

for video_id in video_ids:
    print("videooo")
    youtube_downloader(video_id)


print("end!!!!!!")
