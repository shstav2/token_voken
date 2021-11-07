import os

from src.common.commands import run_command
from src.common.path_resolvers import resolve_interval_all_faces_dir
from src.common.file_utils import listdir_nohidden, delete_file
from src.data.interval_to_video.oliver import INTERVAL_TO_VIDEO_OLIVER


REDUCTED_QUALITY = 10
REDUCE_IMAGE_QUALITY_COMMAND = 'convert -quality {quality}% {faces_all}/annotated_faces.jpg {faces_all}/annotated_faces_{quality}.jpg'


interval_ids = ['104792']
# interval_id = interval_ids[0]

oliver_interval_ids = list(INTERVAL_TO_VIDEO_OLIVER.keys())

interval_ids = ['105339']
for interval_id in oliver_interval_ids[100:1000]:
    # /home/stav/Data/PATS_DATA/Videos/oliver/Nn_Zln_4pA8/104792/FacesAll
    dir_faces_all = resolve_interval_all_faces_dir(interval_id)
    # ['/home/stav/Data/PATS_DATA/Videos/oliver/Nn_Zln_4pA8/104792/FacesAll/00213',
    #  '/home/stav/Data/PATS_DATA/Videos/oliver/Nn_Zln_4pA8/104792/FacesAll/00214']
    frame_faces_dirs = sorted(listdir_nohidden(dir_faces_all))
    for all_faces_of_single_frame_dir in frame_faces_dirs:
        command = REDUCE_IMAGE_QUALITY_COMMAND.format(faces_all=all_faces_of_single_frame_dir, quality=REDUCTED_QUALITY)
        run_command(command)
        original_annotated_faces = os.path.join(all_faces_of_single_frame_dir, 'annotated_faces.jpg')
        delete_file(original_annotated_faces)
