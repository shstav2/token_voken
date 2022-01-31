import os

from src.common.commands import run_command
from src.common.path_resolvers import resolve_interval_all_faces_dir
from src.common.file_utils import listdir_nohidden, delete_file
from src.data.interval_to_video.all import INTERVAL_TO_VIDEO


REDUCTED_QUALITY = 10
REDUCE_IMAGE_QUALITY_COMMAND = 'convert -quality {quality}% {faces_all}/annotated_faces.jpg {faces_all}/annotated_faces_{quality}.jpg'


SPEAKER_NAME = 'noah'
interval_ids = list(INTERVAL_TO_VIDEO[SPEAKER_NAME].keys())


errors = []
for interval_id in interval_ids:
    # /home/stav/Data/PATS_DATA/Videos/oliver/Nn_Zln_4pA8/104792/FacesAll
    dir_faces_all = resolve_interval_all_faces_dir(interval_id)
    # ['/home/stav/Data/PATS_DATA/Videos/oliver/Nn_Zln_4pA8/104792/FacesAll/00213',
    #  '/home/stav/Data/PATS_DATA/Videos/oliver/Nn_Zln_4pA8/104792/FacesAll/00214']
    frame_faces_dirs = sorted(listdir_nohidden(dir_faces_all))
    for all_faces_of_single_frame_dir in frame_faces_dirs:
        try:
            command = REDUCE_IMAGE_QUALITY_COMMAND.format(faces_all=all_faces_of_single_frame_dir, quality=REDUCTED_QUALITY)
            run_command(command)
            original_annotated_faces = os.path.join(all_faces_of_single_frame_dir, 'annotated_faces.jpg')
            delete_file(original_annotated_faces)
        except Exception as e:
            errors.append(interval_id)
            print(f'ERROR: {interval_id}, frame {all_faces_of_single_frame_dir[-10:]}, {e}')
