import os
import logging
import shutil
from src.common.debug import one_percent_chance
from src.common.constants import VOKENS_VOCAB_NOAH_V1_DIR, FRAME_EXTENSION
from src.common.file_utils import listdir_nohidden
from src.common.path_resolvers import get_video_id, \
    resolve_interval_all_faces_dir, resolve_frame_face_path


SPEAKER_PREFIX = 'n'  # Noah

interval_ids = ['cmu0000033573']

for interval_id in interval_ids:
    video_id = get_video_id(interval_id)
    # /home/stav/Data/PATS_DATA/Videos/oliver/Nn_Zln_4pA8/104792/FacesAll
    dir_faces_all = resolve_interval_all_faces_dir(interval_id)
    # ['/home/stav/Data/PATS_DATA/Videos/oliver/Nn_Zln_4pA8/104792/FacesAll/00213',
    #  '/home/stav/Data/PATS_DATA/Videos/oliver/Nn_Zln_4pA8/104792/FacesAll/00214']
    frame_faces_dirs = sorted(listdir_nohidden(dir_faces_all))
    # [213, 214]
    frame_ids = [int(all_faces_of_single_frame_dir.split("/")[-1]) for all_faces_of_single_frame_dir in
                 frame_faces_dirs]
    for frame_id in frame_ids:
        target_voken_filename = f'{SPEAKER_PREFIX}_{video_id}_{interval_id}_{frame_id}.{FRAME_EXTENSION}'
        target_voken_path     = os.path.join(VOKENS_VOCAB_NOAH_V1_DIR, target_voken_filename)
        # [Faces] Videos/oliver/0Rnq1NpHdmw/101462/Faces/00012.jpg
        frame_face_path       = resolve_frame_face_path(interval_id, frame_id)
        shutil.copyfile(frame_face_path, target_voken_path)
        if one_percent_chance():
            logging.info(f'Face Detection {interval_id} frame #{frame_id} {frame_face_path} → {target_voken_path}.')

