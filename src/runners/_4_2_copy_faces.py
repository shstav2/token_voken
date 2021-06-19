import os
import shutil
import logging

from src.common.data_loader import load_valid_intervals
from src.common.debug import one_percent_chance
from src.common.path_resolvers import resolve_interval_all_faces_dir, resolve_detected_face_path, \
    resolve_frame_face_path


logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)


def copy_face_frame(df_intervals):
    face_id = 0
    for interval_id in df_intervals['interval_id']:
        interval_all_faces_dir = resolve_interval_all_faces_dir(interval_id)
        frames = sorted(os.listdir(interval_all_faces_dir))
        for frame in frames:
            frame_id = int(frame)
            # [FacesAll] Videos/oliver/0Rnq1NpHdmw/101462/FacesAll/00012/face_0.jpg
            face_0_path = resolve_detected_face_path(interval_id, frame_id, face_id)
            # Videos/oliver/0Rnq1NpHdmw/101462/Faces/00012.jpg
            frame_face_path = resolve_frame_face_path(interval_id, frame_id, create=True)
            shutil.copyfile(face_0_path, frame_face_path)
            if one_percent_chance():
                logger.info(f'Copy {face_0_path} → {frame_face_path}.')

def run():
    df_intervals = load_valid_intervals()
    copy_face_frame(df_intervals)


if __name__ == '__main__':
    run()
