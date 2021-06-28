import os
import logging
import operator
import numpy as np
from tqdm import tqdm

from src.common.debug import one_percent_chance
from src.common.display_utils import V, WRN, ERR, IMP
from src.common.path_resolvers import resolve_interval_resnet_embeddings_dir, \
    resolve_single_frame_resnet_faces_dir
from src.vision.distance import distance


logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)


OLIVER_FACE_RESNET_EMBEDDING_PATH = '/home/stav/Data/Sample/oliver/face.npy'
e_base = np.load(OLIVER_FACE_RESNET_EMBEDDING_PATH)

TAG = '[SelectFace]'

def select_interval_relevant_faces(interval_id):
    # [ResNet] Videos/oliver/0Rnq1NpHdmw/101462/ResNet
    interval_resnet_dir = resolve_interval_resnet_embeddings_dir(interval_id)
    frames = sorted(os.listdir(interval_resnet_dir))
    mislabeled_frames = {}
    for i, frame in enumerate(tqdm(frames)):
        frame_id = int(frame.split(".")[0])
        # [ResNet] Videos/oliver/0Rnq1NpHdmw/101462/ResNet/00012
        frame_resnet_dir = resolve_single_frame_resnet_faces_dir(interval_id, frame_id)
        face_filenames = sorted(os.listdir(frame_resnet_dir))
        distances = []
        debug = one_percent_chance()
        if debug:
            logger.info(f'{TAG} Interval={interval_id}, Frame={frame_id}')
        for face_filename in face_filenames:
            # [ResNet] Videos/oliver/0Rnq1NpHdmw/101462/ResNet/00012/face_0.npy
            face_id = face_filename.split('.')[0].split('_')[1]
            face_resnet_embedding_path = os.path.join(frame_resnet_dir, face_filename)
            e = np.load(face_resnet_embedding_path)
            dist_euclidian = distance(e, e_base, distance_metric=0)
            dist_cosine = distance(e, e_base, distance_metric=1)
            if debug:
                logger.info(f'\tFace={face_id} | Euclidian: {dist_euclidian:.4f}, Cosine: {dist_cosine:.4f}')
            distances.append((face_id, dist_cosine, dist_euclidian))
        face_id_of_min_euclidan = min(distances, key=operator.itemgetter(1))[0]
        face_id_of_min_cosine = min(distances, key=operator.itemgetter(2))[0]
        delta_euclidian = max(distances, key=operator.itemgetter(1))[1] - min(distances, key=operator.itemgetter(1))[1]
        delta_cosine = max(distances, key=operator.itemgetter(2))[2] - min(distances, key=operator.itemgetter(2))[2]
        if face_id_of_min_euclidan != face_id_of_min_cosine:
            logger.error(f'{TAG} {ERR} min(Cosine) ≠ min(Euclidan) {interval_id}#{frame_id}#{face_id}'\
                f'\n{distances}')
        else:
            most_similar_face_id = face_id_of_min_cosine
            if most_similar_face_id != '0':
                logger.warning(f'\t{WRN}  Most similar face is not 0, but {most_similar_face_id}.'\
                        f' Diff: Euclidian={delta_euclidian:.4f}, Cosine={delta_cosine:.4f} ({frame_resnet_dir}).')
                mislabeled_frames[frame_id] = most_similar_face_id
            if debug:
                diff_str = ''
                if 1 < len(distances):
                    diff_str = f' Diff: Euclidian={delta_euclidian:.4f}, Cosine={delta_cosine:.4f} ({frame_resnet_dir})'
                logger.info(f'\tmin(Euclidan) = min(Cosine) ➜ Face {most_similar_face_id}{diff_str}')
    if len(mislabeled_frames) == 0:
        logging.info(f'{TAG} {V} Interval {interval_id} no frame was mislabeld ({interval_resnet_dir}).')
    else:
        logging.warning(f'{TAG} {WRN}  Interval {interval_id} has {len(mislabeled_frames)}' \
                        f' frames mislabeld: {list(mislabeled_frames.keys())} ({interval_resnet_dir}).')
    return mislabeled_frames, interval_resnet_dir


def _copy_first_face(interval_id, frame_id):
    # [FacesAll] Videos/oliver/0Rnq1NpHdmw/101462/FacesAll/00012/face_0.jpg
    face_id = 0
    face_0_path = resolve_detected_face_path(interval_id, frame_id, face_id)
    # Videos/oliver/0Rnq1NpHdmw/101462/Faces/00012.jpg
    frame_face_path = resolve_frame_face_path(interval_id, frame_id, create=True)
    shutil.copyfile(face_0_path, frame_face_path)
    if one_percent_chance():
        logger.info(f'Face Detection {interval_id} frame {frame_id} {face_0_path} → {frame_face_path}.')
