import os
import logging
import operator
import shutil

import numpy as np
from tqdm import tqdm

from src.common.debug import one_percent_chance
from src.common.display_utils import V, WRN, ERR, IMP, SML
from src.common.file_utils import listdir_nohidden
from src.common.path_resolvers import resolve_interval_resnet_embeddings_dir, \
    resolve_single_frame_resnet_faces_dir, resolve_detected_face_path, resolve_frame_face_path, \
    resolve_interval_local_faces_dir, resolve_interval_faces_dir
from src.vision.distance import distance


logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)


OLIVER_FACE_RESNET_EMBEDDING_PATH = '/home/stav/Data/Sample/oliver/face.npy'
e_base = np.load(OLIVER_FACE_RESNET_EMBEDDING_PATH)

TAG = '[FaceRecognition]'

FACE_ID_IDX = 0
DIST_COSINE_IDX = 1
DIST_EUCLIDEAN_IDX = 2


DEFAULT_FACE_ID = '0'


def detect_face_recognition_errors(interval_id):
    """
    Returns:
        - Mapping of frame id to the id of the most relevant face, only if it is not face_0:
            {34: '1',
             36: '1',
             37: '1',
             42: '1',
             43: '1'}
        - Interval's ResNet embedding directory path:
            '/home/stav/Data/PATS_DATA/Videos/oliver/vU8dCYocuyI/216895/ResNet'
    """
    # [ResNet] Videos/oliver/0Rnq1NpHdmw/101462/ResNet
    interval_resnet_dir = resolve_interval_resnet_embeddings_dir(interval_id)
    frames = sorted(listdir_nohidden(interval_resnet_dir))
    frame_to_recognized_face = {}
    # For each frame
    for i, frame in enumerate(frames):
        # Initialize values for face recognition (if needed)
        requires_face_recognition, face_filenames, frame_resnet_dir, frame_id, debug = \
            requires_face_recognition_and_metadta(interval_id, frame)
        if not requires_face_recognition:
            continue
        # For each face, get distances from base image
        distances = []
        for face_filename in face_filenames:
            distances.append(get_distances(frame_resnet_dir, face_filename, debug))
        # Save recognized face id if it is not the default face (face_0)
        recognized_face_id = frame_face_recognition_and_some_logs(interval_id, frame_id, distances, frame_resnet_dir, debug)
        if recognized_face_id != DEFAULT_FACE_ID:
            frame_to_recognized_face[frame_id] = recognized_face_id

    log_recognition_summary(interval_id, frame_to_recognized_face)

    return frame_to_recognized_face, interval_resnet_dir


def requires_face_recognition_and_metadta(interval_id, frame_resnet_dir):
    # [ResNet] Videos/oliver/0Rnq1NpHdmw/101462/ResNet/00012
    frame_id = int(os.path.basename(frame_resnet_dir))
    frame_resnet_dir = resolve_single_frame_resnet_faces_dir(interval_id, frame_id)
    face_filenames = sorted(os.listdir(frame_resnet_dir))
    requires_face_recognition = 1 < len(face_filenames)

    debug = one_percent_chance()
    # if debug: logger.info(f'{TAG} Interval={interval_id}, Frame={frame_id}')

    return requires_face_recognition, face_filenames, frame_resnet_dir, frame_id, debug


def get_distances(frame_resnet_dir, face_filename, debug=False):
    face_id = face_filename.split('.')[0].split('_')[1]
    # [ResNet] Videos/oliver/0Rnq1NpHdmw/101462/ResNet/00012/face_0.npy
    face_resnet_embedding_path = os.path.join(frame_resnet_dir, face_filename)
    e = np.load(face_resnet_embedding_path)
    dist_euclidian = distance(e, e_base, distance_metric='euc')
    dist_cosine = distance(e, e_base, distance_metric='cos')
    # if debug:
    #     logger.info(f'\tFace={face_id} | Euclidian: {dist_euclidian:.4f}, Cosine: {dist_cosine:.4f}')
    return (face_id, dist_euclidian, dist_cosine)


def frame_face_recognition_and_some_logs(interval_id, frame_id, distances, frame_resnet_dir, debug):
    face_id_of_min_euclidan, delta_euclidean = face_and_delta_by_distance(distances, DIST_EUCLIDEAN_IDX)
    face_id_of_min_cosine, delta_cosine = face_and_delta_by_distance(distances, DIST_COSINE_IDX)

    inconsistent_recognition = face_id_of_min_euclidan != face_id_of_min_cosine
    if inconsistent_recognition:
        logger.error(f'{TAG} {ERR} min(Cosine) ≠ min(Euclidan) Interval={interval_id}, Frame={frame_id}\n{distances}')
        return None

    most_similar_face_id = face_id_of_min_cosine
    recognized_non_default_face = most_similar_face_id != DEFAULT_FACE_ID
    distance_delta_log_msg =  f'Diff: Euclidian={delta_euclidean:.4f}, Cosine={delta_cosine:.4f} ({frame_resnet_dir}).'

    # if recognized_non_default_face:
    #     logger.warning(f'\t{WRN} Recognized face id {most_similar_face_id}. {distance_delta_log_msg}')
    # if debug: logger.info(f'\tmin(Euclidan) = min(Cosine) ➜ Face {most_similar_face_id}. {distance_delta_log_msg}')

    return most_similar_face_id


def face_and_delta_by_distance(distances, dist_idx):
    """
    Params:
        - distances:
            List of tuples of the form:
                [<face_id, dist1, dist2.. >,
                 <face_id, dist1, dist2.. >]
        - dist_idx:
            Which distance metric to use:
                dist_idx=1 --> use dist1 for calculations.
    Returns:
        - The id of the most similar face in respect to the specificed distance metric:
            '0'
        - The maximal diff between two faces in respect to the specified distance metric:
            0.0913 = (cosine_distance(e(face_0) - e(face_1))
    """
    min_distance_record = min(distances, key=operator.itemgetter(dist_idx))
    max_distance_record = max(distances, key=operator.itemgetter(dist_idx))

    most_similar_face_id = min_distance_record[FACE_ID_IDX]
    distance_delta = max_distance_record[dist_idx] - min_distance_record[dist_idx]

    return most_similar_face_id, distance_delta


def _copy_recognized_face(interval_id, frame_id, face_id):
    # [FacesAll] Videos/oliver/0Rnq1NpHdmw/101462/FacesAll/00012/face_0.jpg
    face_path = resolve_detected_face_path(interval_id, frame_id, face_id)
    # Videos/oliver/0Rnq1NpHdmw/101462/Faces/00012.jpg
    frame_face_path = resolve_frame_face_path(interval_id, frame_id, create=True)
    shutil.copyfile(face_path, frame_face_path)
    logger.info(f'{IMP} {TAG} {interval_id} frame {frame_id} {face_path} → {frame_face_path}.')


def log_recognition_summary(interval_id, frame_to_recognized_face):
    faces_dir = resolve_interval_faces_dir(interval_id)
    local_faces_dir = resolve_interval_local_faces_dir(interval_id)
    if len(frame_to_recognized_face) == 0:
        logger.info(f'{TAG} {V} Interval {interval_id} no frame was mislabeld' \
                    f'\n\t{local_faces_dir}' \
                    f'\n\t{faces_dir}')
    else:
        logger.warning(f'{TAG} {WRN}  Interval {interval_id} has {len(frame_to_recognized_face)} frames mislabeld.'\
                       f'\n\tfix_interval("{interval_id}", {list(frame_to_recognized_face.keys())})'\
                       f'\n\t{local_faces_dir}' \
                       f'\n\t{faces_dir}')
