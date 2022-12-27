import operator
import os
import math
import numpy as np


# LFW functions taken from David Sandberg's FaceNet implementation
def distance(embeddings1, embeddings2, distance_metric='euc'):
    return np.linalg.norm(embeddings1-embeddings2)


# TODO: Generalize
def get_distances(e_base, frame_resnet_dir, face_filename):
    face_id = face_filename.split('.')[0].split('_')[1]
    # [ResNet] Videos/oliver/0Rnq1NpHdmw/101462/ResNet/00012/face_0.npy
    face_resnet_embedding_path = os.path.join(frame_resnet_dir, face_filename)
    print(face_resnet_embedding_path) #TODO: tmp
    e = np.load(face_resnet_embedding_path)
    dist_euclidian = distance(e, e_base, distance_metric='euc')
    dist_cosine = distance(e, e_base, distance_metric='cos')
    # if debug:
    #     logger.info(f'\tFace={face_id} | Euclidian: {dist_euclidian:.4f}, Cosine: {dist_cosine:.4f}')
    return (face_id, dist_euclidian, dist_cosine)


# TODO: Generalize
def face_and_delta_by_distance(face_id_idx, distances, dist_idx):
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

    most_similar_face_id = min_distance_record[face_id_idx]
    distance_delta = max_distance_record[dist_idx] - min_distance_record[dist_idx]

    return most_similar_face_id, distance_delta
