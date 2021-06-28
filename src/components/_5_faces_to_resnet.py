import os
import logging
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm

from src.common.debug import one_percent_chance
from src.common.path_resolvers import resolve_interval_all_faces_dir, \
    resolve_single_frame_faces_dir, resolve_face_resnet_embedding_path
from src.vision.data_loader import get_data_loader
from src.vision.models.resnet import get_resnet_model


logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)


device = 'cuda:0'
resnet_model = get_resnet_model(device)


def create_resnet_embeddings(interval_id):
    # [FacesAll] Videos/oliver/0Rnq1NpHdmw/101462/FacesAll
    interval_all_faces_dir = resolve_interval_all_faces_dir(interval_id)
    frame_id_to_embeddings = get_interval_faces_embeddings(interval_all_faces_dir, device, resnet_model)
    for frame_id, face_to_embedding in frame_id_to_embeddings.items():
        for face_id, face_embedding in face_to_embedding.items():
            # [ResNet] Videos/oliver/0Rnq1NpHdmw/101462/ResNet/00012/face_0.npy
            face_resnet_embedding_path = resolve_face_resnet_embedding_path(interval_id, frame_id, face_id, create=True)
            np.save(face_resnet_embedding_path, face_embedding)
            if one_percent_chance():
                logger.info(f'[Face2ResNet] saved {interval_id}#{frame_id}#{face_id}: {face_embedding} ➡️  {face_resnet_embedding_path}.')


def get_interval_faces_embeddings(data_dir, device, resnet_model):
    dataloader = get_data_loader(data_dir)
    frame_id_to_embeddings = defaultdict(dict)
    i = 0
    with torch.no_grad():
        for inputs, labels, paths in dataloader:
            assert len(paths) == len(labels) == 1
            path = paths[0]
            frame_id = labels[0].item()
            face_id = os.path.basename(path).split('.')[0].split('_')[-1]
            inputs = inputs.to(device)
            img_embedding = resnet_model(inputs)
            np_img_embedding = img_embedding.to('cpu').numpy().reshape(-1)
            if one_percent_chance():
                logger.info(f'[Face2ResNet] DataLoader dir: {data_dir}. i={i}, path={path} ({len(paths)}), frame_id={frame_id}, face_id={face_id}')
            frame_id_to_embeddings[frame_id].update({face_id: np_img_embedding})
            i += 1
    return frame_id_to_embeddings
