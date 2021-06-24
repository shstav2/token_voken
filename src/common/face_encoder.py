import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms#, fixed_image_standardization
from torch.utils.data import DataLoader
from collections import defaultdict


from src.img_utils.reader import get_tensor_image, ImageFolderWithPaths
from src.common.constants import OLIVER_FACE_PATH

# https://github.com/timesler/facenet-pytorch/blob/master/examples/lfw_evaluate.ipynb

device = 'cuda:0'

def get_resnet_model():
    resnet = InceptionResnetV1(
        classify=False,
        pretrained='vggface2'
    ).to(device)
    resnet.eval()
    return resnet

trans = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    # fixed_image_standardization
])

# data_dir = '/home/stav/Data/PATS_DATA/Videos/oliver/vU8dCYocuyI/216641/FacesAll/'
data_dir = '/home/stav/Data/PATS_DATA/Videos/oliver/vU8dCYocuyI/216641_delete/FacesAll/FacesAll'


def is_face_path(path):
    return 'face_' in path

def get_data_loader(data_dir):
    dataset = ImageFolderWithPaths(data_dir, transform=trans, is_valid_file=is_face_path)
    dataloader = DataLoader(dataset)
    return dataloader

dataloader = get_data_loader(data_dir)
embeddings = []

np_oliver_face_image = get_tensor_image(OLIVER_FACE_PATH)

resnet = get_resnet_model()
oliver_face_embedding = resnet(np_oliver_face_image.unsqueeze(0).to(device))
np_oliver_face_embedding = oliver_face_embedding.detach().to('cpu').numpy().reshape(-1)

dataloader = get_data_loader(data_dir)
embeddings = []
i=0
frame_id_to_embeddings = defaultdict(list)
with torch.no_grad():
    for inputs, labels, paths in dataloader:
        assert len(paths) == len(labels) == 1
        path = paths[0]
        frame_id = labels[0].item()
        print('inputs shape: ', inputs.shape)
        inputs = inputs.to(device)
        img_embedding = resnet(inputs)
        np_img_embedding = img_embedding.to('cpu').numpy().reshape(-1)
        embeddings.extend(np_img_embedding)
        print(f'i={i}, paths={paths} ({len(paths)}), labels={labels}\n\tframe_id={type(frame_id)} {frame_id}')
        frame_id_to_embeddings[frame_id].append((path, np_img_embedding))
        i += 1
        # if i == 10:
        #     break

import math
# # LFW functions taken from David Sandberg's FaceNet implementation
# def distance(embeddings1, embeddings2, distance_metric=0):
#     if distance_metric==0:
#         # Euclidian distance
#         diff = np.subtract(embeddings1, embeddings2)
#         print('diff: ', diff.shape)
#         sqr = np.square(diff)
#         print('sqr: ', sqr.shape)
#         print('np.sum(sqr)')
#         dist = np.sum(sqr)
#         # dist = np.sum(np.square(diff),1)
#     elif distance_metric==1:
#         # Distance based on cosine similarity
#         dot = np.sum(np.multiply(embeddings1, embeddings2))
#         norm = np.linalg.norm(embeddings1) * np.linalg.norm(embeddings2)
#         similarity = dot / norm
#         dist = np.arccos(similarity) / math.pi
#     else:
#         raise 'Undefined distance metric %d' % distance_metric
#     return dist

# # LFW functions taken from David Sandberg's FaceNet implementation
# def distance(embeddings1, embeddings2, distance_metric=0):
#     if distance_metric==0:
#         # Euclidian distance
#         diff = np.subtract(embeddings1, embeddings2)
#         dist = np.sum(np.square(diff),1)
#     elif distance_metric==1:
#         # Distance based on cosine similarity
#         dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
#         norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
#         similarity = dot / norm
#         dist = np.arccos(similarity) / math.pi
#     else:
#         raise 'Undefined distance metric %d' % distance_metric
#
#     return dist

# LFW functions taken from David Sandberg's FaceNet implementation
def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric==0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff))
    elif distance_metric==1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2))
        norm = np.linalg.norm(embeddings1) * np.linalg.norm(embeddings2)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric
    return dist
#
# for frame_id, pairs in frame_id_to_embeddings.items():
#     print(f'Frame ID: {frame_id}')
#     dists = []
#     for pair in pairs:
#         face_path, face_resent_emb = pair
#         dist_norm = np.linalg.norm(face_resent_emb - np_oliver_face_embedding)
#         print(f'\tface_path={face_path}')
#         embeddings1 = face_resent_emb
#         embeddings2 = np_oliver_face_embedding
#         diff = np.subtract(embeddings1, embeddings2)
#         dist_eucl = np.sum(np.square(diff))
#         # dist_eucl = distance(embeddings1, embeddings2, distance_metric=0)
#         # dist_eucl = distance(embeddings1, np_oliver_face_embedding, distance_metric=0)
#         dist_cos = distance(face_resent_emb, np_oliver_face_embedding, distance_metric=1)
#         print(f'\tdist_norm={dist_norm:.4f}, dist_eucl={dist_eucl:.4f}, dist_cos={dist_eucl:.4f}')
#         dists.append(dist_norm)
#     # if np.argmin(dists) != 0:
#     #     print(f'\t🛑{frame_id}')


dist = defaultdict(dict)
for frame_id, pairs in frame_id_to_embeddings.items():
    print(f'Frame ID: {frame_id}')
    for pair1 in pairs:
        path1, e1 = pair1
        for pair2 in pairs:
            path2, e2 = pair2
            # dist_norm = np.linalg.norm(e1 - e2)
            dist_eucl = distance(e1, e2, distance_metric=0)
            dist_cos = distance(e1, e2, distance_metric=1)
            # if path1 < path2:
            print(f'\tdist_cos={dist_cos:.4f}, dist_eucl={dist_eucl:.4f} path1={path1[-15:]}, path2={path2[-15:]}')
            path_dist = {path2: (dist_cos, dist_eucl)}
            dist[path1].update(path_dist)
            if path1 == path2:
                if dist_cos > 0 or dist_eucl > 0:
                    print('!!!!!')
                # assert dist_eucl == dist_cos ==  0





