import os
import logging
import numpy as np
from tqdm import tqdm
import torch
from models.FECNet import FECNet
from src.common.path_resolvers import resolve_interval_faces_dir, resolve_interval_facial_embedding_path
from src.common.file_utils import is_empty_file, create_empty_file, listdir_nohidden
from src.common.debug import one_percent_chance
from src.vision.data_loader import get_tensor_image


device = 'cuda:0'
BATCH_SIZE = 64


logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)


class VisnFECNetModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        fecnet = FECNet(device=device, pretrained=True) # Setup Backbone
        for param in fecnet.parameters():
            param.requires_grad = False
        self.backbone = fecnet
    def forward(self, img):
        x = self.backbone(img)
        x = x.detach()
        # x = x / x.norm(2, dim=-1, keepdim=True)
        return x

def get_model():
    visn_model = VisnFECNetModel(device).eval()
    assert visn_model.backbone.training == False
    assert visn_model.training == False
    assert next(visn_model.parameters()).is_cuda
    return visn_model

VISN_MODEL = get_model()

def extract_and_save_interval_facial_embeddings(interval_id):
    faces_dir = resolve_interval_faces_dir(interval_id)
    logger.info(f'Extract embeddings for interval {interval_id} from {faces_dir}..')
    interval_face_filenames = sorted(listdir_nohidden(faces_dir))
    batch_tensor_imgs, batch_frame_ids = [], []
    for i, img_path in enumerate(tqdm(interval_face_filenames)):
        face_filename = os.path.basename(img_path)
        frame_id = int(face_filename.split(".")[0])
        if is_empty_file(img_path):
            embedding_path = resolve_interval_facial_embedding_path(interval_id, frame_id, create=True)
            create_empty_file(embedding_path)
            logger.error(f'🛑 Could not extract embeddings for {interval_id} frame {img_path}.\n'\
                    f'Creating empty file {embedding_path}.')
            continue
        img_tensor = get_tensor_image(img_path)
        batch_tensor_imgs.append(img_tensor)
        batch_frame_ids.append(frame_id)
        if len(batch_tensor_imgs) == BATCH_SIZE:
            batch_img_keys = _model_forward_on_img_batch(batch_tensor_imgs, device)
            _save_embeddings(interval_id, batch_img_keys, batch_frame_ids)
            batch_tensor_imgs, batch_frame_ids = [], []
    if 0 < len(batch_tensor_imgs):
        batch_img_keys = _model_forward_on_img_batch(batch_tensor_imgs, device)
        _save_embeddings(interval_id, batch_img_keys, batch_frame_ids)


def _model_forward_on_img_batch(batch_tensor_imgs, device):
    visn_input = torch.stack(batch_tensor_imgs).to(device)
    with torch.no_grad():
        visn_output = VISN_MODEL(visn_input)
    if one_percent_chance():
        logger.info(f'Call FECNet model with {visn_input.shape} output: {visn_output.shape}')
    return visn_output.detach().cpu().numpy()

def _save_embeddings(interval_id, embeddings, frame_ids):
    for embedding, frame_id in zip(embeddings, frame_ids):
        embedding_path = resolve_interval_facial_embedding_path(interval_id, frame_id, create=True)
        if one_percent_chance():
            logger.info(f'Save embedding {embedding} ➡️  {embedding_path}')
        np.save(embedding_path, embedding)
