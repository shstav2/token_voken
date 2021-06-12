import os
import logging
from tqdm import tqdm

import torch
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN, extract_face

from src.common.constants import FRAME_EXTENSION
from src.common.debug import one_percent_chance
from src.common.path_resolvers import resolve_interval_frames_dir, resolve_detected_face_path, \
    resolve_interval_frame_path, resolve_annot_faces_path

logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(select_largest=False, thresholds=[0.9, 0.9, 0.9], device=device)
    logger.info(f'Init ({device}) MTCNN {mtcnn}..')
    return mtcnn


mtcnn = get_model()


def create_face_images(df_intervals, interval_id):
    frames_dir = resolve_interval_frames_dir(df_intervals, interval_id)
    frames = sorted(os.listdir(frames_dir))
    logger.info(f'Extract faces for {len(frames)} frames inteval {interval_id}..')
    for frame_filename in tqdm(frames):
        if frame_filename.endswith(f".{FRAME_EXTENSION}"):
            frame_id = int(frame_filename.split(".")[0])
            save_faces(df_intervals, interval_id, frame_id)


def save_faces(df_intervals, interval_id, frame_id):
    frame_path = resolve_interval_frame_path(df_intervals, interval_id, frame_id)
    image = Image.open(frame_path)
    boxes, probs, points = mtcnn.detect(image, landmarks=True)
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    for i, (box, point, prob) in enumerate(zip(boxes, points, probs)):
        draw.rectangle(box.tolist(), width=10)
        for p in point:
            draw.rectangle((p - 10).tolist() + (p + 10).tolist(), width=5)
        detected_face_path = resolve_detected_face_path(df_intervals, interval_id, frame_id, i, create=True)
        extract_face(image, box, image_size=224, margin=70, save_path=detected_face_path)

    annotated_faces_path = resolve_annot_faces_path(df_intervals, interval_id, frame_id)
    img_draw.save(annotated_faces_path)
    if one_percent_chance():
        logger.info(f'Extract faces {annotated_faces_path}..')

