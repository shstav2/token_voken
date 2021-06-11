import os
import torch
import logging

from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN, extract_face

from src.common.constants import FRAME_EXTENSION
from src.common.path_resolvers import resolve_interval_frames_dir

logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(select_largest=False, thresholds=[0.9, 0.9, 0.9], device=device)
    logger.info(f'Init ({device}) MTCNN {mtcnn}..')


def create_face_images(df_intervals, interval_ids):
    # for interval_id in tqdm(interval_ids):
    for interval_id in interval_ids:
        frames_dir = resolve_interval_frames_dir(df_intervals, interval_id)
        frames = sorted(os.listdir(frames_dir))
        for frame_filename in frames:
            if frame_filename.endswith(f".{FRAME_EXTENSION}"):
                frame_fullpath = os.path.join(frames_dir, frame_filename)
                frame_id = int(frame_filename.split(".")[0])
                save_faces(interval_id, frame_id, frame_fullpath)


def save_faces(mtcnn, interval_id, frame_id, frame_path):
    image = Image.open(frame_path)
    boxes, probs, points = mtcnn.detect(image, landmarks=True)
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    for i, (box, point, prob) in enumerate(zip(boxes, points, probs)):
        draw.rectangle(box.tolist(), width=10)
        for p in point:
            draw.rectangle((p - 10).tolist() + (p + 10).tolist(), width=5)
        detected_face_path = resolve_detected_face_path(df_intervals_valid, interval_id, frame_id,
                                                        i, create=True)
        extract_face(image, box, image_size=224, margin=70, save_path=detected_face_path)

    annotated_faces_path = resolve_annot_faces_path(df_intervals_valid, interval_id, frame_id)
    img_draw.save(annotated_faces_path)



