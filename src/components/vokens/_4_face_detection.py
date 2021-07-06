import os
import logging
import shutil

from tqdm import tqdm
from PIL import Image, ImageDraw
from facenet_pytorch import extract_face

from src.common.constants import FRAME_EXTENSION, FACE_IMAGE_SIZE
from src.common.path_resolvers import resolve_interval_frames_dir, resolve_detected_face_path, \
    resolve_interval_frame_path, resolve_annot_faces_path
from src.common.file_utils import create_empty_file
from src.vision.models.mtcnn import get_mtcnn_model


logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)


DEVICE_ID = '1'
mtcnn = get_mtcnn_model(DEVICE_ID)

def interval_extract_faces(interval_id):
    frames_dir = resolve_interval_frames_dir(interval_id)
    frames = sorted(os.listdir(frames_dir))
    logger.info(f'Extract faces for {len(frames)} frames interval {interval_id}..')
    for frame_filename in tqdm(frames):
        if frame_filename.endswith(f".{FRAME_EXTENSION}"):
            frame_id = int(frame_filename.split(".")[0])
            single_frame_extract_faces(interval_id, frame_id)

def single_frame_extract_faces(interval_id, frame_id):
    image, detection_result = _detect_faces(interval_id, frame_id)
    _save_faces(image, detection_result, interval_id, frame_id)
    # _copy_first_face(interval_id, frame_id)

def _detect_faces(interval_id, frame_id):
    frame_path = resolve_interval_frame_path(interval_id, frame_id)
    image = Image.open(frame_path)
    detection_result = mtcnn.detect(image, landmarks=True)
    return image, detection_result

def _save_faces(image, detection_result, interval_id, frame_id):
    # Save empty result
    annotated_faces_path = resolve_annot_faces_path(interval_id, frame_id, create=True)
    if detection_result[0] is None:
        frame_path = resolve_interval_frame_path(interval_id, frame_id)
        shutil.copyfile(frame_path, annotated_faces_path)
        face_0_path = resolve_detected_face_path(interval_id, frame_id, 0, create=True)
        create_empty_file(face_0_path)
        logger.error(f'🛑 Could not extract faces from {interval_id} frame {frame_id} {annotated_faces_path}.')
        return

    # Save actual face
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    boxes, probs, points = detection_result
    for i, (box, point, prob) in enumerate(zip(boxes, points, probs)):
        draw.rectangle(box.tolist(), width=10)
        for p in point:
            draw.rectangle((p - 10).tolist() + (p + 10).tolist(), width=5)
        detected_face_path = resolve_detected_face_path(interval_id, frame_id, i, create=True)
        extract_face(image, box, image_size=FACE_IMAGE_SIZE, margin=70, save_path=detected_face_path)
    img_draw.save(annotated_faces_path)

# def _copy_first_face(interval_id, frame_id):
#     # [FacesAll] Videos/oliver/0Rnq1NpHdmw/101462/FacesAll/00012/face_0.jpg
#     face_id = 0
#     face_0_path = resolve_detected_face_path(interval_id, frame_id, face_id)
#     # Videos/oliver/0Rnq1NpHdmw/101462/Faces/00012.jpg
#     frame_face_path = resolve_frame_face_path(interval_id, frame_id, create=True)
#     shutil.copyfile(face_0_path, frame_face_path)
#     if one_percent_chance():
#         logger.info(f'Face Detection {interval_id} frame {frame_id} {face_0_path} → {frame_face_path}.')
