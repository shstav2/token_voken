from src.common.constants import DF_INTERVALS_NOAH
from src.common.data_loader import load_valid_intervals
from src.components.vokens._6_face_recognition import detect_face_recognition_errors, _copy_recognized_face


def detect_errors_and_fix(interval_id):
    frame_to_face, _ = detect_face_recognition_errors(interval_id)
    fix_interval(interval_id, frame_to_face)


def fix_interval(interval_id, frame_to_face):
    for frame_id, face_id in frame_to_face.items():
        _copy_recognized_face(interval_id, frame_id, face_id)


df_intervals = load_valid_intervals(DF_INTERVALS_NOAH)
interval_ids = df_intervals['interval_id'].tolist()
for interval_id in interval_ids:
    detect_errors_and_fix(interval_id)


# def fix_interval(interval_id, frames):
#     face_id = 0
#     print(f'{interval_id}  # {len(frames)} fixes')
#     for frame_id in frames:
#         _copy_recognized_face(interval_id, frame_id, face_id)