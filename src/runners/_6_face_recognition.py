import logging
from tqdm import tqdm

from src.common.data_loader import load_valid_intervals
from src.components._6_face_recognition import detect_face_recognition_errors, _copy_recognized_face

logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)


# def detect_intervals_recognition_errors(df_intervals):
#     logging.info('-------- Faces ➜ Relevant Face -----------')
#     interval_ids = df_intervals['interval_id'].tolist()
#     # Find recognition errors
intervals_no_errors2 = []
interval_to_recognized_faces2 = {}
for interval_id in tqdm(interval_ids2):
    frame_to_recognized_face, _ = detect_face_recognition_errors(interval_id)
    if len(frame_to_recognized_face) == 0:
        intervals_no_errors2.append(interval_id)
    else:
        interval_to_recognized_faces2[interval_id] = frame_to_recognized_face
return intervals_no_errors, interval_to_recognized_faces


def fix_interval(interval_id, frames):
    face_id = 1
    print(f'{interval_id}  # {len(frames)} fixes')
    for frame_id in frames:
        _copy_recognized_face(interval_id, frame_id, face_id)

# def fix_interval(interval_id, frame_to_face):
#     for frame_id, face_id in frame_to_face.items():
#         _copy_recognized_face(interval_id, frame_id, face_id)

# def run():
#     df_intervals = load_valid_intervals()
#     find_and_fix_face_recognition_errors(df_intervals)
#
#
INTERVALS_CORRECT_RECOGNITION = [
    '101462',
    '101463',
    '102744', # V checked manually
    '102749',
    '102825', # V checked manually
    '102826',
    '103850',
    '104200',
    '104201',
    '104698',
    '101053',
    '101057',
    '101814',
    '103294',
    '103295',
    '104005',
    '102063'
]

INTERVALS_CORRECT_RECOGNITION_AFTER_FIXES = [
    '101834',  # 1  fix
    '101843',  # 5  fixes
    '102184',  # 14 fixes
    '102187',  # 35 fixes
    '102750',  # 38 fixes
    '104075',  # 1  fix
    '104204',  # 31 fixes
    '104695',  # 5  fixes
    '100984',  # 59 fixes
    '100987',  # 52 fixes
    '100991',  # 55 fixes
    '101816',  # 2  fixes
    '101821',  # 23 fixes
]

# if __name__ == '__main__':
#     run()
