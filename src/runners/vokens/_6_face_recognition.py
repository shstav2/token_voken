from src.components.vokens._6_face_recognition import detect_face_recognition_errors, _copy_recognized_face


def fix_interval(interval_id, frame_to_face):
    for frame_id, face_id in frame_to_face.items():
        _copy_recognized_face(interval_id, frame_id, face_id)


def fix(interval_id):
    frame_to_face, _ = detect_face_recognition_errors(interval_id)
    fix_interval(interval_id, frame_to_face)

# _copy_recognized_face(interval_id, 111, 0)

def fix_interval(interval_id, frames):
    face_id = 0
    print(f'{interval_id}  # {len(frames)} fixes')
    for frame_id in frames:
        _copy_recognized_face(interval_id, frame_id, face_id)

fix_interval("214714", [0, 1, 3, 4, 5, 8, 15, 17, 18, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 102, 103, 104, 105, 106, 107, 108, 109, 110, 114, 115, 116, 117, 118, 125, 126, 127, 128, 129, 130, 131, 132, 133, 135, 142, 143, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 156, 157, 158, 159, 160, 161, 163, 164, 165, 167, 168, 169, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 221, 223, 224, 225, 226, 227, 228, 229, 230, 233, 235, 236, 237, 238, 239, 240, 241, 244, 245, 246, 247])

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
    # ---
    '215431',  # 215 fixes
    '104413',  # 146 fixes
    '214934',  # lots of fixes
]

# if __name__ == '__main__':
#     run()
