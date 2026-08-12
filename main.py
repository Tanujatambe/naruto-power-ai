import cv2
import mediapipe as mp
import numpy as np
import math
import os


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "hand_landmarker.task"
)

NARUTO_VIDEO = os.path.join(
    BASE_DIR,
    "assets",
    "naruto.mp4"
)

SASUKE_VIDEO = os.path.join(
    BASE_DIR,
    "assets",
    "sasuke.mp4"
)


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "hand_landmarker.task not found!"
    )

if not os.path.exists(NARUTO_VIDEO):
    raise FileNotFoundError(
        "assets/naruto.mp4 not found!"
    )

if not os.path.exists(SASUKE_VIDEO):
    raise FileNotFoundError(
        "assets/sasuke.mp4 not found!"
    )


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = (
    mp.tasks.vision.HandLandmarker
)

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

RunningMode = (
    mp.tasks.vision.RunningMode
)


options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=RunningMode.VIDEO,

    num_hands=2,

    min_hand_detection_confidence=0.65,

    min_hand_presence_confidence=0.65,

    min_tracking_confidence=0.65
)


# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open webcam."
    )


# ============================================================
# EFFECT VIDEOS
# ============================================================

naruto = cv2.VideoCapture(
    NARUTO_VIDEO
)

sasuke = cv2.VideoCapture(
    SASUKE_VIDEO
)


if not naruto.isOpened():
    raise RuntimeError(
        "Could not open Naruto video."
    )


if not sasuke.isOpened():
    raise RuntimeError(
        "Could not open Sasuke video."
    )


# ============================================================
# VARIABLES
# ============================================================

pwr = [0.0, 0.0]

was_open = [
    False,
    False
]

timestamp = 0


# ============================================================
# HAND CONNECTIONS
# ============================================================

HAND_CONNECTIONS = [

    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    (0, 17)
]


# ============================================================
# CHECK OPEN HAND
# ============================================================

def check_open(points):

    wrist = points[0]

    tips = [
        8,
        12,
        16,
        20
    ]

    pips = [
        6,
        10,
        14,
        18
    ]

    count = 0

    for tip_index, pip_index in zip(
        tips,
        pips
    ):

        tip = points[tip_index]

        pip = points[pip_index]

        tip_distance = math.hypot(
            tip.x - wrist.x,
            tip.y - wrist.y
        )

        pip_distance = math.hypot(
            pip.x - wrist.x,
            pip.y - wrist.y
        )

        if tip_distance > pip_distance:
            count += 1

    return count >= 3


# ============================================================
# DRAW HAND
# ============================================================

def draw_hand(
    frame,
    points
):

    h, w = frame.shape[:2]

    pixel_points = []

    for point in points:

        x = int(point.x * w)

        y = int(point.y * h)

        pixel_points.append(
            (x, y)
        )

    # Cyan skeleton

    for start, end in HAND_CONNECTIONS:

        cv2.line(

            frame,

            pixel_points[start],

            pixel_points[end],

            (255, 212, 0),

            3
        )

    # White landmarks

    for x, y in pixel_points:

        cv2.circle(

            frame,

            (x, y),

            3,

            (255, 255, 255),

            -1
        )


# ============================================================
# GET NEXT EFFECT FRAME
# ============================================================

def get_video_frame(video):

    success, frame = video.read()

    if not success:

        video.set(
            cv2.CAP_PROP_POS_FRAMES,
            0
        )

        success, frame = video.read()

    if success:

        return frame

    return None


# ============================================================
# SCREEN BLEND EFFECT
# ============================================================

def overlay_video(
    background,
    effect,
    x,
    y,
    opacity,
    size=(500, 350)
):

    if effect is None:
        return background


    # --------------------------------------------------------
    # Resize effect
    # --------------------------------------------------------

    effect = cv2.resize(
        effect,
        size
    )


    eh, ew = effect.shape[:2]


    # --------------------------------------------------------
    # Position
    # --------------------------------------------------------

    x = int(
        x - ew / 2
    )

    y = int(
        y - eh / 2
    )


    h, w = background.shape[:2]


    # --------------------------------------------------------
    # Clip to screen
    # --------------------------------------------------------

    x1 = max(
        0,
        x
    )

    y1 = max(
        0,
        y
    )

    x2 = min(
        w,
        x + ew
    )

    y2 = min(
        h,
        y + eh
    )


    if x1 >= x2 or y1 >= y2:

        return background


    # --------------------------------------------------------
    # Effect crop
    # --------------------------------------------------------

    ex1 = x1 - x

    ey1 = y1 - y

    ex2 = ex1 + (
        x2 - x1
    )

    ey2 = ey1 + (
        y2 - y1
    )


    effect_crop = effect[
        ey1:ey2,
        ex1:ex2
    ]


    roi = background[
        y1:y2,
        x1:x2
    ]


    # --------------------------------------------------------
    # SCREEN BLENDING
    #
    # Black background becomes invisible.
    # Bright effect remains visible.
    # --------------------------------------------------------

    background_float = (
        roi.astype(
            np.float32
        )
    )

    effect_float = (
        effect_crop.astype(
            np.float32
        )
    )


    screen = 255 - (

        (255 - background_float)
        *
        (255 - effect_float)
        / 255
    )


    # --------------------------------------------------------
    # Opacity
    # --------------------------------------------------------

    result = (

        background_float
        *
        (1 - opacity)

        +

        screen
        *
        opacity
    )


    result = np.clip(
        result,
        0,
        255
    ).astype(
        np.uint8
    )


    background[
        y1:y2,
        x1:x2
    ] = result


    return background


# ============================================================
# MAIN PROGRAM
# ============================================================

print()
print(
    "===================================="
)

print(
    "        NARUTO POWER"
)

print(
    "===================================="
)

print(
    "LEFT HAND  -> NARUTO"
)

print(
    "RIGHT HAND -> SASUKE"
)

print(
    "Press Q to quit."
)

print()


# ============================================================
# MEDIAPIPE HAND LANDMARKER
# ============================================================

with HandLandmarker.create_from_options(
    options
) as landmarker:


    while True:


        # ----------------------------------------------------
        # CAMERA
        # ----------------------------------------------------

        success, frame = cap.read()

        if not success:

            print(
                "Could not read webcam."
            )

            break


        # Mirror webcam

        frame = cv2.flip(
            frame,
            1
        )


        height, width = (
            frame.shape[:2]
        )


        # ----------------------------------------------------
        # BGR -> RGB
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(

            frame,

            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------------------------
        # MediaPipe Image
        # ----------------------------------------------------

        mp_image = mp.Image(

            image_format=(
                mp.ImageFormat.SRGB
            ),

            data=rgb_frame
        )


        # ----------------------------------------------------
        # HAND DETECTION
        # ----------------------------------------------------

        result = (
            landmarker.detect_for_video(

                mp_image,

                timestamp
            )
        )


        timestamp += 1


        # ----------------------------------------------------
        # FLAGS
        # ----------------------------------------------------

        fL = False

        fR = False


        # ====================================================
        # PROCESS DETECTED HANDS
        # ====================================================

        for i, points in enumerate(
            result.hand_landmarks
        ):


            # ------------------------------------------------
            # LEFT / RIGHT
            # ------------------------------------------------

            label = (
                result
                .handedness[i][0]
                .category_name
            )


            is_right = (
                label == "Right"
            )


            idx = (
                1 if is_right
                else 0
            )


            # ------------------------------------------------
            # DRAW SKELETON
            # ------------------------------------------------

            draw_hand(
                frame,
                points
            )


            # ------------------------------------------------
            # CHECK OPEN HAND
            # ------------------------------------------------

            opened = check_open(
                points
            )


            # ------------------------------------------------
            # POWER INCREASE / DECREASE
            # ------------------------------------------------

            if opened:

                pwr[idx] += 0.05

            else:

                pwr[idx] -= 0.15


            pwr[idx] = max(
                0,
                min(
                    1,
                    pwr[idx]
                )
            )


            # ------------------------------------------------
            # RESTART EFFECT WHEN HAND OPENS
            # ------------------------------------------------

            if (
                opened
                and
                not was_open[idx]
            ):

                if is_right:

                    sasuke.set(
                        cv2.CAP_PROP_POS_FRAMES,
                        0
                    )

                else:

                    naruto.set(
                        cv2.CAP_PROP_POS_FRAMES,
                        0
                    )


            was_open[idx] = opened


            # ------------------------------------------------
            # LANDMARKS
            # ------------------------------------------------

            wrist = points[0]

            knuckle = points[9]


            wrist_x = (
                wrist.x * width
            )

            wrist_y = (
                wrist.y * height
            )


            knuckle_x = (
                knuckle.x * width
            )

            knuckle_y = (
                knuckle.y * height
            )


            # =================================================
            # RIGHT HAND -> SASUKE
            # =================================================

            if (
                is_right
                and
                pwr[idx] > 0.01
            ):

                fR = True


                x = (

                    wrist_x
                    +
                    knuckle_x

                ) / 2


                y = (

                    wrist_y
                    +
                    knuckle_y

                ) / 2


                effect = (
                    get_video_frame(
                        sasuke
                    )
                )


                frame = overlay_video(

                    frame,

                    effect,

                    x,

                    y,

                    pwr[idx],

                    size=(
                        600,
                        450
                    )
                )


            # =================================================
            # LEFT HAND -> NARUTO
            # =================================================

            elif (
                not is_right
                and
                pwr[idx] > 0.01
            ):

                fL = True


                dx = (
                    knuckle_x
                    -
                    wrist_x
                )

                dy = (
                    knuckle_y
                    -
                    wrist_y
                )


                x = (
                    knuckle_x
                    +
                    dx * 0.8
                )


                y = (
                    knuckle_y
                    +
                    dy * 0.8
                )


                effect = (
                    get_video_frame(
                        naruto
                    )
                )


                frame = overlay_video(

                    frame,

                    effect,

                    x,

                    y - 100,

                    pwr[idx],

                    size=(
                        500,
                        350
                    )
                )


        # ====================================================
        # FADE OUT
        # ====================================================

        if not fL:

            pwr[0] -= 0.15

            pwr[0] = max(
                0,
                pwr[0]
            )

            was_open[0] = False


        if not fR:

            pwr[1] -= 0.15

            pwr[1] = max(
                0,
                pwr[1]
            )

            was_open[1] = False


        # ====================================================
        # SHOW
        # ====================================================

        cv2.imshow(
            "Naruto Power",
            frame
        )


        # ====================================================
        # QUIT
        # ====================================================

        key = (
            cv2.waitKey(1)
            &
            0xFF
        )


        if key == ord("q"):

            break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

naruto.release()

sasuke.release()

cv2.destroyAllWindows()

print()
print(
    "Naruto Power closed."
)