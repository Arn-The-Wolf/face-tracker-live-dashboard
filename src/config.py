from pathlib import Path

# Paths
DB_NPZ_PATH = Path("data/db/face_db.npz")
HISTORY_DIR = Path("data/history")
ARCFACE_MODEL_PATH = "models/embedder_arcface.onnx"

# Camera
CAMERA_INDEX = 0

# Detection & Locking
HAAR_MIN_SIZE = (70, 70)
DEFAULT_DISTANCE_THRESHOLD = 0.75
LOCK_RELEASE_FRAMES = 30
LOCK_MOVEMENT_THRESHOLD_PX = 50
LOCK_ACTION_COOLDOWN_FRAMES = 15

# Mediapipe Landmarks Indices
# EAR (Eye Aspect Ratio) 6 points
LOCK_EAR_LEFT_INDICES = [33, 160, 158, 133, 153, 144]
LOCK_EAR_RIGHT_INDICES = [362, 385, 387, 263, 373, 380]
LOCK_EAR_BLINK_THRESHOLD = 0.25

LOCK_MOUTH_LEFT_INDEX = 61
LOCK_MOUTH_RIGHT_INDEX = 291
LOCK_SMILE_MOUTH_RATIO = 1.15

# MQTT settings - match your ESP8266
MQTT_BROKER   = "157.173.101.159"
MQTT_PORT     = 1883
MQTT_TOPIC    = "vision/user398/movement"
MQTT_CLIENT_ID = "python_face_tracker_user398"

# Angle publishing
MIN_ANGLE = 0
MAX_ANGLE = 180
CENTER_ANGLE = 90
ANGLE_HYSTERESIS = 3     # only publish if change >= this (degrees)
last_published_angle = CENTER_ANGLE

# Head Pose (Yaw)
LOCK_POSE_NOSE_INDEX = 1
LOCK_POSE_LEFT_EYE_INDEX = 33
LOCK_POSE_RIGHT_EYE_INDEX = 263
LOCK_YAW_LOOK_LEFT_THRESHOLD = 0.6  # Ratio d(nose, left) / d(nose, right)
LOCK_YAW_LOOK_RIGHT_THRESHOLD = 1.6 # Ratio d(nose, left) / d(nose, right)

def ensure_dirs():
    DB_NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

