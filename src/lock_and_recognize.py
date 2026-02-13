import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from paho.mqtt import client as mqtt_client_module

try:
    import mediapipe as mp
except ImportError:
    mp = None

# Suppress MediaPipe / Abseil verbose logs
os.environ["GLOG_minloglevel"] = "2"          # 0=all, 1=INFO, 2=WARNING+, 3=ERROR+

from . import config   # assuming this is your config module

# ────────────────────────────────────────────────
# MQTT Configuration
# ────────────────────────────────────────────────
MQTT_BROKER   = "157.173.101.159"
MQTT_PORT     = 1883
MQTT_TOPIC    = "vision/banki_Senbonzakura_Kageyoshi/movement"
MQTT_CLIENT_ID = f"python_face_tracker_{int(time.time())}"

MIN_ANGLE       = 0
MAX_ANGLE       = 180
ANGLE_HYSTERESIS = 3          # only publish if change >= this degrees

# ────────────────────────────────────────────────
# Math & Geometry Helpers
# ────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    return float(np.dot(a, b))

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cosine_similarity(a, b)

def _clip_xyxy(x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(W - 1, round(x1)))
    y1 = max(0, min(H - 1, round(y1)))
    x2 = max(0, min(W - 1, round(x2)))
    y2 = max(0, min(H - 1, round(y2)))
    return (int(x1), int(y1), int(x2), int(y2)) if x1 <= x2 and y1 <= y2 else (int(x2), int(y2), int(x1), int(y1))

def _bbox_from_5pt(kps: np.ndarray, pad_x=0.55, pad_y_top=0.85, pad_y_bot=1.15) -> np.ndarray:
    k = kps.astype(np.float32)
    x_min, x_max = np.min(k[:, 0]), np.max(k[:, 0])
    y_min, y_max = np.min(k[:, 1]), np.max(k[:, 1])
    w = max(1.0, x_max - x_min)
    h = max(1.0, y_max - y_min)
    return np.array([
        x_min - pad_x * w,
        y_min - pad_y_top * h,
        x_max + pad_x * w,
        y_max + pad_y_bot * h
    ], dtype=np.float32)

def _kps_span_ok(kps: np.ndarray, min_eye_dist: float) -> bool:
    le, re, no, lm, rm = kps
    eye_dist = np.linalg.norm(re - le)
    if eye_dist < min_eye_dist:
        return False
    if not (lm[1] > no[1] and rm[1] > no[1]):
        return False
    return True

# ────────────────────────────────────────────────
# Face Alignment & Detection
# ────────────────────────────────────────────────

@dataclass
class FaceDet:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    kps: np.ndarray  # (5,2)

def _estimate_norm_5pt(kps_5x2: np.ndarray, out_size=(112, 112)) -> np.ndarray:
    dst = np.array([
        [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
        [41.5493, 92.3655], [70.7299, 92.2041]
    ], dtype=np.float32)
    scale = np.array(out_size) / 112.0
    if out_size != (112, 112):
        dst *= scale
    M, _ = cv2.estimateAffinePartial2D(kps_5x2.astype(np.float32), dst, method=cv2.LMEDS)
    if M is None:
        M = cv2.getAffineTransform(kps_5x2[:3].astype(np.float32), dst[:3].astype(np.float32))
    return M.astype(np.float32)

def align_face_5pt(frame_bgr: np.ndarray, kps_5x2: np.ndarray, out_size=(112, 112)) -> Tuple[np.ndarray, np.ndarray]:
    M = _estimate_norm_5pt(kps_5x2, out_size)
    aligned = cv2.warpAffine(frame_bgr, M, out_size, flags=cv2.INTER_LINEAR, borderValue=(0,0,0))
    return aligned, M

# ────────────────────────────────────────────────
# Face Detector (Haar + MediaPipe FaceMesh)
# ────────────────────────────────────────────────

class HaarFaceMesh5pt:
    def __init__(self, min_size=(70, 70)):
        if mp is None:
            raise RuntimeError("MediaPipe is required but not installed.")
        
        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(haar_path)
        self.min_size = min_size
        
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.idx_map = [33, 263, 1, 61, 291]  # left eye, right eye, nose, left mouth, right mouth

    def detect(self, frame, max_faces=5) -> List[FaceDet]:
        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE, minSize=self.min_size
        )
        if len(faces) == 0:
            return []

        # Sort by size (largest first)
        areas = faces[:, 2] * faces[:, 3]
        order = np.argsort(-areas)
        faces = faces[order][:max_faces]

        detections = []
        for x, y, w, h in faces:
            # Expand ROI slightly for better mesh detection
            mx, my = int(0.25 * w), int(0.35 * h)
            rx1 = max(0, x - mx)
            ry1 = max(0, y - my)
            rx2 = min(W, x + w + mx)
            ry2 = min(H, y + h + my)

            roi = frame[ry1:ry2, rx1:rx2]
            if roi.size == 0:
                continue

            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            results = self.mesh.process(roi_rgb)

            if not results.multi_face_landmarks:
                continue

            lm = results.multi_face_landmarks[0].landmark
            pts = []
            for idx in self.idx_map:
                p = lm[idx]
                pts.append([p.x * (rx2 - rx1) + rx1, p.y * (ry2 - ry1) + ry1])
            kps = np.array(pts, dtype=np.float32)

            # Swap if needed
            if kps[0, 0] > kps[1, 0]:
                kps[[0, 1]] = kps[[1, 0]]
            if kps[3, 0] > kps[4, 0]:
                kps[[3, 4]] = kps[[4, 3]]

            if not _kps_span_ok(kps, min_eye_dist=max(10.0, 0.18 * w)):
                continue

            bb = _bbox_from_5pt(kps)
            bx1, by1, bx2, by2 = _clip_xyxy(*bb, W, H)
            detections.append(FaceDet(bx1, by1, bx2, by2, 1.0, kps))

        return detections

# ────────────────────────────────────────────────
# ArcFace Embedder (ONNX)
# ────────────────────────────────────────────────

class ArcFaceEmbedderONNX:
    def __init__(self, model_path: Path, input_size=(112, 112)):
        self.sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        self.w, self.h = input_size

    def embed(self, img_bgr: np.ndarray) -> np.ndarray:
        if img_bgr.shape[:2] != (self.h, self.w):
            img_bgr = cv2.resize(img_bgr, (self.w, self.h))
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 128.0
        x = np.transpose(rgb, (2, 0, 1))[None].astype(np.float32)
        emb = self.sess.run([self.output_name], {self.input_name: x})[0]
        emb = emb.reshape(-1)
        return emb / (np.linalg.norm(emb) + 1e-12)

# ────────────────────────────────────────────────
# Action Detection Helpers
# ────────────────────────────────────────────────

def _ear_from_landmarks(landmarks, indices, W, H):
    pts = np.array([[lm.x * W, lm.y * H] for lm in [landmarks[i] for i in indices]])
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * h) if h > 1e-6 else 0.5

def _mouth_width(landmarks, i1, i2, W, H):
    l, r = landmarks[i1], landmarks[i2]
    return np.hypot((r.x - l.x) * W, (r.y - l.y) * H)

def detect_actions(frame, mesh, prev_cx, cx, baseline_mouth, frame_idx, last_actions):
    actions = []
    if mesh is None:
        return actions, None, None

    H, W = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = mesh.process(rgb)
    if not res.multi_face_landmarks:
        return actions, None, None

    lms = res.multi_face_landmarks[0].landmark

    cooldown = config.LOCK_ACTION_COOLDOWN_FRAMES

    # ── Horizontal movement ──
    if prev_cx is not None:
        dx = cx - prev_cx
        if dx <= -config.LOCK_MOVEMENT_THRESHOLD_PX:
            if frame_idx - last_actions.get("move_left", -999) > cooldown:
                actions.append(("face_moved_left", "Left"))
                last_actions["move_left"] = frame_idx
        elif dx >= config.LOCK_MOVEMENT_THRESHOLD_PX:
            if frame_idx - last_actions.get("move_right", -999) > cooldown:
                actions.append(("face_moved_right", "Right"))
                last_actions["move_right"] = frame_idx

    # ── Blink ──
    ear_l = _ear_from_landmarks(lms, config.LOCK_EAR_LEFT_INDICES, W, H)
    ear_r = _ear_from_landmarks(lms, config.LOCK_EAR_RIGHT_INDICES, W, H)
    ear = (ear_l + ear_r) / 2
    if ear < config.LOCK_EAR_BLINK_THRESHOLD:
        if frame_idx - last_actions.get("blink", -999) > cooldown:
            actions.append(("eye_blink", "Blink"))
            last_actions["blink"] = frame_idx

    # ── Smile ──
    mw = _mouth_width(lms, config.LOCK_MOUTH_LEFT_INDEX, config.LOCK_MOUTH_RIGHT_INDEX, W, H)
    if baseline_mouth and mw >= baseline_mouth * config.LOCK_SMILE_MOUTH_RATIO:
        if frame_idx - last_actions.get("smile", -999) > cooldown:
            actions.append(("smile", "Smile"))
            last_actions["smile"] = frame_idx

    # ── Head yaw (looking left/right) ──
    nose = lms[config.LOCK_POSE_NOSE_INDEX]
    eye_l = lms[config.LOCK_POSE_LEFT_EYE_INDEX]
    eye_r = lms[config.LOCK_POSE_RIGHT_EYE_INDEX]
    d_nose_l = abs(nose.x - eye_l.x)
    d_nose_r = abs(nose.x - eye_r.x)
    if d_nose_r > 1e-6:
        yaw_ratio = d_nose_l / d_nose_r
        if yaw_ratio < config.LOCK_YAW_LOOK_LEFT_THRESHOLD:
            if frame_idx - last_actions.get("look_left", -999) > cooldown:
                actions.append(("look_left", "Looking Left"))
                last_actions["look_left"] = frame_idx
        elif yaw_ratio > config.LOCK_YAW_LOOK_RIGHT_THRESHOLD:
            if frame_idx - last_actions.get("look_right", -999) > cooldown:
                actions.append(("look_right", "Looking Right"))
                last_actions["look_right"] = frame_idx

    return actions, ear, mw

# ────────────────────────────────────────────────
# Database loading
# ────────────────────────────────────────────────

def load_database():
    if not config.DB_NPZ_PATH.exists():
        return {}
    data = np.load(config.DB_NPZ_PATH, allow_pickle=True)
    return {k: data[k].astype(np.float32) for k in data}

# ────────────────────────────────────────────────
# Main loop
# ────────────────────────────────────────────────

def main():
    db = load_database()
    if not db:
        print("No database found. Run enroll.py first.")
        return

    names = sorted(db.keys())
    print("\nAvailable Identities:")
    for i, name in enumerate(names, 1):
        print(f" {i}. {name}")

    target = input("\nEnter name to LOCK onto (others will be recognized): ").strip()
    if not target:
        target = names[0] if names else ""
    if target not in db:
        print(f"'{target}' not in database. Using first entry." if names else "No identities available.")
        target = names[0] if names else ""

    print(f"Locking onto: {target}")

    # ── Initialize components ──
    detector = HaarFaceMesh5pt()
    embedder = ArcFaceEmbedderONNX(config.ARCFACE_MODEL_PATH)

    action_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True
    )

    # ── MQTT ──
    mqtt_client = mqtt_client_module.Client(
        mqtt_client_module.CallbackAPIVersion.VERSION1,
        client_id=MQTT_CLIENT_ID,
        clean_session=True
    )

    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        mqtt_client.loop_start()
        print(f"MQTT connected → {MQTT_BROKER}:{MQTT_PORT}  topic: {MQTT_TOPIC}")
    except Exception as e:
        print(f"MQTT connection failed: {e}")
        return

    db_matrix = np.stack([db[n] for n in names])
    cap = cv2.VideoCapture(config.CAMERA_INDEX)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # History file
    history_file = None
    if config.HISTORY_DIR:
        config.HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d%H%M%S")
        safe_name = target.replace(" ", "_").lower()
        path = config.HISTORY_DIR / f"{safe_name}_lock_{ts}.txt"
        history_file = open(path, "w", encoding="utf-8")
        history_file.write(f"# Lock history for {target}\n# Time, Action, Description\n")
        print(f"Logging to: {path}")

    # State variables
    prev_cx = None
    baseline_mw = None
    mw_samples = []
    last_actions = {}
    frame_idx = 0
    last_published_angle = 90
    t0 = time.time()
    frame_count = 0
    fps = 0.0

    DIST_THRESH = config.DEFAULT_DISTANCE_THRESHOLD

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        frame_idx += 1
        now = time.time()
        elapsed = now - t0
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            t0 = now
        frame_count += 1

        vis = frame.copy()
        detections = detector.detect(frame, max_faces=5)

        locked = None

        for face in detections:
            aligned, _ = align_face_5pt(frame, face.kps)
            emb = embedder.embed(aligned)

            dists = np.array([cosine_distance(emb, db_matrix[i]) for i in range(len(names))])
            best_idx = np.argmin(dists)
            best_dist = dists[best_idx]

            name = names[best_idx] if best_dist <= DIST_THRESH else "Unknown"
            is_target = (name == target)

            if is_target:
                if locked is None or best_dist < locked['dist']:
                    locked = {
                        'face': face,
                        'dist': best_dist,
                        'cx': (face.x1 + face.x2) / 2.0
                    }
            else:
                color = (0, 255, 255) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(vis, (face.x1, face.y1), (face.x2, face.y2), color, 2)
                cv2.putText(vis, f"{name} {best_dist:.2f}", (face.x1, face.y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if locked:
            f = locked['face']
            cx = locked['cx']

            cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 255, 0), 3)
            cv2.putText(vis, f"LOCKED: {target}", (f.x1, f.y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # ── Servo angle calculation & publish ──
            frame_w = frame.shape[1]
            norm_x = cx / frame_w
            angle = int(MIN_ANGLE + (MAX_ANGLE - MIN_ANGLE) * norm_x)
            angle = max(MIN_ANGLE, min(MAX_ANGLE, angle))

            # Invert if needed:
            # angle = MAX_ANGLE - angle

            if abs(angle - last_published_angle) >= ANGLE_HYSTERESIS:
                mqtt_client.publish(MQTT_TOPIC, str(angle))
                print(f"→ Servo angle: {angle:3d}°   (cx={cx:5.1f}/{frame_w})")
                last_published_angle = angle

            cv2.putText(vis, f"Servo: {angle}°", (f.x1, f.y1 - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # ── Action detection ──
            acts, _, mw = detect_actions(frame, action_mesh, prev_cx, cx,
                                         baseline_mw, frame_idx, last_actions)

            if mw is not None:
                mw_samples.append(mw)
                if len(mw_samples) > 20:
                    mw_samples.pop(0)
                if baseline_mw is None and len(mw_samples) >= 10:
                    baseline_mw = float(np.median(mw_samples))

            prev_cx = cx

            for act_type, desc in acts:
                cv2.putText(vis, f"ACTION: {desc}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if history_file:
                    history_file.write(f"{time.time():.2f}, {act_type}, {desc}\n")
                    history_file.flush()

        else:
            cv2.putText(vis, f"Searching for {target}...", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            prev_cx = None

        cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Face Lock & Track", vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    if history_file:
        history_file.close()
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    print("Exited cleanly.")

if __name__ == "__main__":
    main()