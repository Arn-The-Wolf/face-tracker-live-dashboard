import sys
import time
from pathlib import Path
import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None

from . import config
from .haar_5pt import Haar5ptDetector, align_face_5pt
from .embed import ArcFaceEmbedderONNX


def load_database():
    """Load enrolled face database."""
    if not config.DB_NPZ_PATH.exists():
        return {}
    data = np.load(str(config.DB_NPZ_PATH), allow_pickle=True)
    return {k: data[k].astype(np.float32) for k in data.files}


def cosine_distance(a, b):
    """Compute cosine distance between two vectors."""
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    return 1.0 - float(np.dot(a, b))


def _ear_from_landmarks(landmarks_list, indices, W, H):
    """Eye Aspect Ratio from 6 landmark indices."""
    pts = []
    for i in indices:
        lm = landmarks_list[i]
        pts.append((lm.x * W, lm.y * H))
    p1, p2, p3, p4, p5, p6 = [np.array(p) for p in pts]
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    h = np.linalg.norm(p1 - p4)
    if h < 1e-6:
        return 0.5
    return (v1 + v2) / (2.0 * h)


def _mouth_width_from_landmarks(landmarks_list, left_idx, right_idx, W, H):
    """Mouth width in pixels."""
    l = landmarks_list[left_idx]
    r = landmarks_list[right_idx]
    return np.hypot((r.x - l.x) * W, (r.y - l.y) * H)


def get_full_landmarks(frame, mesh):
    """Run MediaPipe Face Mesh on frame; return first face landmark list or None."""
    if mesh is None:
        return None
    H, W = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None
    return results.multi_face_landmarks[0].landmark


def detect_actions(frame, mesh, prev_center_x, center_x, prev_ear, prev_mouth_width,
                   baseline_mouth_width, frame_idx, last_action_frame):
    """Detect actions: face_moved_left, face_moved_right, eye_blink, smile."""
    actions = []
    H, W = frame.shape[:2]
    cooldown = config.LOCK_ACTION_COOLDOWN_FRAMES

    if prev_center_x is not None:
        dx = center_x - prev_center_x
        if dx <= -config.LOCK_MOVEMENT_THRESHOLD_PX:
            if frame_idx - last_action_frame.get("face_moved_left", -999) >= cooldown:
                actions.append(("face_moved_left", "face moved left"))
                last_action_frame["face_moved_left"] = frame_idx
        elif dx >= config.LOCK_MOVEMENT_THRESHOLD_PX:
            if frame_idx - last_action_frame.get("face_moved_right", -999) >= cooldown:
                actions.append(("face_moved_right", "face moved right"))
                last_action_frame["face_moved_right"] = frame_idx

    landmarks_list = get_full_landmarks(frame, mesh)
    if landmarks_list is None:
        return actions

    ear_left = _ear_from_landmarks(landmarks_list, config.LOCK_EAR_LEFT_INDICES, W, H)
    ear_right = _ear_from_landmarks(landmarks_list, config.LOCK_EAR_RIGHT_INDICES, W, H)
    ear = (ear_left + ear_right) / 2.0

    if ear < config.LOCK_EAR_BLINK_THRESHOLD:
        if frame_idx - last_action_frame.get("eye_blink", -999) >= cooldown:
            actions.append(("eye_blink", "eye blink"))
            last_action_frame["eye_blink"] = frame_idx

    mouth_width = _mouth_width_from_landmarks(
        landmarks_list, config.LOCK_MOUTH_LEFT_INDEX, config.LOCK_MOUTH_RIGHT_INDEX, W, H
    )
    if baseline_mouth_width is not None and baseline_mouth_width > 1.0:
        if mouth_width >= baseline_mouth_width * config.LOCK_SMILE_MOUTH_RATIO:
            if frame_idx - last_action_frame.get("smile", -999) >= cooldown:
                actions.append(("smile", "smile or laugh"))
                last_action_frame["smile"] = frame_idx

    return actions


def main():
    """Face Locking: select one identity, lock onto that face, track and record actions."""
    db = load_database()
    if not db:
        print("ERROR: No enrolled identities. Run: python -m src.enroll")
        return False

    names = sorted(db.keys())
    print("\nEnrolled identities:")
    for i, n in enumerate(names, 1):
        print(f"  {i}. {n}")
    print("\nEnter the name of the identity to lock (exact match): ", end="")
    try:
        choice = input().strip()
    except EOFError:
        choice = names[0] if names else ""
    if not choice:
        choice = names[0] if names else ""
    if choice not in db:
        print(f"ERROR: '{choice}' not in database. Choose from: {names}")
        return False
    lock_identity = choice
    print("Will lock onto:", lock_identity)

    config.ensure_dirs()
    config.HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    detector = Haar5ptDetector(min_size=config.HAAR_MIN_SIZE, smooth_alpha=0.8, debug=False)
    # Note: Haar5ptDetector returns FaceKpsBox objects, which have .kps (5 keypoints)
    
    embedder = ArcFaceEmbedderONNX(config.ARCFACE_MODEL_PATH)
    
    embeddings_matrix = np.stack([db[n].reshape(-1) for n in names], axis=0)
    lock_idx = names.index(lock_identity)
    threshold = config.DEFAULT_DISTANCE_THRESHOLD

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return False

    locked = False
    fail_count = 0
    history_file = None
    history_path = None
    prev_center_x = None
    baseline_mouth_width = None
    mouth_width_samples = []
    last_action_frame = {}
    prev_ear = None
    prev_mouth_width = None
    frame_idx = 0

    # Initialize FaceMesh once for action detection
    action_mesh = None
    if mp:
        action_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    print("\nFace Locking - When the selected face appears, system will lock. q=Quit")

    t0 = time.time()
    frame_count = 0
    fps = 0.0
    
    # Optimization: Verify identity only every N frames when locked
    VERIFY_INTERVAL = 15 
    frames_since_verify = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            elapsed = time.time() - t0
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                t0 = time.time()
            frame_count += 1

            vis = frame.copy()
            H, W = frame.shape[:2]
            
            # Use smaller max_faces to speed up Haar if possible, but 5 is okay
            faces = detector.detect(frame, max_faces=5)

            if not locked:
                for face in faces:
                    # align_face_5pt uses face.kps
                    aligned, _ = align_face_5pt(frame, face.kps, out_size=(112, 112))
                    res = embedder.embed(aligned)
                    query_emb = res.embedding
                    
                    dists = np.array([cosine_distance(query_emb, embeddings_matrix[i]) for i in range(len(names))])
                    best_idx = int(np.argmin(dists))
                    best_dist = dists[best_idx]
                    
                    if best_idx == lock_idx and best_dist <= threshold:
                        locked = True
                        fail_count = 0
                        ts = time.strftime("%Y%m%d%H%M%S", time.localtime())
                        safe_name = lock_identity.replace(" ", "_").lower()
                        history_path = config.HISTORY_DIR / (safe_name + "_history_" + ts + ".txt")
                        history_file = open(history_path, "w", encoding="utf-8")
                        history_file.write("# Face Lock history: " + lock_identity + "\n")
                        history_file.write("# Started: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n")
                        history_file.write("# Format: timestamp  action_type  description\n# ---\n")
                        history_file.flush()
                        prev_center_x = (face.x1 + face.x2) / 2.0
                        mouth_width_samples = []
                        baseline_mouth_width = None
                        frames_since_verify = 0
                        print("LOCKED onto", lock_identity, "Recording to", history_path.name)
                        break

            else:
                matched_face = None
                best_dist = 0.0 # Placeholder
                
                # Spatial tracking optimization
                # If we have a previous center, look for the face closest to it
                candidates = []
                if prev_center_x is not None:
                    for face in faces:
                        cx = (face.x1 + face.x2) / 2.0
                        dist_x = abs(cx - prev_center_x)
                        candidates.append((dist_x, face))
                    candidates.sort(key=lambda x: x[0])
                
                # Default to verifying every frame if no candidates or tracking lost, 
                # but valid candidates mean we can skip heavy embedder
                should_verify = (frames_since_verify >= VERIFY_INTERVAL) or (not candidates)
                
                if not should_verify and candidates:
                    # Assume the closest face is the target (Spatial Lock)
                    # Only check if it's reasonably close (e.g. within 100px jump)
                    dist_x, candidate_face = candidates[0]
                    if dist_x < 150: # valid track
                         matched_face = candidate_face
                         best_dist = -1.0 # Virtual distance indicating "tracked"
                         fail_count = 0
                         frames_since_verify += 1
                    else:
                        should_verify = True
                
                if should_verify:
                    # Run full recognition on candidates to find the lock
                    frames_since_verify = 0
                    for face in faces:
                        aligned, _ = align_face_5pt(frame, face.kps, out_size=(112, 112))
                        res = embedder.embed(aligned)
                        query_emb = res.embedding
                        
                        dists = np.array([cosine_distance(query_emb, embeddings_matrix[i]) for i in range(len(names))])
                        idx = int(np.argmin(dists))
                        d = dists[idx]
                        
                        if idx == lock_idx and d <= threshold:
                            matched_face = face
                            best_dist = d
                            fail_count = 0
                            break
                        # Heuristic for difficult angles: if only 1 face and it was valid recently, accept slightly worse match?
                        # Keeping it strict for now to avoid drift
                        
                    if matched_face is None:
                        # If verify failed but we had a face, maybe just a partial occlusion?
                        # Fail count handles this.
                        pass

                if matched_face is None:
                    fail_count += 1
                    if fail_count >= config.LOCK_RELEASE_FRAMES:
                        locked = False
                        if history_file:
                            history_file.close()
                            history_file = None
                        print("Lock released (face not seen for", config.LOCK_RELEASE_FRAMES, "frames).")
                else:
                    center_x = (matched_face.x1 + matched_face.x2) / 2.0
                    action_list = detect_actions(
                        frame, action_mesh, prev_center_x, center_x, prev_ear, prev_mouth_width,
                        baseline_mouth_width, frame_idx, last_action_frame
                    )
                    ts_val = time.time()
                    for action_type, desc in action_list:
                        line = "%.2f  %s  %s\n" % (ts_val, action_type, desc)
                        if history_file:
                            history_file.write(line)
                            history_file.flush()

                    landmarks_list = get_full_landmarks(frame, action_mesh)
                    if landmarks_list:
                        mw = _mouth_width_from_landmarks(
                            landmarks_list, config.LOCK_MOUTH_LEFT_INDEX,
                            config.LOCK_MOUTH_RIGHT_INDEX, W, H
                        )
                        ear_left = _ear_from_landmarks(landmarks_list, config.LOCK_EAR_LEFT_INDICES, W, H)
                        ear_right = _ear_from_landmarks(landmarks_list, config.LOCK_EAR_RIGHT_INDICES, W, H)
                        prev_ear = (ear_left + ear_right) / 2.0
                        mouth_width_samples.append(mw)
                        if len(mouth_width_samples) > 30:
                            mouth_width_samples.pop(0)
                        if baseline_mouth_width is None and len(mouth_width_samples) >= 15:
                            baseline_mouth_width = float(np.median(mouth_width_samples))
                        prev_mouth_width = mw
                    prev_center_x = center_x

                    cv2.rectangle(vis, (matched_face.x1, matched_face.y1), (matched_face.x2, matched_face.y2), (0, 255, 0), 4)
                    cv2.putText(vis, "LOCKED: " + lock_identity, (matched_face.x1, max(0, matched_face.y1 - 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    if best_dist < 0: # Virtual Distance
                        cv2.putText(vis, "tracked", (matched_face.x1, matched_face.y2 + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 50), 1)
                    else:
                        cv2.putText(vis, "dist=%.3f" % best_dist, (matched_face.x1, matched_face.y2 + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            if locked:
                for face in faces:
                    cx = (face.x1 + face.x2) / 2.0
                    if prev_center_x is not None and abs(cx - prev_center_x) < 50:
                        continue
                    cv2.rectangle(vis, (face.x1, face.y1), (face.x2, face.y2), (0, 0, 255), 2)
                    cv2.putText(vis, "Other", (face.x1, face.y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            else:
                for face in faces:
                    cv2.rectangle(vis, (face.x1, face.y1), (face.x2, face.y2), (0, 255, 0), 2)
                    cv2.putText(vis, "Looking for lock...", (face.x1, face.y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            header = "Lock target: " + lock_identity + " | " + ("LOCKED" if locked else "Searching...") + " | FPS: %.1f" % fps
            cv2.putText(vis, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis, "q=quit", (10, vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow("Face Locking", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        if action_mesh:
            action_mesh.close()
        cap.release()
        cv2.destroyAllWindows()
        if history_file:
            history_file.close()
        if history_path:
            print("History saved to:", history_path)

    print("Face Locking ended.")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)