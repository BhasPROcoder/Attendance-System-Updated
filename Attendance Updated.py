import os
import sys
import logging
import warnings
import contextlib
import threading

# --- Silence warnings everywhere (module= restriction is unreliable, drop it) ---
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Silence everything except your own print() calls ---
os.environ["ORT_LOG_SEVERITY_LEVEL"] = "3"       # ONNXRuntime: only errors
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 4)
logging.disable(logging.CRITICAL)                # kill Python logging (insightface uses this)


_stdio_lock = threading.Lock()


@contextlib.contextmanager
def suppress_stdout_stderr():
    """Silence noisy print()-based logs from insightface/onnxruntime.
    Thread-safe: only one thread may swap sys.stdout/stderr at a time,
    since sys.stdout/stderr are process-global, not per-thread."""
    with _stdio_lock:
        with open(os.devnull, "w") as devnull:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = devnull, devnull
            try:
                yield
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr


import cv2
import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from insightface.app import FaceAnalysis

# ======================================================
# SETTINGS
# ======================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STUDENT_FOLDER = os.path.join(BASE_DIR, "Students")
LOG_FOLDER = os.path.join(BASE_DIR, "Logs")
CACHE_FILE = os.path.join(BASE_DIR, "face_cache.pkl")

SIMILARITY_THRESHOLD = 0.55
INFER_SIZE = (240, 180)        # size used for the (slower) recognition pass
DET_SIZE = (160, 160)          # detector's internal input size

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# How close a fast-detected box has to be (in original-frame pixels) to a
# recently recognized face to inherit its name label.
MATCH_DISTANCE = 90

cv2.setNumThreads(1)

CPU_CORES = os.cpu_count() or 4
os.environ["OMP_NUM_THREADS"] = str(CPU_CORES)

# ======================================================
# INITIALIZE INSIGHTFACE
# ======================================================

print("[INFO] Loading InsightFace...")

with suppress_stdout_stderr():
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
        allowed_modules=["detection", "recognition"],
    )
    app.prepare(
        ctx_id=-1,          # CPU mode
        det_size=DET_SIZE,
    )

print("[INFO] InsightFace ready.")

# ======================================================
# LOAD OR CREATE FACE CACHE
# ======================================================

if os.path.exists(CACHE_FILE):

    print("[INFO] Loading cached face embeddings...")

    with open(CACHE_FILE, "rb") as f:
        known_names, known_embeddings = pickle.load(f)

else:

    print("[INFO] Encoding student faces...")

    known_names = []
    known_embeddings = []

    for filename in os.listdir(STUDENT_FOLDER):

        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(STUDENT_FOLDER, filename)

        image = cv2.imread(path)

        with suppress_stdout_stderr():
            faces = app.get(image)

        if len(faces) == 0:
            print(f"[WARNING] No face found in {filename}")
            continue

        known_names.append(os.path.splitext(filename)[0])
        known_embeddings.append(faces[0].embedding)

        print(f"Loaded: {filename}")

    with open(CACHE_FILE, "wb") as f:
        pickle.dump((known_names, known_embeddings), f)

    print("[INFO] Face cache saved.")

known_embeddings = np.array(known_embeddings)
known_norms = np.linalg.norm(known_embeddings, axis=1)

# ======================================================
# ATTENDANCE FILE
# ======================================================

os.makedirs(LOG_FOLDER, exist_ok=True)

today = datetime.now().strftime("%d-%m-%Y")

csv_file = os.path.join(
    LOG_FOLDER,
    f"Attendance_{today}.csv"
)

if os.path.exists(csv_file):
    attendance = pd.read_csv(csv_file)
else:
    attendance = pd.DataFrame(columns=["Name", "Time"])

marked = set(attendance["Name"])

# ======================================================
# THREADED CAMERA READER
# ======================================================

class CameraStream:
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.lock = threading.Lock()
        self.frame = None
        self.stopped = False

        ret, frame = self.cap.read()
        if ret:
            self.frame = frame

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def release(self):
        self.stopped = True
        self.thread.join(timeout=1)
        self.cap.release()


# ======================================================
# BACKGROUND RECOGNIZER (slower — detection + recognition + matching)
# ======================================================
# Runs continuously but only feeds a name cache. It does NOT drive what
# gets drawn on screen, so its speed no longer determines box lag.

class Recognizer:
    def __init__(self, camera):
        self.camera = camera
        self.lock = threading.Lock()
        self.last_faces = []   # list of (x1, y1, x2, y2, name) in full-frame coords
        self.stopped = False
        self.scale_x = CAMERA_WIDTH / INFER_SIZE[0]
        self.scale_y = CAMERA_HEIGHT / INFER_SIZE[1]
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stopped:
            frame = self.camera.read()
            if frame is None:
                continue

            small = cv2.resize(frame, INFER_SIZE)

            # app.get() triggers face_align's SimilarityTransform.estimate(),
            # which emits the FutureWarning/print noise on every frame —
            # suppress stdout/stderr here too so it doesn't spam the console.
            with suppress_stdout_stderr():
                faces = app.get(small)

            results = []

            for face in faces:
                embedding = face.embedding

                similarities = np.dot(
                    known_embeddings,
                    embedding
                ) / (
                    known_norms
                    * np.linalg.norm(embedding)
                )

                best_index = np.argmax(similarities)
                score = similarities[best_index]

                if score >= SIMILARITY_THRESHOLD:
                    name = known_names[best_index]

                    if name not in marked:
                        now = datetime.now()
                        time_str = now.strftime("%H:%M:%S")
                        attendance.loc[len(attendance)] = [
                            name,
                            time_str
                        ]
                        marked.add(name)
                        print(f"[LOGGED] {name} at {time_str}")

                    x1, y1, x2, y2 = face.bbox.astype(int)
                    x1 = int(x1 * self.scale_x)
                    y1 = int(y1 * self.scale_y)
                    x2 = int(x2 * self.scale_x)
                    y2 = int(y2 * self.scale_y)

                    results.append((x1, y1, x2, y2, name))

            with self.lock:
                self.last_faces = results

    def get_names(self):
        with self.lock:
            return list(self.last_faces)

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=1)


def match_name(box, named_boxes):
    """Find the name whose last-known box center is closest to `box`'s
    center, within MATCH_DISTANCE pixels. Returns None if no good match."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    best_name = None
    best_dist = MATCH_DISTANCE

    for nx1, ny1, nx2, ny2, name in named_boxes:
        ncx, ncy = (nx1 + nx2) / 2, (ny1 + ny2) / 2
        dist = ((cx - ncx) ** 2 + (cy - ncy) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_name = name

    return best_name


camera = CameraStream(0, CAMERA_WIDTH, CAMERA_HEIGHT)
recognizer = Recognizer(camera)

print("[INFO] Camera started.")
print("[INFO] Press Q to quit.")

det_scale_x = CAMERA_WIDTH / DET_SIZE[0]
det_scale_y = CAMERA_HEIGHT / DET_SIZE[1]

# ======================================================
# MAIN LOOP — fast per-frame detection only, drawn immediately
# ======================================================

while True:

    frame = camera.read()

    if frame is None:
        continue

    display = frame

    # Cheap, detection-only pass every displayed frame keeps the box
    # glued to the current head position instead of trailing behind.
    small = cv2.resize(frame, DET_SIZE)

    with suppress_stdout_stderr():
        bboxes, _ = app.det_model.detect(small, max_num=0, metric="default")

    named_boxes = recognizer.get_names()

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox[:4].astype(int)
        x1 = int(x1 * det_scale_x)
        y1 = int(y1 * det_scale_y)
        x2 = int(x2 * det_scale_x)
        y2 = int(y2 * det_scale_y)

        name = match_name((x1, y1, x2, y2), named_boxes) or "..."

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            display,
            name,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Attendance System", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ======================================================
# SAVE
# ======================================================
# Stop the background recognizer thread FIRST so nothing is still inside
# suppress_stdout_stderr() while we do our final prints/save below.

recognizer.stop()
camera.release()
cv2.destroyAllWindows()

attendance.to_csv(csv_file, index=False)

print(f"[INFO] Attendance saved to {csv_file}")
