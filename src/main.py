from os import listdir, path, makedirs
from time import time, sleep
from cv2 import VideoCapture, imread
from numpy import dot
from numpy.linalg import norm
from pywslocker import lock
from insightface.app import FaceAnalysis

# CONFIG
IMAGE_FOLDER_PATH = "assets/images"
ABSENCE_THRESHOLD = 10.0
SIM_THRESHOLD = 0.32 

def load_insightface():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

def load_person_embeddings(app, folder):
    embs = []
    if not path.isdir(folder):
        makedirs(folder, exist_ok=True)
        print(f"[INFO] Created folder: {folder}")
        print("[INFO] Add your face images there and rerun.")
        return embs

    for file in listdir(folder):
        path = path.join(folder, file)
        img = imread(path)
        if img is None:
            continue
        faces = app.get(img)
        if len(faces) == 0:
            continue
        embs.append(faces[0].embedding)
        print(f"[INFO] Loaded embedding from {file}")

    return embs

def cosine_sim(a, b):
    return dot(a, b) / (norm(a) * norm(b))

def person_in_frame(app, frame, known_embs, threshold=SIM_THRESHOLD):
    if not known_embs:
        return False

    faces = app.get(frame)
    if len(faces) == 0:
        return False

    for face in faces:
        emb = face.embedding
        for known in known_embs:
            sim = cosine_sim(emb, known)
            if sim > threshold:
                return True
    return False

def wait_until_person_back(app, cam, known_embs, check_interval=0.5):
    # After lock, keep checking in background until person is back
    while True:
        ret, frame = cam.read()
        if not ret:
            time.sleep(check_interval)
            continue
        if person_in_frame(app, frame, known_embs):
            return
        time.sleep(check_interval)

def main():
    app = load_insightface()
    known_embs = load_person_embeddings(app, IMAGE_FOLDER_PATH)
    if not known_embs:
        print("[ERROR] No valid embeddings found. Add face images and rerun.")
        return

    cam = VideoCapture(0)
    if not cam.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    last_seen = time()

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                sleep(0.1)
                continue

            if person_in_frame(app, frame, known_embs):
                last_seen = time()
            else:
                elapsed = time() - last_seen
                if elapsed >= ABSENCE_THRESHOLD:
                    print("[INFO] Person absent for 10s. Locking...")
                    lock()
                    wait_until_person_back(app, cam, known_embs)
                    print("[INFO] Person detected again. Resetting timer.")
                    last_seen = time()

            sleep(0.1)
    finally:
        cam.release()

if __name__ == "__main__":
    main()
