# Attendance System Updated

A webcam attendance prototype using InsightFace. It detects faces, matches
them with photos in `Students`, and saves dated attendance CSV files in
`Logs`. Recognition runs in the background while the camera display updates.

## Setup

From Command Prompt in this folder:

```cmd
python -m pip install -r requirements.txt
```

Put one clear face photo per person in `Students`. Name each photo after
the person, such as `Student Name.jpg`. The first run may download the
InsightFace `buffalo_l` model and create `face_cache.pkl`.
Photos, face-cache data, and attendance logs stay local and are not included
in the GitHub repository.

## Run

```cmd
python "Attendance Updated.py"
```

Press **Q** in the camera window to quit. This is a prototype, not a secure
identity or attendance system. Obtain consent before storing or using
anyone's face photo.

