# Face Detection with InsightFace

A modular Python application that detects all human faces in JPEG, PNG, and bitmap images. It uses InsightFace for detection and OpenCV for image loading, processing, and annotation.

## Project layout

```text
app/
├── __init__.py
├── main.py          # command-line interface
├── face_detector.py # InsightFace model wrapper and box drawing
└── utils.py         # image validation, loading, and saving
requirements.txt
```

## Run locally

Python 3.9–3.11 is recommended. From the project directory:

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first command that loads the detector downloads the selected InsightFace model pack and caches it locally.

## Commands

All commands accept `.jpg`, `.jpeg`, `.png`, and `.bmp` files. CPU inference is the default.

```bash
# Is there at least one face?
python -m app.main detect path/to/photo.jpg

# How many faces were found?
python -m app.main count path/to/photo.png

# Get JSON including each box and confidence score
python -m app.main analyze path/to/photo.bmp

# Draw boxes and save an annotated image
python -m app.main draw path/to/photo.jpg output/annotated.jpg
```

For a GPU provider, if the required ONNX Runtime GPU setup is installed:

```bash
python -m app.main --gpu analyze path/to/photo.jpg
```

Use `python -m app.main --help` or `python -m app.main draw --help` for command help.

