"""Command-line interface for face detection."""

import argparse
import json
import sys
from pathlib import Path

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect human faces with InsightFace.")
    parser.add_argument(
        "--model",
        default="buffalo_l",
        help="InsightFace model pack to use (default: buffalo_l).",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use the default GPU provider when available; CPU is the default.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command, help_text in (
        ("detect", "Report whether the image contains at least one face."),
        ("count", "Print the number of detected faces."),
        ("analyze", "Print JSON with detection status, count, boxes, and scores."),
    ):
        command_parser = subparsers.add_parser(command, help=help_text)
        command_parser.add_argument("image", type=Path, help="Input JPEG, PNG, or bitmap image.")

    draw_parser = subparsers.add_parser("draw", help="Draw bounding boxes around detected faces.")
    draw_parser.add_argument("image", type=Path, help="Input JPEG, PNG, or bitmap image.")
    draw_parser.add_argument("output", type=Path, help="Output JPEG, PNG, or bitmap image.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        # Keep imports after argument parsing so --help works even before the
        # optional computer-vision dependencies have been installed.
        from app.face_detector import FaceDetector
        from app.utils import load_image, save_image

        image = load_image(args.image)
        detector = FaceDetector(model_name=args.model, ctx_id=0 if args.gpu else -1)
        faces = detector.detect(image)

        if args.command == "detect":
            print("Face detected: yes" if faces else "Face detected: no")
        elif args.command == "count":
            print(len(faces))
        elif args.command == "analyze":
            print(
                json.dumps(
                    {
                        "image": str(args.image),
                        "face_detected": bool(faces),
                        "face_count": len(faces),
                        "faces": [
                            {"bounding_box": list(face.bounding_box), "confidence": face.confidence}
                            for face in faces
                        ],
                    },
                    indent=2,
                )
            )
        elif args.command == "draw":
            output = detector.draw_bounding_boxes(image, faces)
            save_image(output, args.output)
            print(f"Detected {len(faces)} face(s). Saved annotated image to {args.output}")
        return 0
    except (FileNotFoundError, OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
