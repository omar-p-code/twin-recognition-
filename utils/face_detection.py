import cv2
import numpy as np
from PIL import Image as PILImage
import os


class FaceDetector:
    """
    Face detector with two tiers:
      1. MTCNN  — deep-learning detector, much more accurate on varied poses/lighting.
                  Used automatically when the `facenet-pytorch` package is installed.
      2. Haar cascade (OpenCV) — lightweight fallback, always available.
    """

    def __init__(self):
        self.mtcnn       = None
        self.face_cascade = None
        self.eye_cascade  = None
        self._load_detectors()

    def _load_detectors(self):
        # Try MTCNN first (pip install facenet-pytorch)
        try:
            from facenet_pytorch import MTCNN
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.mtcnn = MTCNN(
                keep_all=False,
                min_face_size=40,
                device=device,
                post_process=False,
            )
            print("✓ MTCNN face detector loaded (high accuracy mode)")
        except ImportError:
            print("! facenet-pytorch not installed — falling back to OpenCV Haar cascade")
            print("  Install with: pip install facenet-pytorch")
            self._load_haar()

    def _load_haar(self):
        try:
            cascade_path = cv2.data.haarcascades
            self.face_cascade = cv2.CascadeClassifier(
                os.path.join(cascade_path, "haarcascade_frontalface_default.xml")
            )
            self.eye_cascade = cv2.CascadeClassifier(
                os.path.join(cascade_path, "haarcascade_eye.xml")
            )
            print("✓ OpenCV Haar cascade loaded")
        except Exception as e:
            print(f"! Could not load Haar cascades: {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_and_crop_face(self, image_path, padding=30):
        """Detect the largest face and return a cropped PIL image."""
        try:
            img = (
                PILImage.open(image_path).convert("RGB")
                if isinstance(image_path, str)
                else image_path.convert("RGB")
            )
        except Exception as e:
            return None, f"Could not open image: {e}"

        if self.mtcnn is not None:
            return self._detect_mtcnn(img, padding)
        elif self.face_cascade is not None:
            return self._detect_haar(img, padding)
        else:
            img.thumbnail((300, 300), PILImage.LANCZOS)
            return img, "No detector available — using full image"

    def _detect_mtcnn(self, img, padding):
        """Use MTCNN for high-accuracy detection."""
        try:
            boxes, _ = self.mtcnn.detect(img)
            if boxes is not None and len(boxes) > 0:
                # Pick the largest box
                areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
                x1, y1, x2, y2 = boxes[int(np.argmax(areas))]

                # Add padding
                x1 = max(0, int(x1) - padding)
                y1 = max(0, int(y1) - padding)
                x2 = min(img.width,  int(x2) + padding)
                y2 = min(img.height, int(y2) + padding)

                cropped = img.crop((x1, y1, x2, y2))
                return cropped, "Face detected (MTCNN) ✓"

            # No face found — fall back to full image
            return img, "No face detected — using full image"

        except Exception as e:
            print(f"MTCNN error: {e}")
            return self._detect_haar(img, padding)

    def _detect_haar(self, img, padding):
        """Use OpenCV Haar cascade as a fallback."""
        try:
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            gray   = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )

            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(img.width,  x + w + padding)
                y2 = min(img.height, y + h + padding)
                cropped = img.crop((x1, y1, x2, y2))
                return cropped, "Face detected (Haar) ✓"

            return img, "No face detected — using full image"

        except Exception as e:
            return None, f"Detection error: {e}"

    def detect_faces_only(self, image_path):
        """Return list of bounding boxes [x1,y1,x2,y2]."""
        try:
            img = (
                PILImage.open(image_path).convert("RGB")
                if isinstance(image_path, str)
                else image_path
            )
            if self.mtcnn is not None:
                boxes, _ = self.mtcnn.detect(img)
                return boxes.tolist() if boxes is not None else []
            elif self.face_cascade is not None:
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                gray   = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                faces  = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
                )
                return faces.tolist() if len(faces) > 0 else []
        except Exception as e:
            print(f"Detection error: {e}")
        return []


face_detector = FaceDetector()
