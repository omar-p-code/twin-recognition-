# ui/face_detector.py - Lightweight face detection using OpenCV
import cv2
import numpy as np
from PIL import Image as PILImage
import os

class FaceDetector:
    """Lightweight face detector using OpenCV - no TensorFlow needed"""
    
    def __init__(self):
        self.face_cascade = None
        self.eye_cascade = None
        self._load_cascades()
    
    def _load_cascades(self):
        """Load OpenCV Haar cascades"""
        try:
            # OpenCV built-in cascades
            cascade_path = cv2.data.haarcascades
            self.face_cascade = cv2.CascadeClassifier(
                os.path.join(cascade_path, 'haarcascade_frontalface_default.xml')
            )
            self.eye_cascade = cv2.CascadeClassifier(
                os.path.join(cascade_path, 'haarcascade_eye.xml')
            )
            print("✓ OpenCV face detector loaded")
        except Exception as e:
            print(f"! Could not load cascades: {e}")
    
    def detect_and_crop(self, image_path, padding=30):
        """Detect face and return cropped image"""
        try:
            # Load image
            if isinstance(image_path, str):
                img = PILImage.open(image_path).convert("RGB")
            else:
                img = image_path
            
            # Convert to OpenCV format
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60)
            )
            
            if len(faces) > 0:
                # Get the largest face
                best_face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = best_face
                
                # Add padding
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(img.width - x, w + (2 * padding))
                h = min(img.height - y, h + (2 * padding))
                
                # Crop face
                cropped = img.crop((x, y, x + w, y + h))
                cropped.thumbnail((250, 250), PILImage.LANCZOS)
                
                return cropped, f"Face detected ✓"
            
            # Fallback: center crop
            size = min(img.width, img.height)
            left = (img.width - size) // 2
            top = (img.height - size) // 2
            cropped = img.crop((left, top, left + size, top + size))
            cropped.thumbnail((250, 250), PILImage.LANCZOS)
            
            return cropped, "No face found (center crop)"
            
        except Exception as e:
            print(f"Detection error: {e}")
            return None, f"Error: {str(e)[:30]}"
    
    def detect_faces_only(self, image_path):
        """Return list of face coordinates"""
        try:
            if isinstance(image_path, str):
                img = PILImage.open(image_path).convert("RGB")
            else:
                img = image_path
            
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60)
            )
            
            return faces.tolist() if len(faces) > 0 else []
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []