# utils/face_detection.py - Fixed version
import cv2
import numpy as np
from PIL import Image
import os

# Simple face detector that always works
class FaceDetector:
    def __init__(self):
        self.has_mtcnn = False
        self.detector = None
        
        # Try to load MTCNN if available
        try:
            from mtcnn import MTCNN
            self.detector = MTCNN()
            self.has_mtcnn = True
            print("✓ MTCNN face detector loaded")
        except ImportError:
            print("! MTCNN not available, using simple fallback")
    
    def detect_and_crop_face(self, image_path, add_padding=20):
        """
        Detect face in image and return cropped face
        Returns: (PIL Image of cropped face, message)
        """
        try:
            # Load image
            if isinstance(image_path, str):
                img = Image.open(image_path).convert("RGB")
                img_np = np.array(img)
            else:
                img = image_path
                img_np = np.array(img)
            
            # Try to detect face
            face_cropped = None
            message = ""
            
            if self.has_mtcnn and self.detector:
                # Use MTCNN for face detection
                faces = self.detector.detect_faces(img_np)
                
                if faces:
                    # Get the largest face
                    best_face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
                    x, y, w, h = best_face['box']
                    
                    # Add padding
                    x = max(0, x - add_padding)
                    y = max(0, y - add_padding)
                    w = min(img.width - x, w + (2 * add_padding))
                    h = min(img.height - y, h + (2 * add_padding))
                    
                    # Crop face
                    face_cropped = img.crop((x, y, x + w, y + h))
                    message = f"Face detected (conf: {best_face['confidence']:.2f})"
                else:
                    # No face detected, use full image
                    face_cropped = img
                    message = "No face detected, using full image"
            else:
                # Fallback: just resize the image (mock detection)
                face_cropped = img
                img.thumbnail((200, 200))
                message = "Mock detection: using full image"
            
            # Make square for circular display
            if face_cropped:
                size = max(face_cropped.width, face_cropped.height)
                squared = Image.new('RGB', (size, size), (0, 0, 0))
                x = (size - face_cropped.width) // 2
                y = (size - face_cropped.height) // 2
                squared.paste(face_cropped, (x, y))
                squared.thumbnail((250, 250))
                return squared, message
            
            return None, "Could not process image"
            
        except Exception as e:
            print(f"Face detection error: {e}")
            return None, f"Error: {str(e)[:30]}"

# Create global instance
face_detector = FaceDetector()