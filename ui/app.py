# ui/app.py - Fixed with cropped face display
import sys
import os
import io
from PIL import Image as PILImage

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.image import Image
from kivy.graphics import Color, RoundedRectangle
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.metrics import dp, sp
from kivy.uix.popup import Popup
from kivy.uix.progressbar import ProgressBar
from kivy.core.image import Image as CoreImage

# Mobile screen size
Window.size = (360, 640)
Window.clearcolor = (0.07, 0.07, 0.1, 1)

# Colors
SIDEBAR = (0.12, 0.12, 0.18, 1)
PRIMARY = (0.4, 0.6, 0.9, 1)
SUCCESS = (0.3, 0.7, 0.5, 1)
WARNING = (0.95, 0.6, 0.2, 1)
DANGER = (0.9, 0.3, 0.3, 1)
TEXT = (0.9, 0.9, 0.95, 1)
TEXT_DIM = (0.6, 0.6, 0.7, 1)

# ================== FACE DETECTION ==================
class FaceDetector:
    """Detect and crop face from image"""
    
    def __init__(self):
        self.has_mtcnn = False
        self.detector = None
        
        # Try to load MTCNN
        try:
            from mtcnn import MTCNN
            self.detector = MTCNN()
            self.has_mtcnn = True
            print("✓ MTCNN face detector loaded")
        except ImportError:
            print("! MTCNN not available, using simple fallback")
    
    def detect_and_crop(self, image_path, padding=30):
        """Detect face and return cropped face image"""
        try:
            # Load image
            if isinstance(image_path, str):
                img = PILImage.open(image_path).convert("RGB")
            else:
                img = image_path
            
            original = img.copy()
            
            # Try MTCNN first
            if self.has_mtcnn and self.detector:
                import numpy as np
                img_np = np.array(img)
                faces = self.detector.detect_faces(img_np)
                
                if faces:
                    # Get the largest face
                    best_face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
                    x, y, w, h = best_face['box']
                    
                    # Add padding
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(original.width - x, w + (2 * padding))
                    h = min(original.height - y, h + (2 * padding))
                    
                    # Crop face
                    cropped = original.crop((x, y, x + w, y + h))
                    
                    # Make square for circular display
                    size = max(cropped.width, cropped.height)
                    squared = PILImage.new('RGB', (size, size), (0, 0, 0))
                    x_offset = (size - cropped.width) // 2
                    y_offset = (size - cropped.height) // 2
                    squared.paste(cropped, (x_offset, y_offset))
                    
                    # Resize for display
                    squared.thumbnail((250, 250))
                    
                    return squared, f"✓ Face detected (conf: {best_face['confidence']:.2f})"
            
            # Fallback: center crop (mock face detection)
            size = min(original.width, original.height)
            left = (original.width - size) // 2
            top = (original.height - size) // 2
            cropped = original.crop((left, top, left + size, top + size))
            cropped.thumbnail((250, 250))
            
            return cropped, "! No face detected (using center crop)"
            
        except Exception as e:
            print(f"Detection error: {e}")
            return None, f"Error: {str(e)[:30]}"

# Create global detector
face_detector = FaceDetector()

# ================== SIMPLE FACE COMPARISON ==================
def compare_faces_simple(face1, face2):
    """Simple comparison without heavy models"""
    try:
        import numpy as np
        
        # Resize to same size
        face1_resized = face1.resize((64, 64))
        face2_resized = face2.resize((64, 64))
        
        # Convert to arrays
        arr1 = np.array(face1_resized).astype(np.float32)
        arr2 = np.array(face2_resized).astype(np.float32)
        
        # Calculate MSE
        mse = np.mean((arr1 - arr2) ** 2)
        max_mse = 255 ** 2
        distance = min(1.0, mse / max_mse)
        
        if distance < 0.35:
            result = "Same Person"
        elif distance < 0.65:
            result = "Twins"
        else:
            result = "Different People"
        
        return distance, result
        
    except Exception as e:
        print(f"Comparison error: {e}")
        return 0.5, "Comparison Failed"

# ================== UI COMPONENTS ==================

class CircularImage(Image):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.allow_stretch = True
        self.keep_ratio = True
        self.bind(size=self._update_rounding, pos=self._update_rounding)
        
    def _update_rounding(self, *args):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(1, 1, 1, 1)
            if self.width > 0:
                RoundedRectangle(pos=self.pos, size=self.size, radius=[self.width/2])

class CroppedFaceDisplay(BoxLayout):
    """Display cropped face with status"""
    def __init__(self, title="Face", **kwargs):
        super().__init__(orientation='vertical', spacing=dp(5), **kwargs)
        
        # Image container
        self.image = CircularImage()
        self.image.size_hint = (1, 0.8)
        self.add_widget(self.image)
        
        # Title label
        self.title_label = Label(
            text=title, 
            size_hint=(1, 0.1), 
            color=TEXT_DIM, 
            font_size=sp(12), 
            bold=True
        )
        self.add_widget(self.title_label)
        
        # Status label
        self.status_label = Label(
            text="No image", 
            size_hint=(1, 0.1), 
            color=TEXT_DIM, 
            font_size=sp(9),
            halign='center'
        )
        self.status_label.bind(size=self.status_label.setter('text_size'))
        self.add_widget(self.status_label)
    
    def set_face(self, pil_image, status_text):
        if pil_image:
            # Convert PIL to texture
            data = io.BytesIO()
            pil_image.save(data, format='png')
            data.seek(0)
            self.image.texture = CoreImage(data, ext='png').texture
            self.title_label.color = SUCCESS
            self.status_label.text = status_text[:30]
            self.status_label.color = SUCCESS
            return True
        else:
            self.image.texture = None
            self.title_label.color = DANGER
            self.status_label.text = status_text[:30]
            self.status_label.color = DANGER
            return False
    
    def clear(self):
        self.image.texture = None
        self.title_label.color = TEXT_DIM
        self.status_label.text = "No image"
        self.status_label.color = TEXT_DIM

class LoadingPopup(Popup):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.title = "Processing"
        self.size_hint = (0.7, 0.3)
        self.auto_dismiss = False
        layout = BoxLayout(orientation='vertical', padding=dp(15), spacing=dp(10))
        self.msg_label = Label(text="Comparing faces...", color=TEXT, font_size=sp(14))
        self.progress = ProgressBar(max=100, value=50)
        layout.add_widget(self.msg_label)
        layout.add_widget(self.progress)
        self.content = layout

# ================== MAIN APPLICATION ==================

class TwinUI(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='horizontal', **kwargs)
        
        self.img1_path = None
        self.img2_path = None
        self.cropped_face1 = None
        self.cropped_face2 = None
        self.detection_enabled = True
        
        # Setup UI
        self._setup_sidebar()
        self._setup_main()
        
        print("✓ UI ready - Cropped faces will appear when selected")
    
    def _setup_sidebar(self):
        """Setup sidebar with file chooser"""
        sidebar = BoxLayout(orientation='vertical', size_hint=(0.38, 1), padding=dp(8), spacing=dp(8))
        
        with sidebar.canvas.before:
            Color(*SIDEBAR)
            RoundedRectangle(pos=sidebar.pos, size=sidebar.size, radius=[dp(0), dp(20), dp(20), dp(0)])
        
        # Title
        title = Label(text="TWIN\nMATCH", font_size=sp(18), bold=True, color=PRIMARY, size_hint=(1, 0.08))
        sidebar.add_widget(title)
        
        # File chooser
        self.filechooser = FileChooserIconView(
            size_hint=(1, 0.55),
            path=os.path.expanduser('~'),
            filters=['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']
        )
        sidebar.add_widget(self.filechooser)
        
        # Buttons
        btn1 = Button(text="📷 SELECT FACE 1", size_hint=(1, 0.09), background_color=PRIMARY, color=(1,1,1,1), font_size=sp(12))
        btn1.bind(on_press=self.select_face1)
        sidebar.add_widget(btn1)
        
        btn2 = Button(text="📸 SELECT FACE 2", size_hint=(1, 0.09), background_color=PRIMARY, color=(1,1,1,1), font_size=sp(12))
        btn2.bind(on_press=self.select_face2)
        sidebar.add_widget(btn2)
        
        # Detection toggle
        self.detect_btn = Button(text="🎯 DETECTION ON", size_hint=(1, 0.09), background_color=WARNING, color=(1,1,1,1), font_size=sp(11))
        self.detect_btn.bind(on_press=self.toggle_detection)
        sidebar.add_widget(self.detect_btn)
        
        self.add_widget(sidebar)
    
    def _setup_main(self):
        """Setup main content with cropped face displays"""
        main = BoxLayout(orientation='vertical', padding=dp(12), spacing=dp(12))
        
        # Faces display area
        faces_container = BoxLayout(orientation='horizontal', spacing=dp(15), size_hint=(1, 0.6))
        
        # Face 1 display
        self.face1_display = CroppedFaceDisplay(title="FACE 1")
        faces_container.add_widget(self.face1_display)
        
        # VS label
        vs_label = Label(text="VS", size_hint=(0.12, 1), font_size=sp(22), bold=True, color=PRIMARY)
        faces_container.add_widget(vs_label)
        
        # Face 2 display
        self.face2_display = CroppedFaceDisplay(title="FACE 2")
        faces_container.add_widget(self.face2_display)
        
        main.add_widget(faces_container)
        
        # Result area
        self.result_label = Label(
            text="Select two faces to compare",
            size_hint=(1, 0.15),
            color=TEXT,
            font_size=sp(14),
            bold=True,
            halign='center',
            valign='middle'
        )
        self.result_label.bind(size=self.result_label.setter('text_size'))
        main.add_widget(self.result_label)
        
        self.distance_label = Label(
            text="",
            size_hint=(1, 0.08),
            color=TEXT_DIM,
            font_size=sp(11),
            halign='center'
        )
        self.distance_label.bind(size=self.distance_label.setter('text_size'))
        main.add_widget(self.distance_label)
        
        # Compare button
        compare_btn = Button(
            text="🔍 COMPARE FACES",
            size_hint=(1, 0.1),
            background_color=SUCCESS,
            color=(1,1,1,1),
            font_size=sp(15),
            bold=True
        )
        compare_btn.bind(on_press=self.compare_faces)
        main.add_widget(compare_btn)
        
        self.add_widget(main)
    
    def toggle_detection(self, btn):
        self.detection_enabled = not self.detection_enabled
        if self.detection_enabled:
            btn.text = "🎯 DETECTION ON"
            btn.background_color = WARNING
        else:
            btn.text = "⭕ DETECTION OFF"
            btn.background_color = TEXT_DIM
    
    def select_face1(self, btn):
        if self.filechooser.selection:
            self.img1_path = self.filechooser.selection[0]
            self._crop_and_display_face(1)
            self.result_label.text = "✓ Face 1 selected - cropping..."
            self.result_label.color = SUCCESS
    
    def select_face2(self, btn):
        if self.filechooser.selection:
            self.img2_path = self.filechooser.selection[0]
            self._crop_and_display_face(2)
            self.result_label.text = "✓ Face 2 selected - cropping..."
            self.result_label.color = SUCCESS
    
    def _crop_and_display_face(self, face_num):
        """Crop face and display it"""
        path = self.img1_path if face_num == 1 else self.img2_path
        if not path:
            return
        
        def process(dt):
            try:
                # Detect and crop face
                cropped, msg = face_detector.detect_and_crop(path)
                
                if face_num == 1:
                    self.cropped_face1 = cropped
                    self.face1_display.set_face(cropped, msg)
                    self.result_label.text = "✓ Face 1 cropped successfully"
                else:
                    self.cropped_face2 = cropped
                    self.face2_display.set_face(cropped, msg)
                    self.result_label.text = "✓ Face 2 cropped successfully"
                    
            except Exception as e:
                error_msg = f"Error: {str(e)[:25]}"
                if face_num == 1:
                    self.face1_display.set_face(None, error_msg)
                else:
                    self.face2_display.set_face(None, error_msg)
                self.result_label.text = error_msg
                self.result_label.color = DANGER
        
        Clock.schedule_once(process, 0.1)
    
    def compare_faces(self, btn):
        """Compare the two cropped faces"""
        if not self.img1_path or not self.img2_path:
            self.result_label.text = "⚠ Please select both images first"
            self.result_label.color = WARNING
            self.distance_label.text = ""
            return
        
        # Check if we have cropped faces
        if not self.cropped_face1 or not self.cropped_face2:
            self.result_label.text = "⚠ Please wait for face cropping to complete"
            self.result_label.color = WARNING
            return
        
        # Show loading popup
        popup = LoadingPopup()
        popup.open()
        
        def do_comparison(dt):
            try:
                # Compare the cropped faces
                distance, result = compare_faces_simple(self.cropped_face1, self.cropped_face2)
                popup.dismiss()
                
                # Update UI
                if result == "Same Person":
                    self.result_label.text = "✅ SAME PERSON"
                    self.result_label.color = PRIMARY
                elif result == "Twins":
                    self.result_label.text = "👯 TWINS"
                    self.result_label.color = SUCCESS
                else:
                    self.result_label.text = "❌ DIFFERENT PEOPLE"
                    self.result_label.color = DANGER
                
                self.distance_label.text = f"Similarity distance: {distance:.4f}"
                
            except Exception as e:
                popup.dismiss()
                self.result_label.text = f"Error: {str(e)[:35]}"
                self.result_label.color = DANGER
                self.distance_label.text = ""
        
        Clock.schedule_once(do_comparison, 0.1)

class TwinApp(App):
    def build(self):
        return TwinUI()
    
    def on_start(self):
        print("✓ Application started")
    
    def on_stop(self):
        print("Application closed")

if __name__ == "__main__":
    TwinApp().run()