# ui/app.py - Mobile & Desktop Face Comparison App
import sys
import os
import io
# from PIL import Image as PILImage

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.image import Image
from kivy.graphics import Color, RoundedRectangle
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.metrics import dp, sp
from kivy.uix.popup import Popup
from kivy.uix.progressbar import ProgressBar
from kivy.core.image import Image as CoreImage
from kivy.uix.floatlayout import FloatLayout
from kivy.utils import platform


checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints', 'checkpoint.pth')
if not os.path.exists(checkpoint_path):
    print(f"❌ Model checkpoint not found at: {checkpoint_path}")
    print("Please train the model first using: python train.py")
    exit(1)

from inference.predict import load_model
model, TH_SAME, TH_TWIN = load_model(checkpoint_path, device='cpu')


TEXT_DIM = (0.6, 0.6, 0.65, 1)      # Light grey# Try to import plyer for native file picker
try:
    from plyer import filechooser as plyer_filechooser
    HAS_PLYER = True
except ImportError:
    HAS_PLYER = False
    print("! Plyer not installed - using Kivy file picker")

# Android permissions
if platform == 'android':
    try:
        from android.permissions import request_permissions, Permission
        request_permissions([
            Permission.READ_EXTERNAL_STORAGE,
            Permission.WRITE_EXTERNAL_STORAGE,
            Permission.READ_MEDIA_IMAGES
        ])
    except:
        pass

# Mobile screen size
Window.size = (360, 640)
Window.clearcolor = (0.07, 0.07, 0.1, 1)

# Dark Mode - Black & Gold
PRIMARY = (0.9, 0.75, 0.2, 1)       # Gold
SUCCESS = (0.2, 0.8, 0.55, 1)       # Emerald Green
WARNING = (1.0, 0.65, 0.2, 1)       # Amber
DANGER = (0.9, 0.3, 0.35, 1)        # Coral Red
TEXT = (0.95, 0.95, 0.9, 1)         # Off White
TEXT_DIM = (0.7, 0.7, 0.65, 1)      # Dim Grey


# ================== LAST DIRECTORY MEMORY ==================
def get_last_dir_file():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'last_dir.txt')

def save_last_directory(path):
    try:
        directory = os.path.dirname(path) if os.path.isfile(path) else path
        with open(get_last_dir_file(), 'w') as f:
            f.write(directory)
    except:
        pass

def load_last_directory():
    try:
        last_dir_file = get_last_dir_file()
        if os.path.exists(last_dir_file):
            with open(last_dir_file, 'r') as f:
                directory = f.read().strip()
                if os.path.exists(directory):
                    return directory
    except:
        pass
    return os.path.expanduser('~')

# ================== FACE DETECTION ==================
from utils.face_detection import FaceDetector

face_detector = FaceDetector()

from models.siamese import compare_faces

# ================== FILE PICKER ==================
def open_file_picker(callback):
    current_dir = load_last_directory()
    
    if HAS_PLYER and platform in ['android', 'ios']:
        try:
            plyer_filechooser.open_file(
                on_selection=lambda selection: _on_picked(selection, callback),
                filters=['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.webp'],
                path=current_dir
            )
            return
        except Exception as e:
            print(f"Native picker failed: {e}, using Kivy picker")
    
    _open_kivy_picker(callback, current_dir)

def _on_picked(selection, callback):
    if selection:
        filepath = selection[0]
        save_last_directory(filepath)
        callback(filepath)

def _open_kivy_picker(callback, start_dir):
    content = BoxLayout(orientation='vertical', spacing=dp(8), padding=dp(8))
    
    fc = FileChooserListView(
        path=start_dir,
        filters=['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG', '*.gif', '*.bmp', '*.webp'],
        size_hint=(1, 0.85)
    )
    content.add_widget(fc)
    
    btn_box = BoxLayout(size_hint=(1, 0.15), spacing=dp(10))
    
    cancel = Button(text="CANCEL", background_color=DANGER, color=(1,1,1,1), font_size=sp(14), bold=True)
    btn_box.add_widget(cancel)
    
    popup = Popup(title="Select Image", content=content, size_hint=(0.92, 0.92), auto_dismiss=False)
    
    def on_select(instance):
        if fc.selection:
            filepath = fc.selection[0]
            save_last_directory(filepath)
            callback(filepath)
            popup.dismiss()
    
    select = Button(text="SELECT", background_color=SUCCESS, color=(1,1,1,1), font_size=sp(14), bold=True)
    select.bind(on_press=on_select)
    btn_box.add_widget(select)
    
    cancel.bind(on_press=popup.dismiss)
    
    content.add_widget(btn_box)
    popup.open()

# ================== UI COMPONENTS ==================
class CircularImage(Image):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.allow_stretch = True
        self.keep_ratio = True
        self.bind(size=self._update_rounding, pos=self._update_rounding)
        self.bind(texture=self._update_rounding)
        
    def _update_rounding(self, *args):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(0.07, 0.07, 0.1, 1)
            if self.width > 0 and self.height > 0:
                size = min(self.width, self.height)
                radius = size / 2
                x = self.center_x - radius
                y = self.center_y - radius
                RoundedRectangle(
                    pos=(x, y),
                    size=(size, size),
                    radius=[radius]
                )

class CroppedFaceDisplay(BoxLayout):
    def __init__(self, title="Face", **kwargs):
        super().__init__(orientation='vertical', spacing=dp(5), **kwargs)
        
        self.image_container = FloatLayout(size_hint=(1, 0.78))
        
        self.image = CircularImage(
            size_hint=(None, None),
            pos_hint={'center_x': 0.5, 'center_y': 0.5}
        )
        self.image_container.bind(size=self._update_image_size)
        
        self.image_container.add_widget(self.image)
        self.add_widget(self.image_container)
        
        self.title_label = Label(
            text=title, 
            size_hint=(1, 0.1), 
            color=TEXT_DIM, 
            font_size=sp(12), 
            bold=True,
            halign='center'
        )
        self.title_label.bind(size=self.title_label.setter('text_size'))
        self.add_widget(self.title_label)
        
        self.status_label = Label(
            text="No image", 
            size_hint=(1, 0.12), 
            color=TEXT_DIM, 
            font_size=sp(9),
            halign='center'
        )
        self.status_label.bind(size=self.status_label.setter('text_size'))
        self.add_widget(self.status_label)
    
    def _update_image_size(self, instance, value):
        size = min(value[0], value[1]) * 0.9
        self.image.size = (size, size)
    
    def set_face(self, pil_image, status_text):
        if pil_image:
            data = io.BytesIO()
            pil_image.save(data, format='png')
            data.seek(0)
            self.image.texture = CoreImage(data, ext='png').texture
            self.image._update_rounding()
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
        super().__init__(orientation='vertical', padding=dp(12), spacing=dp(10), **kwargs)
        
        self.cropped_face1 = None
        self.cropped_face2 = None
        
        # Title
        title = Label(
            text="👥 TWIN MATCH",
            font_size=sp(20),
            bold=True,
            color=PRIMARY,
            size_hint=(1, 0.06),
            halign='center',
            valign='middle'
        )
        title.bind(size=title.setter('text_size'))
        self.add_widget(title)
        
        # Faces display
        faces_container = BoxLayout(orientation='horizontal', spacing=dp(12), size_hint=(1, 0.5))
        
        self.face1_display = CroppedFaceDisplay(title="FACE 1")
        faces_container.add_widget(self.face1_display)
        
        vs_label = Label(
            text="VS",
            size_hint=(0.1, 1),
            font_size=sp(20),
            bold=True,
            color=PRIMARY,
            halign='center',
            valign='middle'
        )
        faces_container.add_widget(vs_label)
        
        self.face2_display = CroppedFaceDisplay(title="FACE 2")
        faces_container.add_widget(self.face2_display)
        
        self.add_widget(faces_container)
        
        # Result area
        self.result_label = Label(
            text="Select two faces to compare",
            size_hint=(1, 0.08),
            color=TEXT,
            font_size=sp(13),
            bold=True,
            halign='center',
            valign='middle'
        )
        self.result_label.bind(size=self.result_label.setter('text_size'))
        self.add_widget(self.result_label)
        
        self.distance_label = Label(
            text="",
            size_hint=(1, 0.05),
            color=TEXT_DIM,
            font_size=sp(10),
            halign='center'
        )
        self.distance_label.bind(size=self.distance_label.setter('text_size'))
        self.add_widget(self.distance_label)
        
        # Buttons area
        buttons_area = BoxLayout(
            orientation='vertical',
            size_hint=(1, 0.28),
            spacing=dp(8),
            padding=[dp(5), dp(5)]
        )
        
        select_row = BoxLayout(size_hint=(1, 0.3), spacing=dp(10))
        
        btn_face1 = Button(
            text="📷 SELECT FACE 1",
            size_hint=(0.5, 1),
            background_color=PRIMARY,
            color=(1,1,1,1),
            font_size=sp(13),
            bold=True
        )
        btn_face1.bind(on_press=lambda x: self._open_picker_for_face(1))
        select_row.add_widget(btn_face1)
        
        btn_face2 = Button(
            text="📸 SELECT FACE 2",
            size_hint=(0.5, 1),
            background_color=PRIMARY,
            color=(1,1,1,1),
            font_size=sp(13),
            bold=True
        )
        btn_face2.bind(on_press=lambda x: self._open_picker_for_face(2))
        select_row.add_widget(btn_face2)
        
        buttons_area.add_widget(select_row)
        
        compare_btn = Button(
            text="🔍 COMPARE FACES",
            size_hint=(1, 0.35),
            background_color=SUCCESS,
            color=(1,1,1,1),
            font_size=sp(15),
            bold=True
        )
        compare_btn.bind(on_press=self.compare_faces)
        buttons_area.add_widget(compare_btn)
        
        info_label = Label(
            text="Tap SELECT FACE to browse images\nThen tap COMPARE to analyze",
            size_hint=(1, 0.35),
            color=TEXT_DIM,
            font_size=sp(10),
            halign='center',
            valign='middle'
        )
        info_label.bind(size=info_label.setter('text_size'))
        buttons_area.add_widget(info_label)
        
        self.add_widget(buttons_area)
        
        print("✓ UI ready")
    
    def _open_picker_for_face(self, face_num):
        def on_file_selected(filepath):
            def process(dt):
                cropped, msg = face_detector.detect_and_crop(filepath)
                
                if face_num == 1:
                    self.cropped_face1 = cropped
                    self.face1_display.set_face(cropped, msg)
                    self.result_label.text = "✓ Face 1 selected"
                else:
                    self.cropped_face2 = cropped
                    self.face2_display.set_face(cropped, msg)
                    self.result_label.text = "✓ Face 2 selected"
                
                self.result_label.color = SUCCESS
            
            Clock.schedule_once(process, 0.05)
        
        open_file_picker(on_file_selected)
    
    def compare_faces(self, btn):
        if not self.cropped_face1 or not self.cropped_face2:
            self.result_label.text = "⚠ Select both faces first"
            self.result_label.color = WARNING
            self.distance_label.text = ""
            return
        
        popup = LoadingPopup()
        popup.open()
        
        def do_comparison(dt):
            try:
                distance, result, detection_info = compare_faces(model ,self.cropped_face1, self.cropped_face2, TH_SAME, TH_TWIN, device='cpu', detect_faces=True)
                popup.dismiss()
                print(f"Comparison result: {result} (distance: {distance:.4f})")
                print(f"Detection info: {detection_info}")
                print(f"Thresholds: same={TH_SAME}, twin={TH_TWIN}")
                
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

if __name__ == "__main__":
    TwinApp().run()