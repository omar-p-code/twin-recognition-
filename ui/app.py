# ui/app_v1_minimal.py - Modern Minimal Design
import sys
import os
import io
from PIL import Image as PILImage

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.behaviors import ButtonBehavior
from kivy.graphics import Color, RoundedRectangle, Ellipse
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.metrics import dp, sp
from kivy.uix.popup import Popup
from kivy.uix.progressbar import ProgressBar
from kivy.core.image import Image as CoreImage
from kivy.uix.floatlayout import FloatLayout
from kivy.animation import Animation

# Mobile screen size
Window.size = (360, 740)
Window.clearcolor = (0.95, 0.95, 0.97, 1)

# Modern Color Palette
BACKGROUND = (0.95, 0.95, 0.97, 1)
CARD_BG = (1, 1, 1, 1)
PRIMARY = (0.23, 0.44, 0.96, 1)  # Royal Blue
SECONDARY = (0.51, 0.40, 0.93, 1)  # Purple
SUCCESS = (0.18, 0.80, 0.44, 1)  # Green
WARNING = (0.95, 0.61, 0.07, 1)  # Orange
DANGER = (0.93, 0.29, 0.31, 1)  # Red
TEXT_PRIMARY = (0.15, 0.15, 0.2, 1)
TEXT_SECONDARY = (0.55, 0.55, 0.65, 1)
SHADOW = (0, 0, 0, 0.1)

# ================== CUSTOM ICONS ==================
class IconButton(ButtonBehavior, FloatLayout):
    """Custom icon button with emoji"""
    def __init__(self, icon="📁", text="", bg_color=PRIMARY, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)
        self.size = (dp(50), dp(50))
        
        with self.canvas.before:
            Color(*bg_color)
            self.bg = RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(12)])
        
        self.icon_label = Label(
            text=icon,
            font_size=sp(24),
            size_hint=(1, 0.7),
            pos_hint={'center_x': 0.5, 'top': 1}
        )
        self.add_widget(self.icon_label)
        
        if text:
            self.text_label = Label(
                text=text,
                font_size=sp(9),
                color=(1,1,1,1) if bg_color[0] > 0.5 else TEXT_PRIMARY,
                size_hint=(1, 0.3),
                pos_hint={'center_x': 0.5, 'y': 0}
            )
            self.add_widget(self.text_label)
        
        self.bind(pos=self._update_bg, size=self._update_bg)
    
    def _update_bg(self, *args):
        self.bg.pos = self.pos
        self.bg.size = self.size

class FileCard(ButtonBehavior, BoxLayout):
    """Stylish file card"""
    def __init__(self, filepath, callback, **kwargs):
        super().__init__(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(60),
            padding=dp(10),
            spacing=dp(10),
            **kwargs
        )
        self.filepath = filepath
        self.callback = callback
        
        with self.canvas.before:
            Color(*CARD_BG)
            self.bg = RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(10)])
        
        # File icon
        ext = os.path.splitext(filepath)[1].lower()
        icon_map = {
            '.jpg': '🖼️', '.jpeg': '🖼️', '.png': '🖼️',
            '.gif': '🎞️', '.bmp': '🖼️', '.webp': '🖼️'
        }
        icon = icon_map.get(ext, '📄')
        
        self.icon_label = Label(
            text=icon,
            size_hint=(None, 1),
            width=dp(40),
            font_size=sp(24)
        )
        self.add_widget(self.icon_label)
        
        # File info
        info_box = BoxLayout(orientation='vertical', size_hint=(1, 1))
        self.name_label = Label(
            text=os.path.basename(filepath)[:25],
            font_size=sp(12),
            color=TEXT_PRIMARY,
            halign='left',
            valign='middle'
        )
        self.name_label.bind(size=self.name_label.setter('text_size'))
        
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        self.size_label = Label(
            text=f"{size_mb:.1f} MB",
            font_size=sp(10),
            color=TEXT_SECONDARY
        )
        
        info_box.add_widget(self.name_label)
        info_box.add_widget(self.size_label)
        self.add_widget(info_box)
        
        self.bind(pos=self._update_bg, size=self._update_bg)
    
    def _update_bg(self, *args):
        self.bg.pos = self.pos
        self.bg.size = self.size
    
    def on_press(self):
        Animation(scale=0.95, duration=0.1).start(self)
        self.callback(self.filepath)
    
    def on_release(self):
        Animation(scale=1, duration=0.1).start(self)

# ================== FACE DETECTION (Same as before) ==================
class FaceDetector:
    def __init__(self):
        self.has_mtcnn = False
        self.detector = None
        try:
            from mtcnn import MTCNN
            self.detector = MTCNN()
            self.has_mtcnn = True
            print("✓ MTCNN loaded")
        except ImportError:
            print("! Using fallback detector")
    
    def detect_and_crop(self, image_path, padding=30):
        try:
            if isinstance(image_path, str):
                img = PILImage.open(image_path).convert("RGB")
            else:
                img = image_path
            
            original = img.copy()
            
            if self.has_mtcnn and self.detector:
                import numpy as np
                img_np = np.array(img)
                faces = self.detector.detect_faces(img_np)
                
                if faces:
                    best_face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
                    x, y, w, h = best_face['box']
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(original.width - x, w + (2 * padding))
                    h = min(original.height - y, h + (2 * padding))
                    
                    cropped = original.crop((x, y, x + w, y + h))
                    size = max(cropped.width, cropped.height)
                    squared = PILImage.new('RGB', (size, size), (0, 0, 0))
                    x_offset = (size - cropped.width) // 2
                    y_offset = (size - cropped.height) // 2
                    squared.paste(cropped, (x_offset, y_offset))
                    squared.thumbnail((300, 300))
                    return squared, f"Face detected ✓"
            
            size = min(original.width, original.height)
            left = (original.width - size) // 2
            top = (original.height - size) // 2
            cropped = original.crop((left, top, left + size, top + size))
            cropped.thumbnail((300, 300))
            return cropped, "Center crop"
            
        except Exception as e:
            return None, f"Error: {str(e)[:20]}"

face_detector = FaceDetector()

def compare_faces_simple(face1, face2):
    try:
        import numpy as np
        face1_resized = face1.resize((64, 64))
        face2_resized = face2.resize((64, 64))
        arr1 = np.array(face1_resized).astype(np.float32)
        arr2 = np.array(face2_resized).astype(np.float32)
        mse = np.mean((arr1 - arr2) ** 2)
        distance = min(1.0, mse / (255 ** 2))
        
        if distance < 0.35:
            result = "Same Person"
        elif distance < 0.65:
            result = "Twins"
        else:
            result = "Different People"
        return distance, result
    except Exception as e:
        return 0.5, "Error"

# ================== MODERN FACE DISPLAY ==================
class FaceCircle(BoxLayout):
    """Circular face display with modern styling"""
    def __init__(self, label="FACE", **kwargs):
        super().__init__(
            orientation='vertical',
            spacing=dp(8),
            size_hint=(None, None),
            size=(dp(140), dp(180)),
            **kwargs
        )
        
        # Circle container with shadow
        circle_container = FloatLayout(size_hint=(1, 0.75))
        
        # Shadow circle
        with circle_container.canvas.before:
            Color(*SHADOW)
            self.shadow = Ellipse(
                pos=(dp(5), dp(5)),
                size=(dp(130), dp(130))
            )
        
        self.image = Image(
            size_hint=(None, None),
            size=(dp(130), dp(130)),
            pos_hint={'center_x': 0.5, 'center_y': 0.5}
        )
        
        with self.image.canvas.before:
            Color(1, 1, 1, 1)
            self.mask = RoundedRectangle(
                pos=self.image.pos,
                size=self.image.size,
                radius=[dp(65)]
            )
        
        circle_container.add_widget(self.image)
        self.add_widget(circle_container)
        
        # Label with badge style
        self.label = Label(
            text=label,
            size_hint=(1, 0.15),
            font_size=sp(11),
            color=TEXT_SECONDARY,
            bold=True
        )
        self.add_widget(self.label)
        
        self.status_label = Label(
            text="No image",
            size_hint=(1, 0.1),
            font_size=sp(9),
            color=TEXT_SECONDARY
        )
        self.add_widget(self.status_label)
        
        self.bind(pos=self._update_mask, size=self._update_mask)
    
    def _update_mask(self, *args):
        self.mask.pos = self.image.pos
        self.mask.size = self.image.size
    
    def set_face(self, pil_image, status=""):
        if pil_image:
            data = io.BytesIO()
            pil_image.save(data, format='png')
            data.seek(0)
            self.image.texture = CoreImage(data, ext='png').texture
            self.label.color = SUCCESS
            self.status_label.text = status
            self.status_label.color = SUCCESS
            return True
        self.image.texture = None
        self.label.color = DANGER
        self.status_label.text = status
        self.status_label.color = DANGER
        return False

# ================== MAIN APP LAYOUT ==================
class TwinAppModern(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', padding=dp(15), spacing=dp(15), **kwargs)
        
        self.cropped_face1 = None
        self.cropped_face2 = None
        self.selected_face = 1
        
        self._build_header()
        self._build_face_comparison()
        self._build_action_buttons()
        self._build_file_browser()
    
    def _build_header(self):
        """Modern header with gradient effect"""
        header = BoxLayout(
            size_hint=(1, 0.08),
            orientation='horizontal',
            spacing=dp(10)
        )
        
        with header.canvas.before:
            Color(*PRIMARY)
            self.header_bg = RoundedRectangle(
                pos=header.pos,
                size=header.size,
                radius=[0, 0, dp(20), dp(20)]
            )
        header.bind(pos=self._update_header_bg, size=self._update_header_bg)
        
        icon = Label(
            text="👥",
            font_size=sp(28),
            size_hint=(None, 1),
            width=dp(45)
        )
        header.add_widget(icon)
        
        title = Label(
            text="Face Match AI",
            font_size=sp(18),
            color=(1,1,1,1),
            bold=True,
            halign='left',
            valign='middle'
        )
        title.bind(size=title.setter('text_size'))
        header.add_widget(title)
        
        self.add_widget(header)
    
    def _update_header_bg(self, *args):
        self.header_bg.pos = self.header_bg.parent.pos if hasattr(self.header_bg, 'parent') else (0, 0)
        self.header_bg.size = self.header_bg.parent.size if hasattr(self.header_bg, 'parent') else (0, 0)
    
    def _build_face_comparison(self):
        """Modern face comparison display"""
        faces_box = BoxLayout(
            size_hint=(1, 0.35),
            spacing=dp(10),
            padding=[0, dp(10)]
        )
        
        # Face 1 card
        card1 = BoxLayout(
            orientation='vertical',
            size_hint=(0.4, 1),
            padding=dp(10),
            spacing=dp(5)
        )
        with card1.canvas.before:
            Color(*CARD_BG)
            self.card1_bg = RoundedRectangle(pos=card1.pos, size=card1.size, radius=[dp(15)])
        card1.bind(pos=self._update_card_bg1, size=self._update_card_bg1)
        
        self.face1_display = FaceCircle(label="FACE 1")
        card1.add_widget(self.face1_display)
        
        # VS badge
        vs_badge = BoxLayout(
            size_hint=(0.15, 0.3),
            pos_hint={'center_y': 0.5}
        )
        with vs_badge.canvas.before:
            Color(*SECONDARY)
            self.vs_bg = RoundedRectangle(
                pos=vs_badge.pos,
                size=vs_badge.size,
                radius=[dp(20)]
            )
        vs_badge.bind(pos=self._update_vs_bg, size=self._update_vs_bg)
        
        vs_label = Label(
            text="VS",
            font_size=sp(16),
            bold=True,
            color=(1,1,1,1)
        )
        vs_badge.add_widget(vs_label)
        
        # Face 2 card
        card2 = BoxLayout(
            orientation='vertical',
            size_hint=(0.4, 1),
            padding=dp(10),
            spacing=dp(5)
        )
        with card2.canvas.before:
            Color(*CARD_BG)
            self.card2_bg = RoundedRectangle(pos=card2.pos, size=card2.size, radius=[dp(15)])
        card2.bind(pos=self._update_card_bg2, size=self._update_card_bg2)
        
        self.face2_display = FaceCircle(label="FACE 2")
        card2.add_widget(self.face2_display)
        
        faces_box.add_widget(card1)
        faces_box.add_widget(vs_badge)
        faces_box.add_widget(card2)
        
        self.add_widget(faces_box)
    
    def _update_card_bg1(self, *args):
        if hasattr(self, 'card1_bg'):
            self.card1_bg.pos = self.card1_bg.parent.pos
            self.card1_bg.size = self.card1_bg.parent.size
    
    def _update_card_bg2(self, *args):
        if hasattr(self, 'card2_bg'):
            self.card2_bg.pos = self.card2_bg.parent.pos
            self.card2_bg.size = self.card2_bg.parent.size
    
    def _update_vs_bg(self, *args):
        if hasattr(self, 'vs_bg'):
            self.vs_bg.pos = self.vs_bg.parent.pos
            self.vs_bg.size = self.vs_bg.parent.size
    
    def _build_action_buttons(self):
        """Action buttons with modern design"""
        actions = BoxLayout(
            size_hint=(1, 0.15),
            spacing=dp(12)
        )
        
        # Compare button
        compare_btn = Button(
            text="🔍 COMPARE FACES",
            size_hint=(1, 0.7),
            background_color=SUCCESS,
            color=(1,1,1,1),
            font_size=sp(15),
            bold=True,
            pos_hint={'center_y': 0.5}
        )
        with compare_btn.canvas.before:
            Color(*SUCCESS)
            self.compare_bg = RoundedRectangle(
                pos=compare_btn.pos,
                size=compare_btn.size,
                radius=[dp(25)]
            )
        compare_btn.bind(
            pos=lambda i, v: setattr(self.compare_bg, 'pos', v),
            size=lambda i, v: setattr(self.compare_bg, 'size', v),
            on_press=self.compare_faces
        )
        compare_btn.background_color = (0,0,0,0)
        
        # Result display
        self.result_label = Label(
            text="Select faces to compare",
            size_hint=(1, 0.3),
            font_size=sp(12),
            color=TEXT_SECONDARY,
            halign='center',
            valign='middle'
        )
        self.result_label.bind(size=self.result_label.setter('text_size'))
        
        actions.add_widget(compare_btn)
        self.add_widget(actions)
        self.add_widget(self.result_label)
    
    def _build_file_browser(self):
        """Floating file browser"""
        file_box = BoxLayout(
            orientation='vertical',
            size_hint=(1, 0.35),
            spacing=dp(8)
        )
        
        # Title bar
        title_bar = BoxLayout(
            size_hint=(1, 0.08),
            orientation='horizontal',
            spacing=dp(10)
        )
        
        title_bar.add_widget(Label(
            text="📁 Gallery",
            font_size=sp(13),
            color=TEXT_PRIMARY,
            bold=True,
            size_hint=(0.6, 1)
        ))
        
        # Select buttons
        btn1 = Button(
            text="Face 1",
            size_hint=(0.2, 1),
            background_color=PRIMARY,
            color=(1,1,1,1),
            font_size=sp(11),
            bold=True
        )
        btn1.bind(on_press=lambda x: setattr(self, 'selected_face', 1))
        
        btn2 = Button(
            text="Face 2",
            size_hint=(0.2, 1),
            background_color=SECONDARY,
            color=(1,1,1,1),
            font_size=sp(11),
            bold=True
        )
        btn2.bind(on_press=lambda x: setattr(self, 'selected_face', 2))
        
        title_bar.add_widget(btn1)
        title_bar.add_widget(btn2)
        
        file_box.add_widget(title_bar)
        
        # File list
        self.file_list = GridLayout(
            cols=1,
            spacing=dp(5),
            size_hint_y=None
        )
        self.file_list.bind(minimum_height=self.file_list.setter('height'))
        
        scroll = ScrollView(
            size_hint=(1, 0.92),
            bar_width=dp(4),
            scroll_type=['bars', 'content']
        )
        scroll.add_widget(self.file_list)
        
        file_box.add_widget(scroll)
        self.add_widget(file_box)
        
        # Load files
        self._load_files()
    
    def _load_files(self):
        """Load image files"""
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')
        test_dirs = [
            os.path.expanduser("~/Pictures"),
            os.path.expanduser("~/Downloads"),
            os.path.expanduser("~/Desktop"),
            os.path.dirname(os.path.abspath(__file__)),
            os.getcwd()
        ]
        
        for directory in test_dirs:
            if os.path.exists(directory):
                for file in sorted(os.listdir(directory)):
                    if file.lower().endswith(image_extensions):
                        filepath = os.path.join(directory, file)
                        card = FileCard(filepath, self._on_file_selected)
                        self.file_list.add_widget(card)
        
        if len(self.file_list.children) == 0:
            self.file_list.add_widget(Label(
                text="No images found",
                font_size=sp(12),
                color=TEXT_SECONDARY
            ))
    
    def _on_file_selected(self, filepath):
        """Handle file selection"""
        def process(dt):
            cropped, status = face_detector.detect_and_crop(filepath)
            
            if self.selected_face == 1:
                self.cropped_face1 = cropped
                self.face1_display.set_face(cropped, status)
                self.result_label.text = f"Face 1: {os.path.basename(filepath)}"
            else:
                self.cropped_face2 = cropped
                self.face2_display.set_face(cropped, status)
                self.result_label.text = f"Face 2: {os.path.basename(filepath)}"
            
            self.result_label.color = SUCCESS
        
        Clock.schedule_once(process, 0.05)
    
    def compare_faces(self, *args):
        """Compare faces with animation"""
        if not self.cropped_face1 or not self.cropped_face2:
            self.result_label.text = "⚠️ Please select both faces first"
            self.result_label.color = WARNING
            return
        
        self.result_label.text = "🔄 Analyzing..."
        self.result_label.color = PRIMARY
        
        def analyze(dt):
            distance, result = compare_faces_simple(self.cropped_face1, self.cropped_face2)
            
            if result == "Same Person":
                self.result_label.text = f"✅ SAME PERSON ({distance:.3f})"
                self.result_label.color = SUCCESS
            elif result == "Twins":
                self.result_label.text = f"👯 POSSIBLE TWINS ({distance:.3f})"
                self.result_label.color = SECONDARY
            else:
                self.result_label.text = f"❌ DIFFERENT PEOPLE ({distance:.3f})"
                self.result_label.color = DANGER
            
            # Pulse animation
            anim = Animation(scale=1.05, duration=0.2) + Animation(scale=1, duration=0.2)
            anim.start(self.result_label)
        
        Clock.schedule_once(analyze, 0.5)

class TwinAppModern(App):
    def build(self):
        return TwinAppModern()
    
    def on_start(self):
        print("✓ Modern Twin App Started")

if __name__ == "__main__":
    TwinAppModern().run()