# ui/app_modern.py
# Modern Light Theme + Different Layout Version

import sys
import os
import io

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
# from kivy.uix.floatlayout import FloatLayout
# from kivy.uix.scrollview import ScrollView
from kivy.utils import platform

checkpoint_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'checkpoints',
    'checkpoint.pth'
)

if not os.path.exists(checkpoint_path):
    print("Checkpoint not found")
    exit(1)

from inference.predict import load_model
from utils.face_detection import FaceDetector
from models.siamese import compare_faces

model, TH_SAME, TH_TWIN = load_model(checkpoint_path, device='cpu')

# Try to import plyer for native file picker
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


face_detector = FaceDetector()

# ================= WINDOW =================
Window.size = (360, 640)
Window.clearcolor = (0.07, 0.07, 0.1, 1)

# ================= COLORS =================
BG = (0.96, 0.97, 0.99, 1)

PRIMARY = (0.20, 0.45, 0.95, 1)
SUCCESS = (0.18, 0.72, 0.45, 1)
WARNING = (1.0, 0.65, 0.1, 1)
DANGER = (0.90, 0.25, 0.25, 1)

TEXT = (0.12, 0.12, 0.15, 1)
TEXT_LIGHT = (0.45, 0.48, 0.55, 1)

CARD = (1, 1, 1, 1)

# ================= FILE PICKER =================
def open_picker(callback):

    content = BoxLayout(
        orientation='vertical',
        spacing=dp(10),
        padding=dp(10)
    )

    fc = FileChooserListView(
        filters=['*.jpg', '*.jpeg', '*.png', '*.webp'],
        size_hint=(1, 0.88)
    )

    content.add_widget(fc)

    btns = BoxLayout(
        size_hint=(1, 0.12),
        spacing=dp(10)
    )

    popup = Popup(
        title="Select Image",
        content=content,
        size_hint=(0.93, 0.93),
        auto_dismiss=False
    )

    cancel = Button(
        text="Cancel",
        background_color=DANGER,
        bold=True
    )

    def select_image(instance):
        if fc.selection:
            callback(fc.selection[0])
            popup.dismiss()

    select = Button(
        text="Select",
        background_color=SUCCESS,
        bold=True
    )

    select.bind(on_press=select_image)
    cancel.bind(on_press=popup.dismiss)

    btns.add_widget(cancel)
    btns.add_widget(select)

    content.add_widget(btns)

    popup.open()

# ================= FACE CARD =================
class FaceCard(BoxLayout):

    def __init__(self, title="Face", **kwargs):
        super().__init__(
            orientation='vertical',
            spacing=dp(10),
            padding=dp(10),
            **kwargs
        )

        with self.canvas.before:
            Color(*CARD)
            self.bg = RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[20]
            )

        self.bind(pos=self.update_bg, size=self.update_bg)

        self.preview = Image(
            allow_stretch=True,
            keep_ratio=True
        )

        self.title_label = Label(
            text=title,
            color=TEXT,
            bold=True,
            font_size=sp(14),
            size_hint=(1, 0.12)
        )

        self.status = Label(
            text="No image selected",
            color=TEXT_LIGHT,
            font_size=sp(10),
            size_hint=(1, 0.12)
        )

        self.add_widget(self.title_label)
        self.add_widget(self.preview)
        self.add_widget(self.status)

    def update_bg(self, *args):
        self.bg.pos = self.pos
        self.bg.size = self.size

    def set_face(self, img, status_text):

        if img is None:
            self.preview.texture = None
            self.status.text = status_text
            self.status.color = DANGER
            return

        data = io.BytesIO()
        img.save(data, format='png')
        data.seek(0)

        self.preview.texture = CoreImage(
            data,
            ext='png'
        ).texture

        self.status.text = status_text
        self.status.color = SUCCESS

# ================= LOADING =================
class LoadingPopup(Popup):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.title = "Analyzing"
        self.size_hint = (0.7, 0.25)
        self.auto_dismiss = False

        layout = BoxLayout(
            orientation='vertical',
            padding=dp(15),
            spacing=dp(15)
        )

        msg = Label(
            text="Comparing faces...",
            color=TEXT,
            font_size=sp(14)
        )

        bar = ProgressBar(
            value=50,
            max=100
        )

        layout.add_widget(msg)
        layout.add_widget(bar)

        self.content = layout

# ================= MAIN UI =================
class TwinUI(BoxLayout):

    def __init__(self, **kwargs):
        super().__init__(
            orientation='vertical',
            spacing=dp(15),
            padding=dp(15),
            **kwargs
        )

        self.face1 = None
        self.face2 = None

        title = Label(
            text="Twin Recognition AI",
            color=PRIMARY,
            bold=True,
            font_size=sp(24),
            size_hint=(1, 0.08)
        )

        self.add_widget(title)

        subtitle = Label(
            text="Compare two faces using Siamese Neural Networks",
            color=TEXT_LIGHT,
            font_size=sp(11),
            size_hint=(1, 0.04)
        )

        self.add_widget(subtitle)

        # ================= CARDS =================
        cards = BoxLayout(
            spacing=dp(15),
            size_hint=(1, 0.5)
        )

        self.card1 = FaceCard(title="FACE 1")
        self.card2 = FaceCard(title="FACE 2")

        cards.add_widget(self.card1)
        cards.add_widget(self.card2)

        self.add_widget(cards)

        # ================= RESULT CARD =================
        result_box = BoxLayout(
            orientation='vertical',
            spacing=dp(5),
            padding=dp(15),
            size_hint=(1, 0.18)
        )

        with result_box.canvas.before:
            Color(*CARD)
            result_box.bg = RoundedRectangle(
                pos=result_box.pos,
                size=result_box.size,
                radius=[20]
            )

        result_box.bind(
            pos=lambda *_: self.update_rect(result_box),
            size=lambda *_: self.update_rect(result_box)
        )

        self.result = Label(
            text="Select two images",
            color=TEXT,
            font_size=sp(18),
            bold=True
        )

        self.distance = Label(
            text="",
            color=TEXT_LIGHT,
            font_size=sp(12)
        )

        result_box.add_widget(self.result)
        result_box.add_widget(self.distance)

        self.add_widget(result_box)

        # ================= BUTTONS =================
        btns = BoxLayout(
            orientation='vertical',
            spacing=dp(10),
            size_hint=(1, 0.2)
        )

        row = BoxLayout(
            spacing=dp(10)
        )

        btn1 = Button(
            text="Select Face 1",
            background_color=PRIMARY,
            bold=True
        )

        btn2 = Button(
            text="Select Face 2",
            background_color=PRIMARY,
            bold=True
        )

        btn1.bind(on_press=lambda x: self.pick_face(1))
        btn2.bind(on_press=lambda x: self.pick_face(2))

        row.add_widget(btn1)
        row.add_widget(btn2)

        compare_btn = Button(
            text="COMPARE",
            background_color=SUCCESS,
            bold=True,
            font_size=sp(16)
        )

        compare_btn.bind(on_press=self.run_compare)

        btns.add_widget(row)
        btns.add_widget(compare_btn)

        self.add_widget(btns)

    def update_rect(self, widget):
        widget.bg.pos = widget.pos
        widget.bg.size = widget.size

    def pick_face(self, idx):

        def selected(path):

            def process(dt):

                face, msg = face_detector.detect_and_crop(path)

                if idx == 1:
                    self.face1 = face
                    self.card1.set_face(face, msg)

                else:
                    self.face2 = face
                    self.card2.set_face(face, msg)

            Clock.schedule_once(process, 0.05)

        open_picker(selected)

    def run_compare(self, instance):

        if self.face1 is None or self.face2 is None:
            self.result.text = "Select both faces first"
            self.result.color = WARNING
            return

        popup = LoadingPopup()
        popup.open()

        def compare(dt):

            try:

                distance, result, info = compare_faces(
                    model,
                    self.face1,
                    self.face2,
                    TH_SAME,
                    TH_TWIN,
                    device='cpu',
                    detect_faces=False
                )

                popup.dismiss()

                if result == "Same Person":
                    self.result.text = "SAME PERSON"
                    self.result.color = PRIMARY

                elif result == "Twins":
                    self.result.text = "TWINS"
                    self.result.color = SUCCESS

                else:
                    self.result.text = "DIFFERENT PEOPLE"
                    self.result.color = DANGER

                self.distance.text = (
                    f"Distance: {distance:.4f} | "
                    f"same<{TH_SAME:.2f} | twin<{TH_TWIN:.2f}"
                )

            except Exception as e:

                popup.dismiss()

                self.result.text = f"Error: {str(e)}"
                self.result.color = DANGER

        Clock.schedule_once(compare, 0.1)

# ================= APP =================
class TwinApp(App):

    def build(self):
        return TwinUI()

if __name__ == "__main__":
    TwinApp().run()