import sys
import os

sys.path.append(
      os.path.dirname(
            os.path.dirname(
                  os.path.abspath(__file__)
            )
      )
      )

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.image import Image

from models.siamese import compare_faces
from config import DEVICE
from inference.predict import load_model


class TwinUI(BoxLayout):
      def __init__(self, **kwargs):
            super().__init__(orientation="horizontal", **kwargs)

            # ================== Load model ==================
            checkpoint_path = os.path.join(
                  os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "checkpoints",
                  "siamese_best.pth"
            )
            self.model, self.th_same, self.th_twin = load_model(checkpoint_path, DEVICE)

            self.img1 = None
            self.img2 = None

            # ================== LEFT: File chooser ==================
            left_panel = BoxLayout(orientation="vertical", size_hint=(0.3, 1))

            self.filechooser = FileChooserIconView()
            left_panel.add_widget(self.filechooser)

            btn1 = Button(text="Select First Image", size_hint=(1, 0.1))
            btn1.bind(on_press=self.select_first)
            left_panel.add_widget(btn1)

            btn2 = Button(text="Select Second Image", size_hint=(1, 0.1))
            btn2.bind(on_press=self.select_second)
            left_panel.add_widget(btn2)

            self.add_widget(left_panel)

            # ================== RIGHT: Images + Result ==================
            right_panel = BoxLayout(orientation="vertical", size_hint=(0.7, 1))

            # Images row
            images_layout = BoxLayout(orientation="horizontal", size_hint=(1, 0.7))

            self.image1_widget = Image()
            self.image2_widget = Image()

            vs_label = Label(text="VS", size_hint=(0.2, 1))

            images_layout.add_widget(self.image1_widget)
            images_layout.add_widget(vs_label)
            images_layout.add_widget(self.image2_widget)

            right_panel.add_widget(images_layout)

            # Result label
            self.result = Label(
                  text="Select two images",
                  size_hint=(1, 0.2)
            )
            right_panel.add_widget(self.result)

            # Compare button
            compare_btn = Button(
                  text="Compare Faces",
                  size_hint=(1, 0.1)
            )
            compare_btn.bind(on_press=self.compare)
            right_panel.add_widget(compare_btn)

            self.add_widget(right_panel)

      def select_first(self, instance):
            if self.filechooser.selection:
                  self.img1 = self.filechooser.selection[0]
                  self.image1_widget.source = self.img1
                  self.result.text = "First image selected"

      def select_second(self, instance):
            if self.filechooser.selection:
                  self.img2 = self.filechooser.selection[0]
                  self.image2_widget.source = self.img2
                  self.result.text = "Second image selected"

      def compare(self, instance):
            if self.img1 and self.img2:
                  dist, result = compare_faces(
                        self.model,   
                        self.img1,
                        self.img2,
                        device=DEVICE,
                        th_same=self.th_same,
                        th_twin=self.th_twin
                        )

                  self.result.text = (
                  f"Distance: {dist:.4f}\n"
                  f"Result: {result}"
                  )
            else:
                  self.result.text = "Please select both images first"


class TwinApp(App):
      def build(self):
            return TwinUI()


if __name__ == "__main__":
      TwinApp().run()