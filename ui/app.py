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

from models.siamese import compare_faces
from config import DEVICE
from inference.predict import load_model


class TwinUI(BoxLayout):
   def __init__(self, **kwargs):
      super().__init__(
            orientation="vertical",
            spacing=10,
            padding=10,
            **kwargs
      )

      checkpoint_path = os.path.join(
            os.path.dirname(
               os.path.dirname(
                  os.path.abspath(__file__)
               )
            ),
            "checkpoints",
            "siamese_best.pth"
      )

      self.model = load_model(checkpoint_path)

      self.img1 = None
      self.img2 = None

      self.result = Label(
            text="Select two images",
            size_hint=(1, 0.1)
      )
      self.add_widget(self.result)

      self.filechooser = FileChooserIconView(
            size_hint=(1, 0.6)
      )
      self.add_widget(self.filechooser)

      btn1 = Button(
            text="Select First Image",
            size_hint=(1, 0.1),
            on_press=self.select_first
      )
      self.add_widget(btn1)

      btn2 = Button(
            text="Select Second Image",
            size_hint=(1, 0.1),
            on_press=self.select_second
      )
      self.add_widget(btn2)

      compare_btn = Button(
            text="Compare Faces / Twins",
            size_hint=(1, 0.1),
            on_press=self.compare
      )
      self.add_widget(compare_btn)

   def select_first(self, instance):
      if self.filechooser.selection:
            self.img1 = self.filechooser.selection[0]
            self.result.text = "First image selected"

   def select_second(self, instance):
      if self.filechooser.selection:
            self.img2 = self.filechooser.selection[0]
            self.result.text = "Second image selected"

   def compare(self, instance):
      if self.img1 and self.img2:
            dist, result = compare_faces(
               self.model,
               self.img1,
               self.img2,
               device=DEVICE
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