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
from kivy.graphics import Color, RoundedRectangle, Rectangle
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.animation import Animation
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.scrollview import ScrollView
from kivy.metrics import dp, sp

from models.siamese import compare_faces
from config import DEVICE
from inference.predict import load_model

# Set window size for mobile simulation
Window.size = (360, 640)
Window.clearcolor = (0.95, 0.95, 0.95, 1)

class NeonButton(ButtonBehavior, FloatLayout):
    def __init__(self, text='', **kwargs):
        super().__init__(size_hint=(None, None), size=(dp(130), dp(48)), **kwargs)
        self.text = text
        
        with self.canvas:
            # Gradient effect
            Color(0.2, 0.8, 0.9, 1)
            self.bg = RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(24)])
            Color(0.5, 0.9, 1, 0.3)
            self.glow = RoundedRectangle(pos=(self.x-2, self.y-2), size=(self.width+4, self.height+4), radius=[dp(26)])
            
        self.label = Label(text=text, color=(1,1,1,1), font_size=sp(15), bold=True)
        self.add_widget(self.label)
        
        self.bind(pos=self.update_graphics, size=self.update_graphics)
        
    def update_graphics(self, *args):
        self.bg.pos = self.pos
        self.bg.size = self.size
        self.glow.pos = (self.x-2, self.y-2)
        self.glow.size = (self.width+4, self.height+4)
        self.label.center = self.center
        
    def on_press(self):
        anim = Animation(size=(dp(125), dp(44)), duration=0.05)
        anim.start(self)
        
    def on_release(self):
        anim = Animation(size=(dp(130), dp(48)), duration=0.05)
        anim.start(self)

class GradientBackground(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas:
            Color(0.05, 0.05, 0.15, 1)
            self.rect = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self.update_rect, size=self.update_rect)
        
    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size

class NeonImage(Image):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(size=self._add_border, pos=self._add_border)
        
    def _add_border(self, *args):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(0.2, 0.8, 0.9, 1)
            RoundedRectangle(pos=(self.x-3, self.y-3), size=(self.width+6, self.height+6), radius=[self.width/2 + 3])
            Color(0.05, 0.05, 0.15, 1)
            RoundedRectangle(pos=self.pos, size=self.size, radius=[self.width/2])

class TwinUIDesign2(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='horizontal', **kwargs)
        
        # Load model
        checkpoint_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "checkpoints",
            "siamese_best.pth"
        )
        self.model, self.th_same, self.th_twin = load_model(checkpoint_path, DEVICE)
        
        self.img1 = None
        self.img2 = None
        
        # Dark sidebar
        sidebar = BoxLayout(orientation='vertical', size_hint=(0.35, 1), spacing=dp(15), padding=dp(15))
        sidebar.canvas.before.clear()
        with sidebar.canvas.before:
            Color(0.02, 0.02, 0.08, 0.95)
            RoundedRectangle(pos=sidebar.pos, size=sidebar.size, radius=[dp(0), dp(25), dp(25), dp(0)])
            
        # Title in sidebar
        title_label = Label(
            text="FACE\nCOMPARE", 
            font_size=sp(22), 
            bold=True, 
            color=(0.2, 0.8, 0.9, 1),
            size_hint=(1, 0.1)
        )
        sidebar.add_widget(title_label)
        
        self.filechooser = FileChooserIconView(
            size_hint=(1, 0.6),
            path=os.path.expanduser('~'),
            filters=['*.jpg', '*.png', '*.jpeg']
        )
        sidebar.add_widget(self.filechooser)
        
        btn1 = NeonButton(text="⚡ SELECT FACE 1")
        btn1.bind(on_press=self.select_first)
        sidebar.add_widget(btn1)
        
        btn2 = NeonButton(text="⚡ SELECT FACE 2")
        btn2.bind(on_press=self.select_second)
        sidebar.add_widget(btn2)
        
        self.add_widget(sidebar)
        
        # Main content
        main_content = BoxLayout(orientation='vertical', padding=dp(20), spacing=dp(20))
        
        # Image display area with neon frames
        images_layout = BoxLayout(orientation='horizontal', spacing=dp(25), size_hint=(1, 0.5))
        
        self.image1_widget = NeonImage()
        self.image2_widget = NeonImage()
        
        vs_badge = BoxLayout(orientation='vertical')
        vs_text = Label(
            text="VS", 
            font_size=sp(28), 
            bold=True, 
            color=(0.2, 0.8, 0.9, 1),
            font_name='Roboto'
        )
        vs_badge.add_widget(vs_text)
        
        images_layout.add_widget(self.image1_widget)
        images_layout.add_widget(vs_badge)
        images_layout.add_widget(self.image2_widget)
        
        main_content.add_widget(images_layout)
        
        # Stats card
        stats_card = BoxLayout(orientation='vertical', size_hint=(1, 0.25), padding=dp(15), spacing=dp(10))
        stats_card.canvas.before.clear()
        with stats_card.canvas.before:
            Color(0.05, 0.05, 0.12, 0.9)
            RoundedRectangle(pos=stats_card.pos, size=stats_card.size, radius=[dp(15)])
            Color(0.2, 0.8, 0.9, 0.2)
            RoundedRectangle(pos=(stats_card.x-1, stats_card.y-1), size=(stats_card.width+2, stats_card.height+2), radius=[dp(16)])
        
        self.result = Label(
            text="READY TO COMPARE",
            color=(0.2, 0.8, 0.9, 1),
            font_size=sp(16),
            bold=True,
            halign='center',
            valign='middle'
        )
        self.result.bind(size=self.result.setter('text_size'))
        stats_card.add_widget(self.result)
        
        self.distance_label = Label(
            text="",
            color=(0.5, 0.5, 0.5, 1),
            font_size=sp(12),
            halign='center'
        )
        self.distance_label.bind(size=self.distance_label.setter('text_size'))
        stats_card.add_widget(self.distance_label)
        
        main_content.add_widget(stats_card)
        
        # Action buttons
        buttons_layout = BoxLayout(orientation='horizontal', spacing=dp(10), size_hint=(1, 0.1))
        
        reset_btn = NeonButton(text="⟳ RESET")
        reset_btn.bind(on_press=self.reset)
        buttons_layout.add_widget(reset_btn)
        
        compare_btn = NeonButton(text="🔍 COMPARE")
        compare_btn.bind(on_press=self.compare)
        buttons_layout.add_widget(compare_btn)
        
        main_content.add_widget(buttons_layout)
        
        self.add_widget(main_content)
        
    def select_first(self, instance):
        if self.filechooser.selection:
            self.img1 = self.filechooser.selection[0]
            self.image1_widget.source = self.img1
            self.result.text = "✓ FACE 1 LOADED"
            self.result.color = (0.3, 0.9, 0.3, 1)
            self.animate_neon(self.image1_widget)
            
    def select_second(self, instance):
        if self.filechooser.selection:
            self.img2 = self.filechooser.selection[0]
            self.image2_widget.source = self.img2
            self.result.text = "✓ FACE 2 LOADED"
            self.result.color = (0.3, 0.9, 0.3, 1)
            self.animate_neon(self.image2_widget)
            
    def animate_neon(self, widget):
        def reset_color(*args):
            with widget.canvas.before:
                Color(0.2, 0.8, 0.9, 1)
                
        for i in range(3):
            Clock.schedule_once(lambda dt: setattr(widget, 'source', widget.source), i * 0.1)
            
    def reset(self, instance):
        self.img1 = None
        self.img2 = None
        self.image1_widget.source = ''
        self.image2_widget.source = ''
        self.result.text = "READY TO COMPARE"
        self.result.color = (0.2, 0.8, 0.9, 1)
        self.distance_label.text = ""
        
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
            
            if result == "Twins":
                color = (0.3, 0.9, 0.3, 1)
                emoji = "👯 TWINS DETECTED!"
            elif result == "Same Person":
                color = (0.2, 0.8, 0.9, 1)
                emoji = "✅ SAME PERSON"
            else:
                color = (0.9, 0.3, 0.3, 1)
                emoji = "❌ DIFFERENT PEOPLE"
                
            self.result.text = emoji
            self.result.color = color
            self.distance_label.text = f"Similarity Distance: {dist:.4f}"
            
            # Flash animation
            def flash(*args):
                self.result.color = (1, 1, 1, 1)
                Clock.schedule_once(lambda dt: setattr(self.result, 'color', color), 0.1)
            Clock.schedule_once(flash, 0)
        else:
            self.result.text = "⚠️ SELECT BOTH FACES"
            self.result.color = (0.9, 0.5, 0.2, 1)
            self.distance_label.text = ""

# Main App with design selector
class TwinApp(App):
    def build(self):
        # Choose design: 1 for Glassmorphism, 2 for Neon Modern
        design_choice = 2  # Change to 1 for first design
        
        if design_choice == 1:
            Window.clearcolor = (0.95, 0.95, 0.95, 1)
            return TwinUIDesign1()
        else:
            Window.clearcolor = (0.05, 0.05, 0.15, 1)
            return TwinUIDesign2()

if __name__ == "__main__":
    TwinApp().run()