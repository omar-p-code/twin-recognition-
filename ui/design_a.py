"""
Design A — Dark Scientific
Dark navy theme with teal/blue accents. Forensic/lab aesthetic.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.scrollview import ScrollView
from kivy.uix.popup import Popup
from kivy.uix.progressbar import ProgressBar
from kivy.graphics import Color, RoundedRectangle, Rectangle, Line
from kivy.core.window import Window
from kivy.metrics import dp, sp
from kivy.clock import Clock
from kivy.utils import platform

# ── Palette ──────────────────────────────────────────────────────────────────
BG          = (0.051, 0.055, 0.071, 1)       # #0d1117
BG2         = (0.071, 0.082, 0.102, 1)       # #12151a
TEAL        = (0.000, 1.000, 0.784, 1)       # #00ffc8
BLUE        = (0.290, 0.608, 1.000, 1)       # #4a9eff
GOLD        = (1.000, 0.843, 0.000, 1)       # #ffd700
TEXT_PRI    = (0.910, 0.957, 1.000, 1)       # #e8f4ff
TEXT_SEC    = (0.627, 0.753, 0.847, 1)       # #a0c4d8
TEXT_DIM    = (0.227, 0.314, 0.376, 1)       # #3a5060
CARD_A      = (0.000, 1.000, 0.784, 0.04)   # teal card bg
CARD_B      = (0.000, 0.502, 1.000, 0.04)   # blue card bg
CARD_RES    = (1.000, 0.784, 0.000, 0.04)   # gold card bg
BORDER_TEAL = (0.000, 1.000, 0.784, 0.15)
BORDER_BLUE = (0.000, 0.502, 1.000, 0.15)
BORDER_GOLD = (1.000, 0.784, 0.000, 0.20)

# ─────────────────────────────────────────────────────────────────────────────

class DarkCard(BoxLayout):
    """Rounded rectangle card with a subtle colored border."""

    def __init__(self, bg_color=None, border_color=None, radius=14, **kwargs):
        super().__init__(**kwargs)
        self._bg    = bg_color    or BG2
        self._bd    = border_color or BORDER_TEAL
        self._r     = radius
        self.bind(pos=self._redraw, size=self._redraw)

    def _redraw(self, *_):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(*self._bg)
            RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(self._r)])
            Color(*self._bd)
            Line(
                rounded_rectangle=[self.x, self.y, self.width, self.height, dp(self._r)],
                width=dp(1),
            )


class GlowButton(Button):
    """Gradient-style button (teal → blue)."""

    def __init__(self, **kwargs):
        kwargs.setdefault("background_normal", "")
        kwargs.setdefault("background_color", (0, 0, 0, 0))
        kwargs.setdefault("color", (0, 0, 0, 1))
        kwargs.setdefault("font_size", sp(13))
        kwargs.setdefault("bold", True)
        kwargs.setdefault("height", dp(52))
        super().__init__(**kwargs)
        self.bind(pos=self._redraw, size=self._redraw)

    def _redraw(self, *_):
        self.canvas.before.clear()
        with self.canvas.before:
            # Simulate gradient with two overlapping rects (teal → mid → blue)
            Color(0.000, 1.000, 0.784, 1)
            RoundedRectangle(pos=self.pos, size=(self.width * 0.5, self.height), radius=[dp(14), dp(0), dp(0), dp(14)])
            Color(0.290, 0.608, 1.000, 1)
            RoundedRectangle(pos=(self.x + self.width * 0.5, self.y), size=(self.width * 0.5, self.height), radius=[dp(0), dp(14), dp(14), dp(0)])


class SmallButton(Button):
    def __init__(self, teal=True, **kwargs):
        kwargs.setdefault("background_normal", "")
        kwargs.setdefault("background_color", (0, 0, 0, 0))
        col = TEAL if teal else BLUE
        kwargs.setdefault("color", col)
        kwargs.setdefault("font_size", sp(10))
        kwargs.setdefault("bold", True)
        kwargs.setdefault("height", dp(30))
        super().__init__(**kwargs)
        self._teal = teal
        self.bind(pos=self._r, size=self._r)

    def _r(self, *_):
        self.canvas.before.clear()
        bc = (0.000, 1.000, 0.784, 0.12) if self._teal else (0.000, 0.502, 1.000, 0.12)
        bd = (0.000, 1.000, 0.784, 0.30) if self._teal else (0.000, 0.502, 1.000, 0.30)
        with self.canvas.before:
            Color(*bc)
            RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(8)])
            Color(*bd)
            Line(rounded_rectangle=[self.x, self.y, self.width, self.height, dp(8)], width=dp(0.8))


class PhotoSlot(DarkCard):
    """One side of the face comparison: icon + label + load button."""

    def __init__(self, label, teal=True, **kwargs):
        bc = CARD_A if teal else CARD_B
        bd = BORDER_TEAL if teal else BORDER_BLUE
        super().__init__(bg_color=bc, border_color=bd, radius=16,
                         orientation="horizontal", padding=dp(14), spacing=dp(14), **kwargs)

        accent = TEAL if teal else BLUE
        scan   = "SCAN A" if teal else "SCAN B"

        # Icon box
        icon_box = FloatLayout(size_hint=(None, 1), width=dp(80))
        with icon_box.canvas.before:
            Color(*(CARD_A if teal else CARD_B))
            self._ib_rect = RoundedRectangle(radius=[dp(12)])
            Color(*(BORDER_TEAL if teal else BORDER_BLUE))
            self._ib_line = Line(width=dp(1))
        icon_box.bind(pos=lambda w, _: self._update_icon(w), size=lambda w, _: self._update_icon(w))
        self._icon_box = icon_box

        icon_lbl = Label(
            text=f"[size=28sp]👤[/size]\n[size=9sp][color={self._hex(accent)}]{scan}[/color][/size]",
            markup=True,
            halign="center",
            size_hint=(1, 1),
        )
        icon_box.add_widget(icon_lbl)

        # Info side
        info = BoxLayout(orientation="vertical", spacing=dp(6), padding=(0, dp(4)))

        subject = Label(
            text=f"[color={self._hex(TEXT_SEC)}][b]{label}[/b][/color]",
            markup=True,
            font_size=sp(13),
            halign="left",
            size_hint_y=None,
            height=dp(22),
        )
        subject.bind(size=lambda w, _: setattr(w, "text_size", (w.width, None)))

        status_lbl = Label(
            text=f"[color={self._hex(TEXT_DIM)}]No biometric data loaded[/color]",
            markup=True,
            font_size=sp(11),
            halign="left",
            size_hint_y=None,
            height=dp(18),
        )
        status_lbl.bind(size=lambda w, _: setattr(w, "text_size", (w.width, None)))
        self.status_lbl = status_lbl

        # Thumbnail (hidden until image loaded)
        self.thumb = Image(size_hint_y=None, height=dp(70), opacity=0)

        load_btn = SmallButton(teal=teal, text="LOAD IMAGE", size_hint=(None, None), width=dp(100), height=dp(28))
        load_btn.bind(on_release=self._on_load)
        self.load_btn = load_btn

        info.add_widget(subject)
        info.add_widget(status_lbl)
        info.add_widget(self.thumb)
        info.add_widget(load_btn)
        info.add_widget(BoxLayout())  # spacer

        self.add_widget(icon_box)
        self.add_widget(info)

        self.image_path = None

    @staticmethod
    def _hex(c):
        r, g, b = int(c[0]*255), int(c[1]*255), int(c[2]*255)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _update_icon(self, w):
        self._ib_rect.pos  = w.pos
        self._ib_rect.size = w.size
        self._ib_line.rounded_rectangle = [w.x, w.y, w.width, w.height, dp(12)]

    def _on_load(self, *_):
        self._open_picker()

    def _open_picker(self):
        if platform == "android":
            try:
                from plyer import filechooser as fc
                fc.open_file(on_selection=self._on_picked, filters=[("*.jpg", "*.jpeg", "*.png")])
                return
            except Exception:
                pass

        from kivy.uix.filechooser import FileChooserIconView
        chooser = FileChooserIconView(filters=["*.jpg", "*.jpeg", "*.png", "*.JPG"])
        popup = Popup(title="Select Image", content=chooser, size_hint=(0.9, 0.9))
        def _pick(fc, sel, _touch):
            if sel:
                self._on_picked(sel)
                popup.dismiss()
        chooser.bind(on_submit=_pick)
        popup.open()

    def _on_picked(self, selection):
        if not selection:
            return
        path = selection[0] if isinstance(selection, list) else selection
        self.image_path = path
        self.thumb.source  = path
        self.thumb.opacity = 1
        self.status_lbl.text = f"[color=#00ffc8]Image loaded ✓[/color]"


class ResultCard(DarkCard):
    def __init__(self, **kwargs):
        super().__init__(bg_color=CARD_RES, border_color=BORDER_GOLD, radius=16,
                         orientation="vertical", padding=dp(18), spacing=dp(10), **kwargs)

        # Header row
        header = BoxLayout(orientation="horizontal", size_hint_y=None, height=dp(48), spacing=dp(10))

        self.result_lbl = Label(
            text="[color=#4a6080]Awaiting analysis...[/color]",
            markup=True,
            font_size=sp(16),
            bold=True,
            halign="left",
            size_hint_x=1,
        )
        self.result_lbl.bind(size=lambda w, _: setattr(w, "text_size", (w.width, None)))

        conf_box = FloatLayout(size_hint=(None, 1), width=dp(60))
        with conf_box.canvas.before:
            Color(1, 0.784, 0, 0.12)
            RoundedRectangle(pos=conf_box.pos, size=conf_box.size, radius=[dp(8)])
        conf_box.bind(pos=self._cb_pos, size=self._cb_size)
        self._conf_rect_ref = conf_box

        self.conf_lbl = Label(
            text="[color=#ffd700]–%[/color]\n[size=9sp][color=#806040]CONF.[/color][/size]",
            markup=True,
            halign="center",
            font_size=sp(16),
            bold=True,
        )
        conf_box.add_widget(self.conf_lbl)

        header.add_widget(self.result_lbl)
        header.add_widget(conf_box)

        # Progress bar
        bar_row = BoxLayout(orientation="vertical", size_hint_y=None, height=dp(36), spacing=dp(4))
        bar_header = BoxLayout(orientation="horizontal", size_hint_y=None, height=dp(16))
        bar_header.add_widget(Label(text="[color=#4a6080][size=10sp]SIMILARITY SCORE[/size][/color]",
                                    markup=True, halign="left", size_hint_x=1))
        self.dist_lbl = Label(text="[color=#00ffc8][size=10sp]–[/size][/color]",
                              markup=True, halign="right", size_hint_x=None, width=dp(60))
        bar_header.add_widget(self.dist_lbl)
        self.bar = ProgressBar(max=100, value=0, size_hint_y=None, height=dp(6))
        bar_row.add_widget(bar_header)
        bar_row.add_widget(self.bar)

        # Classification pills
        pills_row = BoxLayout(orientation="horizontal", size_hint_y=None, height=dp(40), spacing=dp(8))
        self.pills = []
        for label in ["Same Person", "Twins", "Different"]:
            p = Label(
                text=f"[color=#2a3a50]{label}[/color]",
                markup=True,
                font_size=sp(9),
                bold=True,
                halign="center",
            )
            p.bind(size=lambda w, _: setattr(w, "text_size", (w.width, None)))
            self.pills.append((p, label))
            pills_row.add_widget(p)

        self.add_widget(header)
        self.add_widget(bar_row)
        self.add_widget(pills_row)

        self._draw_pills("none")

    def _cb_pos(self, w, _):
        w.canvas.before.clear()
        with w.canvas.before:
            Color(1, 0.784, 0, 0.12)
            RoundedRectangle(pos=w.pos, size=w.size, radius=[dp(8)])

    def _cb_size(self, w, _):
        self._cb_pos(w, _)

    def update(self, distance, result, confidence):
        icons = {"Same Person": "✅ SAME PERSON", "Twins": "👯 TWINS DETECTED", "Different": "❌ DIFFERENT PEOPLE"}
        colors = {"Same Person": "#00ffc8", "Twins": "#ffd700", "Different": "#ff4060"}

        col = colors.get(result, "#4a6080")
        self.result_lbl.text = f"[color={col}]{icons.get(result, result)}[/color]"
        self.conf_lbl.text   = f"[color=#ffd700]{confidence}%[/color]\n[size=9sp][color=#806040]CONF.[/color][/size]"
        self.dist_lbl.text   = f"[color=#00ffc8][size=10sp]{distance:.4f}[/size][/color]"
        self.bar.value       = min(int(confidence), 100)
        self._draw_pills(result)

    def _draw_pills(self, active):
        for pill, label in self.pills:
            if label == active:
                pill.text = f"[color=#ffd700][b]{label}[/b][/color]"
            else:
                pill.text = f"[color=#2a3a50]{label}[/color]"


class DesignARoot(BoxLayout):
    def __init__(self, model=None, th_same=0.35, th_twin=0.60, device="cpu", **kwargs):
        super().__init__(orientation="vertical", **kwargs)
        self.model   = model
        self.th_same = th_same
        self.th_twin = th_twin
        self.device  = device
        Window.clearcolor = BG

        self._build()

    def _build(self):
        # ── Header ────────────────────────────────────────────────────────────
        header = BoxLayout(
            orientation="vertical",
            size_hint_y=None,
            height=dp(88),
            padding=(dp(24), dp(12)),
            spacing=dp(2),
        )
        with header.canvas.before:
            Color(*BG2)
            self._hdr_rect = Rectangle()
            Color(0, 1, 0.784, 0.12)
            self._hdr_line = Line(width=dp(0.8))
        header.bind(pos=self._hdr_pos, size=self._hdr_size)

        tag_row = BoxLayout(orientation="horizontal", size_hint_y=None, height=dp(22), spacing=dp(8))
        dot = Label(text="[color=#00ffc8]⬤[/color]", markup=True, font_size=sp(8), size_hint=(None, 1), width=dp(12))
        tag = Label(
            text="[color=#00ffc8][size=10sp]TWIN RECOGNITION[/size][/color]",
            markup=True,
            halign="left",
            size_hint_x=1,
        )
        tag.bind(size=lambda w, _: setattr(w, "text_size", (w.width, None)))
        tag_row.add_widget(dot)
        tag_row.add_widget(tag)

        title = Label(
            text=f"[color={self._hex(TEXT_PRI)}][b]Face Analysis[/b][/color]",
            markup=True,
            font_size=sp(20),
            halign="left",
            size_hint_y=None,
            height=dp(32),
        )
        title.bind(size=lambda w, _: setattr(w, "text_size", (w.width, None)))

        header.add_widget(tag_row)
        header.add_widget(title)

        # ── Scroll body ───────────────────────────────────────────────────────
        scroll = ScrollView()
        body   = BoxLayout(orientation="vertical", padding=dp(20), spacing=dp(14), size_hint_y=None)
        body.bind(minimum_height=body.setter("height"))

        self.slot_a = PhotoSlot("Subject Alpha", teal=True, size_hint_y=None, height=dp(110))
        self.slot_b = PhotoSlot("Subject Beta",  teal=False, size_hint_y=None, height=dp(110))

        vs_row = BoxLayout(orientation="horizontal", size_hint_y=None, height=dp(28), spacing=dp(10))
        vs_row.add_widget(BoxLayout())
        vs_lbl = Label(text="[color=#00ffc8][size=10sp]VS[/size][/color]", markup=True, halign="center",
                       size_hint=(None, 1), width=dp(40))
        vs_row.add_widget(vs_lbl)
        vs_row.add_widget(BoxLayout())

        self.analyze_btn = GlowButton(text="⚡  RUN BIOMETRIC ANALYSIS", size_hint_y=None)
        self.analyze_btn.bind(on_release=self._on_analyze)

        self.result_card = ResultCard(size_hint_y=None, height=dp(170))

        body.add_widget(self.slot_a)
        body.add_widget(vs_row)
        body.add_widget(self.slot_b)
        body.add_widget(self.analyze_btn)
        body.add_widget(self.result_card)
        scroll.add_widget(body)

        # ── Footer ────────────────────────────────────────────────────────────
        footer = BoxLayout(orientation="horizontal", size_hint_y=None, height=dp(44),
                           padding=(dp(24), 0))
        with footer.canvas.before:
            Color(0, 1, 0.784, 0.08)
            self._ft_line = Line(width=dp(0.5))
        footer.bind(pos=self._ft_pos, size=self._ft_size)

        footer.add_widget(Label(
            text="[color=#2a4060][size=10sp]MODEL: RESNET50 • v2.0[/size][/color]",
            markup=True, halign="left",
        ))
        footer.add_widget(Label(
            text="[color=#2a4060][size=10sp]⬤ READY[/size][/color]",
            markup=True, halign="right",
        ))

        self.add_widget(header)
        self.add_widget(scroll)
        self.add_widget(footer)

    # ── Drawing helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _hex(c):
        r, g, b = int(c[0]*255), int(c[1]*255), int(c[2]*255)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _hdr_pos(self, w, _):
        self._hdr_rect.pos  = w.pos
        self._hdr_line.points = [w.x, w.y, w.x + w.width, w.y]

    def _hdr_size(self, w, _):
        self._hdr_rect.size = w.size
        self._hdr_line.points = [w.x, w.y, w.x + w.width, w.y]

    def _ft_pos(self, w, _):
        self._ft_line.points = [w.x, w.y + w.height, w.x + w.width, w.y + w.height]

    def _ft_size(self, w, _):
        self._ft_line.points = [w.x, w.y + w.height, w.x + w.width, w.y + w.height]

    # ── Logic ─────────────────────────────────────────────────────────────────

    def _on_analyze(self, *_):
        if not self.slot_a.image_path or not self.slot_b.image_path:
            self._show_error("Please load both images first.")
            return

        self.analyze_btn.text = "⏳  ANALYSING..."
        Clock.schedule_once(self._run_comparison, 0.1)

    def _run_comparison(self, *_):
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded.")

            from models.siamese import compare_faces
            dist, result, _ = compare_faces(
                self.model,
                self.slot_a.image_path,
                self.slot_b.image_path,
                self.th_same, self.th_twin,
                device=self.device,
                detect_faces=True,
            )

            if dist is None:
                self._show_error("Face detection failed. Ensure both images contain a clear face.")
                self.analyze_btn.text = "⚡  RUN BIOMETRIC ANALYSIS"
                return

            # Confidence: map distance to 0-100
            max_d = self.th_twin * 1.5
            conf  = max(0, min(100, int((1 - dist / max_d) * 100)))
            self.result_card.update(dist, result, conf)

        except Exception as e:
            self._show_error(str(e))
        finally:
            self.analyze_btn.text = "⚡  RUN BIOMETRIC ANALYSIS"

    def _show_error(self, msg):
        popup = Popup(
            title="Error",
            content=Label(text=msg, color=(1, 0.25, 0.25, 1)),
            size_hint=(0.8, 0.3),
        )
        popup.open()


class DesignAApp(App):
    def build(self):
        Window.clearcolor = BG

        # Try to load the model
        model = th_same = th_twin = device = None
        try:
            import torch
            from inference.predict import load_model
            from config import DEVICE
            ck = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", "checkpoint.pth")
            if os.path.exists(ck):
                model, th_same, th_twin = load_model(ck, DEVICE)
                device = DEVICE
                print("✓ Model loaded")
            else:
                print("! No checkpoint found — running in demo mode")
        except Exception as e:
            print(f"! Could not load model: {e}")

        return DesignARoot(model=model, th_same=th_same or 0.35, th_twin=th_twin or 0.60, device=device or "cpu")


if __name__ == "__main__":
    DesignAApp().run()
