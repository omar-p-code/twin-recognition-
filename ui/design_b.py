"""
Design B — Modern Minimal
Clean white + purple-to-pink gradient. Consumer mobile app aesthetic.
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
from kivy.graphics import Color, RoundedRectangle, Rectangle, Line, Ellipse
from kivy.core.window import Window
from kivy.metrics import dp, sp
from kivy.clock import Clock
from kivy.utils import platform

# ── Palette ──────────────────────────────────────────────────────────────────
BG        = (0.969, 0.969, 0.988, 1)   # #f7f7fc
WHITE     = (1,     1,     1,     1)
PURPLE    = (0.486, 0.227, 0.929, 1)   # #7c3aed
PURPLE_LT = (0.659, 0.333, 0.969, 1)   # #a855f7
PINK      = (0.925, 0.282, 0.600, 1)   # #ec4899
PINK_LT   = (0.957, 0.443, 0.714, 1)   # #f472b6
TEXT_PRI  = (0.067, 0.067, 0.067, 1)   # #111
TEXT_SEC  = (0.424, 0.455, 0.518, 1)   # #6b7480
TEXT_DIM  = (0.612, 0.635, 0.659, 1)   # #9ca3af
SLOT_A_BG = (0.953, 0.933, 1.000, 1)   # #f3e8ff
SLOT_B_BG = (0.988, 0.906, 0.953, 1)   # #fce7f3
SLOT_A_BD = (0.769, 0.714, 0.992, 1)   # #c4b5fd
SLOT_B_BD = (0.976, 0.659, 0.831, 1)   # #f9a8d4

# ─────────────────────────────────────────────────────────────────────────────

class WhiteCard(BoxLayout):
    """White rounded card with a soft shadow simulation."""

    def __init__(self, radius=20, **kwargs):
        super().__init__(**kwargs)
        self._r = radius
        self.bind(pos=self._redraw, size=self._redraw)

    def _redraw(self, *_):
        self.canvas.before.clear()
        with self.canvas.before:
            # Shadow
            Color(0, 0, 0, 0.05)
            RoundedRectangle(pos=(self.x + dp(2), self.y - dp(3)), size=self.size, radius=[dp(self._r)])
            # Card
            Color(*WHITE)
            RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(self._r)])


class GradientButton(Button):
    """Purple → pink gradient button."""

    def __init__(self, **kwargs):
        kwargs.setdefault("background_normal", "")
        kwargs.setdefault("background_color", (0, 0, 0, 0))
        kwargs.setdefault("color", WHITE)
        kwargs.setdefault("font_size", sp(15))
        kwargs.setdefault("bold", True)
        kwargs.setdefault("height", dp(58))
        super().__init__(**kwargs)
        self.bind(pos=self._redraw, size=self._redraw)

    def _redraw(self, *_):
        self.canvas.before.clear()
        third = self.width / 3
        with self.canvas.before:
            Color(*PURPLE)
            RoundedRectangle(pos=self.pos, size=(third, self.height),
                             radius=[dp(18), dp(0), dp(0), dp(18)])
            Color(*PURPLE_LT)
            RoundedRectangle(pos=(self.x + third, self.y), size=(third, self.height), radius=[0])
            Color(*PINK)
            RoundedRectangle(pos=(self.x + third * 2, self.y), size=(third, self.height),
                             radius=[dp(0), dp(18), dp(18), dp(0)])


class PillButton(Button):
    """Small solid-color pill button."""

    def __init__(self, bg=None, **kwargs):
        kwargs.setdefault("background_normal", "")
        kwargs.setdefault("background_color", (0, 0, 0, 0))
        kwargs.setdefault("color", WHITE)
        kwargs.setdefault("font_size", sp(12))
        kwargs.setdefault("bold", True)
        kwargs.setdefault("height", dp(34))
        super().__init__(**kwargs)
        self._bg = bg or PURPLE
        self.bind(pos=self._r, size=self._r)

    def _r(self, *_):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(*self._bg)
            RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(10)])


class PhotoSlot(BoxLayout):
    """Upload slot for one face image."""

    def __init__(self, label, is_left=True, **kwargs):
        super().__init__(orientation="vertical", spacing=dp(8), padding=dp(12), **kwargs)
        self._is_left = is_left

        # Card background
        slot_bg  = SLOT_A_BG if is_left else SLOT_B_BG
        slot_bd  = SLOT_A_BD if is_left else SLOT_B_BD
        btn_col  = PURPLE if is_left else PINK
        tag      = "Photo 1" if is_left else "Photo 2"
        tag_col  = PURPLE if is_left else PINK

        self.bind(pos=self._redraw, size=self._redraw)
        self._bg_col = slot_bg
        self._bd_col = slot_bd

        # Image / placeholder area
        self.thumb = Image(size_hint=(1, 1), opacity=0)
        placeholder = FloatLayout(size_hint=(1, 1))
        with placeholder.canvas.before:
            Color(*slot_bg)
            self._ph_rect = RoundedRectangle(radius=[dp(14)])
            Color(*slot_bd)
            self._ph_line = Line(width=dp(1.5))
        placeholder.bind(pos=self._ph_pos, size=self._ph_size)

        ph_inner = BoxLayout(orientation="vertical", size_hint=(1, 1))
        icon_lbl = Label(
            text=f"[size=28sp]🖼️[/size]\n[color=#{self._hexc(tag_col)}][size=9sp]{tag}[/size][/color]",
            markup=True,
            halign="center",
        )
        ph_inner.add_widget(icon_lbl)
        placeholder.add_widget(ph_inner)
        placeholder.add_widget(self.thumb)
        self.placeholder = placeholder

        # Choose button
        self.choose_btn = PillButton(bg=btn_col, text="+ Choose", size_hint_y=None, height=dp(34))
        self.choose_btn.bind(on_release=self._on_load)

        self.add_widget(placeholder)
        self.add_widget(self.choose_btn)

        self.image_path = None

    def _redraw(self, *_):
        self.canvas.before.clear()

    def _ph_pos(self, w, _):
        self._ph_rect.pos = w.pos
        self._ph_line.rounded_rectangle = [w.x, w.y, w.width, w.height, dp(14)]

    def _ph_size(self, w, _):
        self._ph_rect.size = w.size
        self._ph_line.rounded_rectangle = [w.x, w.y, w.width, w.height, dp(14)]

    @staticmethod
    def _hexc(c):
        r, g, b = int(c[0]*255), int(c[1]*255), int(c[2]*255)
        return f"{r:02x}{g:02x}{b:02x}"

    def _on_load(self, *_):
        if platform == "android":
            try:
                from plyer import filechooser as fc
                fc.open_file(on_selection=self._on_picked, filters=["*.jpg", "*.jpeg", "*.png"])
                return
            except Exception:
                pass

        from kivy.uix.filechooser import FileChooserIconView
        chooser = FileChooserIconView(filters=["*.jpg", "*.jpeg", "*.png"])
        popup   = Popup(title="Select Image", content=chooser, size_hint=(0.9, 0.9))
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
        self.image_path    = path
        self.thumb.source  = path
        self.thumb.opacity = 1
        self.choose_btn.text = "✓ Loaded"


class ResultCard(WhiteCard):
    def __init__(self, **kwargs):
        super().__init__(radius=24, orientation="vertical", padding=dp(18), spacing=dp(12), **kwargs)

        # Header row
        header = BoxLayout(orientation="horizontal", size_hint_y=None, height=dp(54), spacing=dp(10))

        lbl_col = BoxLayout(orientation="vertical")
        self.sub_lbl = Label(
            text="[color=#9ca3af][size=11sp]Result[/size][/color]",
            markup=True,
            halign="left",
            size_hint_y=None,
            height=dp(18),
        )
        self.sub_lbl.bind(size=lambda w, _: setattr(w, "text_size", (w.width, None)))
        self.result_lbl = Label(
            text="[color=#111111][b]Upload images to start[/b][/color]",
            markup=True,
            font_size=sp(18),
            bold=True,
            halign="left",
            size_hint_y=None,
            height=dp(32),
        )
        self.result_lbl.bind(size=lambda w, _: setattr(w, "text_size", (w.width, None)))
        lbl_col.add_widget(self.sub_lbl)
        lbl_col.add_widget(self.result_lbl)

        self.conf_box = FloatLayout(size_hint=(None, 1), width=dp(64))
        self.conf_box.bind(pos=self._conf_draw, size=self._conf_draw)
        self.conf_lbl = Label(
            text="[color=#ffffff][b]–%[/b][/color]\n[size=9sp][color=rgba(255,255,255,0.7)]MATCH[/color][/size]",
            markup=True,
            halign="center",
            font_size=sp(18),
            bold=True,
        )
        self.conf_box.add_widget(self.conf_lbl)

        header.add_widget(lbl_col)
        header.add_widget(self.conf_box)

        # Distance bar
        bar_area = BoxLayout(orientation="vertical", size_hint_y=None, height=dp(38), spacing=dp(6))
        bar_row2 = BoxLayout(orientation="horizontal", size_hint_y=None, height=dp(18))
        bar_row2.add_widget(Label(
            text="[color=#6b7480][size=12sp]Similarity distance[/size][/color]",
            markup=True, halign="left", size_hint_x=1,
        ))
        self.dist_lbl = Label(
            text="[color=#7c3aed][size=12sp]–[/size][/color]",
            markup=True, halign="right", size_hint_x=None, width=dp(60),
        )
        bar_row2.add_widget(self.dist_lbl)
        self.bar = ProgressBar(max=100, value=0, size_hint_y=None, height=dp(8))
        bar_area.add_widget(bar_row2)
        bar_area.add_widget(self.bar)

        # Pills
        pills_row = BoxLayout(orientation="horizontal", size_hint_y=None, height=dp(48), spacing=dp(8))
        self.pills = []
        configs = [
            ("✅", "Same Person", False),
            ("👯", "Twins",       True),
            ("❌", "Different",   False),
        ]
        for icon, lbl, active in configs:
            p = Label(
                text=f"[size=14sp]{icon}[/size]\n[size=9sp][b]{lbl}[/b][/size]",
                markup=True,
                halign="center",
                font_size=sp(9),
            )
            p.bind(pos=self._pill_draw(p, active), size=self._pill_draw(p, active))
            pills_row.add_widget(p)
            self.pills.append((p, lbl))

        self.add_widget(header)
        self.add_widget(bar_area)
        self.add_widget(pills_row)

    def _conf_draw(self, w, _):
        w.canvas.before.clear()
        with w.canvas.before:
            Color(*PURPLE)
            RoundedRectangle(pos=w.pos, size=w.size, radius=[dp(14)])

    def _pill_draw(self, widget, active):
        def _inner(w, _):
            w.canvas.before.clear()
            with w.canvas.before:
                if active:
                    Color(0.953, 0.933, 1.0, 1)
                    RoundedRectangle(pos=w.pos, size=w.size, radius=[dp(12)])
                    Color(*SLOT_A_BD)
                    Line(rounded_rectangle=[w.x, w.y, w.width, w.height, dp(12)], width=dp(1.2))
                else:
                    Color(0.976, 0.980, 0.984, 1)
                    RoundedRectangle(pos=w.pos, size=w.size, radius=[dp(12)])
        return _inner

    def update(self, distance, result, confidence):
        icons_map = {
            "Same Person": "✅ Same Person",
            "Twins":       "👯 Likely Twins",
            "Different":   "❌ Different People",
        }
        col_map = {"Same Person": "#16a34a", "Twins": "#7c3aed", "Different": "#dc2626"}
        col = col_map.get(result, "#111")

        self.result_lbl.text = f"[b][color={col}]{icons_map.get(result, result)}[/color][/b]"
        self.conf_lbl.text   = f"[color=#ffffff][b]{confidence}%[/b][/color]\n[size=9sp][color=rgba(255,255,255,0.7)]MATCH[/color][/size]"
        self.dist_lbl.text   = f"[color=#7c3aed][size=12sp]{distance:.4f}[/size][/color]"
        self.bar.value       = min(int(confidence), 100)

        for p, lbl in self.pills:
            active = (lbl == result)
            if active:
                p.canvas.before.clear()
                with p.canvas.before:
                    Color(0.953, 0.933, 1.0, 1)
                    RoundedRectangle(pos=p.pos, size=p.size, radius=[dp(12)])
                    Color(*SLOT_A_BD)
                    Line(rounded_rectangle=[p.x, p.y, p.width, p.height, dp(12)], width=dp(1.2))
            else:
                p.canvas.before.clear()
                with p.canvas.before:
                    Color(0.976, 0.980, 0.984, 1)
                    RoundedRectangle(pos=p.pos, size=p.size, radius=[dp(12)])


class DesignBRoot(BoxLayout):
    def __init__(self, model=None, th_same=0.35, th_twin=0.60, device="cpu", **kwargs):
        super().__init__(orientation="vertical", **kwargs)
        self.model   = model
        self.th_same = th_same
        self.th_twin = th_twin
        self.device  = device
        Window.clearcolor = BG
        self._build()

    def _build(self):
        # ── Gradient header ───────────────────────────────────────────────────
        header = BoxLayout(
            orientation="vertical",
            size_hint_y=None,
            height=dp(120),
            padding=(dp(22), dp(14)),
            spacing=dp(4),
        )
        with header.canvas.before:
            # Left third — purple
            Color(*PURPLE)
            self._h1 = Rectangle()
            # Middle third — purple-light
            Color(*PURPLE_LT)
            self._h2 = Rectangle()
            # Right third — pink
            Color(*PINK)
            self._h3 = Rectangle()
            # Decorative circles
            Color(1, 1, 1, 0.07)
            self._c1 = Ellipse()
            self._c2 = Ellipse()
        header.bind(pos=self._hdr_draw, size=self._hdr_draw)

        app_row = BoxLayout(orientation="horizontal", size_hint_y=None, height=dp(36), spacing=dp(10))
        icon_box = FloatLayout(size_hint=(None, 1), width=dp(36))
        with icon_box.canvas.before:
            Color(1, 1, 1, 0.18)
            RoundedRectangle(pos=icon_box.pos, size=icon_box.size, radius=[dp(10)])
        icon_box.bind(pos=lambda w, _: self._icon_draw(w), size=lambda w, _: self._icon_draw(w))
        self._icon_box = icon_box
        icon_lbl = Label(text="[size=16sp]👁️[/size]", markup=True)
        icon_box.add_widget(icon_lbl)

        title_col = BoxLayout(orientation="vertical")
        title_col.add_widget(Label(
            text="[color=rgba(255,255,255,0.7)][size=11sp]Face Match[/size][/color]",
            markup=True, halign="left", size_hint_y=None, height=dp(16),
        ))
        title_col.add_widget(Label(
            text="[color=#ffffff][b][size=17sp]Twin Recognition[/size][/b][/color]",
            markup=True, halign="left", size_hint_y=None, height=dp(22),
        ))
        app_row.add_widget(icon_box)
        app_row.add_widget(title_col)

        subtitle = Label(
            text="[color=rgba(255,255,255,0.65)][size=12sp]Compare two photos and discover if they're the same person, twins, or different.[/size][/color]",
            markup=True, halign="left",
        )
        subtitle.bind(size=lambda w, _: setattr(w, "text_size", (w.width, None)))

        header.add_widget(app_row)
        header.add_widget(subtitle)

        # ── Scroll body ───────────────────────────────────────────────────────
        scroll = ScrollView()
        body   = BoxLayout(orientation="vertical", padding=dp(16), spacing=dp(14), size_hint_y=None)
        body.bind(minimum_height=body.setter("height"))

        # Photo upload row
        upload_row = BoxLayout(orientation="horizontal", size_hint_y=None, height=dp(190), spacing=dp(12))

        self.slot_a = PhotoSlot("Photo 1", is_left=True,  size_hint_x=1)
        self.slot_b = PhotoSlot("Photo 2", is_left=False, size_hint_x=1)

        swap_col = BoxLayout(orientation="vertical", size_hint=(None, 1), width=dp(36))
        swap_col.add_widget(BoxLayout())
        swap_btn = Button(
            text="⇄", font_size=sp(16), bold=True, color=PURPLE,
            size_hint=(None, None), width=dp(36), height=dp(36),
            background_normal="", background_color=(0, 0, 0, 0),
        )
        with swap_btn.canvas.before:
            Color(*WHITE)
            RoundedRectangle(pos=swap_btn.pos, size=swap_btn.size, radius=[dp(18)])
        swap_btn.bind(
            pos=lambda w, _: self._swap_draw(w),
            size=lambda w, _: self._swap_draw(w),
        )
        self._swap_btn = swap_btn

        swap_col.add_widget(swap_btn)
        swap_col.add_widget(BoxLayout())

        upload_row.add_widget(self.slot_a)
        upload_row.add_widget(swap_col)
        upload_row.add_widget(self.slot_b)

        # Compare button
        self.compare_btn = GradientButton(text="Compare Faces ✨", size_hint_y=None)
        self.compare_btn.bind(on_release=self._on_compare)

        # Result card
        self.result_card = ResultCard(size_hint_y=None, height=dp(196))

        # History section
        history_title = Label(
            text="[color=#9ca3af][size=12sp][b]Recent Comparisons[/b][/size][/color]",
            markup=True, halign="left", size_hint_y=None, height=dp(24),
        )
        history_title.bind(size=lambda w, _: setattr(w, "text_size", (w.width, None)))

        self.history_box = BoxLayout(orientation="vertical", size_hint_y=None, height=dp(10), spacing=dp(8))
        self.history_box.bind(minimum_height=self.history_box.setter("height"))
        self._history_items = []

        body.add_widget(upload_row)
        body.add_widget(self.compare_btn)
        body.add_widget(self.result_card)
        body.add_widget(history_title)
        body.add_widget(self.history_box)
        scroll.add_widget(body)

        # ── Bottom nav ────────────────────────────────────────────────────────
        nav = BoxLayout(
            orientation="horizontal",
            size_hint_y=None,
            height=dp(60),
            padding=(dp(10), dp(6)),
        )
        with nav.canvas.before:
            Color(*WHITE)
            self._nav_rect = Rectangle()
            Color(0.953, 0.953, 0.965, 1)
            self._nav_line = Line(width=dp(0.8))
        nav.bind(pos=self._nav_draw, size=self._nav_draw)

        for icon, label, active in [("🏠","Home",False),("📷","Compare",True),("📊","History",False),("⚙️","Settings",False)]:
            col = BoxLayout(orientation="vertical", spacing=dp(2))
            col.add_widget(Label(text=f"[size=18sp]{icon}[/size]", markup=True, size_hint_y=None, height=dp(26)))
            lbl_c = "#7c3aed" if active else "#9ca3af"
            col.add_widget(Label(
                text=f"[color={lbl_c}][size=9sp]{label}[/size][/color]",
                markup=True, size_hint_y=None, height=dp(14),
            ))
            nav.add_widget(col)

        self.add_widget(header)
        self.add_widget(scroll)
        self.add_widget(nav)

    # ── Canvas helpers ────────────────────────────────────────────────────────

    def _hdr_draw(self, w, _):
        third = w.width / 3
        self._h1.pos  = w.pos;            self._h1.size = (third, w.height)
        self._h2.pos  = (w.x+third, w.y); self._h2.size = (third, w.height)
        self._h3.pos  = (w.x+third*2, w.y); self._h3.size = (third, w.height)
        self._c1.pos  = (w.right - dp(200), w.top - dp(200)); self._c1.size = (dp(160), dp(160))
        self._c2.pos  = (w.right - dp(100), w.top - dp(100)); self._c2.size = (dp(80), dp(80))

    def _icon_draw(self, w):
        w.canvas.before.clear()
        with w.canvas.before:
            Color(1, 1, 1, 0.18)
            RoundedRectangle(pos=w.pos, size=w.size, radius=[dp(10)])

    def _swap_draw(self, w):
        w.canvas.before.clear()
        with w.canvas.before:
            Color(*WHITE)
            RoundedRectangle(pos=w.pos, size=w.size, radius=[dp(18)])

    def _nav_draw(self, w, _):
        self._nav_rect.pos  = w.pos
        self._nav_rect.size = w.size
        self._nav_line.points = [w.x, w.y + w.height, w.x + w.width, w.y + w.height]

    # ── Logic ─────────────────────────────────────────────────────────────────

    def _on_compare(self, *_):
        if not self.slot_a.image_path or not self.slot_b.image_path:
            Popup(
                title="Notice",
                content=Label(text="Please load both images first."),
                size_hint=(0.8, 0.25),
            ).open()
            return

        self.compare_btn.text = "Analysing... ⏳"
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
                Popup(
                    title="Detection Failed",
                    content=Label(text="Could not detect a face in one of the images."),
                    size_hint=(0.8, 0.28),
                ).open()
                self.compare_btn.text = "Compare Faces ✨"
                return

            max_d = self.th_twin * 1.5
            conf  = max(0, min(100, int((1 - dist / max_d) * 100)))
            self.result_card.update(dist, result, conf)
            self._add_history(result)

        except Exception as e:
            Popup(title="Error", content=Label(text=str(e)), size_hint=(0.8, 0.28)).open()
        finally:
            self.compare_btn.text = "Compare Faces ✨"

    def _add_history(self, result):
        col_map = {"Same Person": "#16a34a", "Twins": "#7c3aed", "Different": "#dc2626"}
        col = col_map.get(result, "#9ca3af")
        n   = len(self._history_items) + 1

        row = WhiteCard(radius=14, orientation="horizontal",
                        padding=(dp(14), dp(10)), spacing=dp(8),
                        size_hint_y=None, height=dp(46))
        row.add_widget(Label(
            text=f"[color=#374151][size=12sp]Photo A vs Photo B (#{n})[/size][/color]",
            markup=True, halign="left",
        ))
        row.add_widget(Label(
            text=f"[color={col}][size=11sp][b]{result}[/b][/size][/color]",
            markup=True, halign="right", size_hint_x=None, width=dp(90),
        ))
        self.history_box.add_widget(row)
        self._history_items.append(row)


class DesignBApp(App):
    def build(self):
        Window.clearcolor = BG

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

        return DesignBRoot(model=model, th_same=th_same or 0.35, th_twin=th_twin or 0.60, device=device or "cpu")


if __name__ == "__main__":
    DesignBApp().run()
