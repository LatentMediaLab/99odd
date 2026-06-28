from __future__ import annotations

import os
import time
import tkinter as tk
from typing import Callable

from PIL import Image, ImageDraw, ImageFont, ImageTk

_WINDOW_SIZE = (720, 960)  # portrait window used in windowed (non-fullscreen) mode
_CROSSFADE_DURATION = 1.5
_TICK_MS = 33  # ~30 fps

_UNICODE_FONT_PATHS = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
]


def _display_geometry(index: int) -> tuple[int, int, int, int]:
    """
    Return (x, y, w, h) for the target display using the full screen bounds,
    including the macOS menu bar area.

    NSScreen.frame() gives the complete bounds; NSScreen.visibleFrame() would
    exclude the menu bar and Dock, leaving a gap at the top of the window.
    NSScreen uses a bottom-left origin, so the y coordinate must be flipped
    relative to the primary screen height before passing it to tkinter.

    Falls back to screeninfo, then to the default windowed size.
    """
    try:
        from AppKit import NSScreen
        screens = list(NSScreen.screens())
        s = screens[min(index, len(screens) - 1)]
        f = s.frame()
        primary_h = int(screens[0].frame().size.height)
        x = int(f.origin.x)
        # NSScreen origin is bottom-left; tkinter origin is top-left
        y = primary_h - int(f.origin.y) - int(f.size.height)
        return x, y, int(f.size.width), int(f.size.height)
    except Exception:
        pass
    try:
        from screeninfo import get_monitors
        monitors = get_monitors()
        m = monitors[min(index, len(monitors) - 1)]
        return m.x, m.y, m.width, m.height
    except Exception:
        pass
    return 0, 0, _WINDOW_SIZE[0], _WINDOW_SIZE[1]


def _hide_macos_menubar() -> None:
    """
    Ask AppKit to hide the menu bar and Dock for the duration of the process.
    Both reappear automatically when the process exits.
    Must be called after the tkinter window is mapped (i.e. via root.after),
    not during __init__ — calling it too early has no effect.
    """
    try:
        from AppKit import NSApplication, NSApplicationPresentationHideMenuBar, NSApplicationPresentationHideDock
        app = NSApplication.sharedApplication()
        app.setPresentationOptions_(
            NSApplicationPresentationHideMenuBar | NSApplicationPresentationHideDock
        )
    except Exception:
        pass


def _waiting_image(width: int, height: int) -> Image.Image:
    # Dim Arabic-Indic "٩٩" (99) centered on black — shown while idle
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    for fp in _UNICODE_FONT_PATHS:
        try:
            font = ImageFont.truetype(fp, 120)
            text = "٩٩"
            bb = draw.textbbox((0, 0), text, font=font)
            tx = (width - (bb[2] - bb[0])) // 2 - bb[0]
            ty = (height - (bb[3] - bb[1])) // 2 - bb[1]
            draw.text((tx, ty), text, font=font, fill=(30, 30, 30))
            break
        except Exception:
            continue
    return img


class DisplayManager:
    def __init__(self) -> None:
        fullscreen = os.environ.get("FULLSCREEN", "0") == "1"
        display_index = int(os.environ.get("DISPLAY_INDEX", "0"))

        self.root = tk.Tk()
        self.root.configure(bg="black")
        # overrideredirect removes the title bar without taking exclusive control
        # of the display — other screens remain usable, unlike pygame.FULLSCREEN
        self.root.overrideredirect(True)

        self._fullscreen = fullscreen

        if fullscreen:
            x, y, w, h = _display_geometry(display_index)
            self.root.geometry(f"{w}x{h}+{x}+{y}")
            self.root.config(cursor="none")
        else:
            w, h = _WINDOW_SIZE
            self.root.geometry(f"{w}x{h}")

        self.width, self.height = w, h

        self.canvas = tk.Canvas(
            self.root, width=w, height=h, bg="black", highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True)

        self.root.bind("<Escape>", lambda _: self._request_quit())
        self.root.protocol("WM_DELETE_WINDOW", self._request_quit)

        self._quit_requested = False
        self._canvas_item: int | None = None
        # tkinter garbage-collects PhotoImage objects unless a reference is kept
        self._tk_image: ImageTk.PhotoImage | None = None

        # Crossfade state: blend from _prev_screen → _curr_screen over time
        self._prev_screen: Image.Image | None = None
        self._curr_screen: Image.Image | None = None
        self._crossfade_start: float | None = None

    def _request_quit(self) -> None:
        self._quit_requested = True

    def _fit_to_screen(self, pil_image: Image.Image) -> Image.Image:
        # Scale to fill the screen while preserving aspect ratio; pad with black
        img_w, img_h = pil_image.size
        scale = min(self.width / img_w, self.height / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        resized = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        out = Image.new("RGB", (self.width, self.height), (0, 0, 0))
        out.paste(resized, ((self.width - new_w) // 2, (self.height - new_h) // 2))
        return out

    def _put_image(self, img: Image.Image) -> None:
        tk_img = ImageTk.PhotoImage(img)
        if self._canvas_item is None:
            self._canvas_item = self.canvas.create_image(0, 0, anchor="nw", image=tk_img)
        else:
            self.canvas.itemconfig(self._canvas_item, image=tk_img)
        self._tk_image = tk_img  # keep reference — tkinter won't keep it alive otherwise

    def show_waiting(self) -> None:
        self._crossfade_start = None
        self._prev_screen = None
        self._curr_screen = None
        self._put_image(_waiting_image(self.width, self.height))

    def show_image(self, pil_image: Image.Image, *, crossfade: bool = False) -> None:
        fitted = self._fit_to_screen(pil_image)
        if crossfade and self._curr_screen is not None:
            self._prev_screen = self._curr_screen
            self._crossfade_start = time.monotonic()
        else:
            # No crossfade — replace instantly and clear any in-progress fade
            self._crossfade_start = None
            self._prev_screen = None
            self._put_image(fitted)
        self._curr_screen = fitted

    def _frame_tick(self, callback: Callable[[], None]) -> None:
        if self._quit_requested:
            try:
                self.root.destroy()
            except Exception:
                pass
            return

        now = time.monotonic()
        if self._crossfade_start is not None and self._prev_screen and self._curr_screen:
            progress = min(1.0, (now - self._crossfade_start) / _CROSSFADE_DURATION)
            # PIL blend is cheaper than per-pixel alpha and produces identical results
            self._put_image(Image.blend(self._prev_screen, self._curr_screen, progress))
            if progress >= 1.0:
                self._crossfade_start = None
                self._prev_screen = None
                self._put_image(self._curr_screen)

        callback()
        self.root.after(_TICK_MS, lambda: self._frame_tick(callback))

    def start(self, tick_callback: Callable[[], None]) -> None:
        """Enter the tkinter event loop. tick_callback is called every frame. Blocks until quit."""
        if self._fullscreen:
            # _hide_macos_menubar must run after the window is mapped, not in __init__
            self.root.after(100, _hide_macos_menubar)
        self.root.after(_TICK_MS, lambda: self._frame_tick(tick_callback))
        self.root.mainloop()

    def quit(self) -> None:
        try:
            self.root.destroy()
        except Exception:
            pass
