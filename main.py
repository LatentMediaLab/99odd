from __future__ import annotations

import os
import queue
import threading
from enum import Enum, auto

import anthropic
import sounddevice as sd
from dotenv import load_dotenv

from clap import CLAP_INPUT, ClapRitual
from display import DisplayManager
from generate import NameResult, ask_claude, generate_image, preload_pipeline

load_dotenv("venv/venv")

class State(Enum):
    WAITING = auto()     # idle, showing the "٩٩" screen
    GENERATING = auto()  # worker thread is running
    DISPLAYING = auto()  # image is on screen


class Installation:
    def __init__(self) -> None:
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.display = DisplayManager()
        self.state = State.WAITING
        # maxsize=1 so the worker blocks after delivering one result rather than
        # queuing up stale images while the display is busy
        self.result_queue: queue.Queue[NameResult | Exception] = queue.Queue(maxsize=1)
        self._abort = threading.Event()

    def _enter_waiting(self) -> None:
        self.state = State.WAITING
        self.display.show_waiting()

    def _start_generation(self) -> None:
        # Signal any running worker to stop, then drain the queue so the new
        # worker has a clean slot to deliver into
        self._abort.set()
        try:
            self.result_queue.get_nowait()
        except queue.Empty:
            pass
        # Replace (don't reset) the Event so the old worker's reference
        # stays set while the new worker gets a fresh cleared one
        self._abort = threading.Event()
        self.state = State.GENERATING
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self) -> None:
        """
        Runs on a daemon thread. Checks abort between the two slow steps
        (Claude API call and image generation) so a second clap can interrupt
        before the GPU work starts.
        """
        try:
            result = ask_claude(self.client)
            if self._abort.is_set():
                return
            print(
                f"[gen] {result.transliteration} ({result.meaning}) → {result.kanji}\n"
                f"[gen] {result.kanji_meaning}"
            )
            result.image = generate_image(result)
            if not self._abort.is_set():
                self.result_queue.put(result)
        except Exception as e:
            if not self._abort.is_set():
                self.result_queue.put(e)

    def _tick(self, ritual: ClapRitual) -> None:
        # A double clap is handled the same way in every state — always restart
        if ritual.double.is_set():
            ritual.double.clear()
            self._start_generation()
            return

        if self.state == State.GENERATING:
            try:
                result = self.result_queue.get_nowait()
                if isinstance(result, Exception):
                    print(f"[error] Generation failed: {result}")
                    self._enter_waiting()
                elif result.image is not None:
                    self.display.show_image(result.image, crossfade=True)
                    self.state = State.DISPLAYING
                else:
                    self._enter_waiting()
            except queue.Empty:
                pass

    def run(self) -> None:
        with ClapRitual() as ritual:
            self.display.show_waiting()
            # display.start() blocks in tkinter's mainloop;
            # _tick is called every frame via root.after()
            self.display.start(lambda: self._tick(ritual))


def main() -> None:
    load_dotenv()
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set.\n"
            "Create a .env file with: ANTHROPIC_API_KEY=sk-ant-..."
        )

    print("[99odd] Available audio input devices:")
    print(sd.query_devices())
    print(f"\n[99odd] Clap mic device index: {CLAP_INPUT!r}")

    preload_pipeline()
    print("[99odd] Ready — double clap to begin\n")

    installation = Installation()
    try:
        installation.run()
    finally:
        installation.display.quit()


if __name__ == "__main__":
    main()
