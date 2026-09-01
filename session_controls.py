import msvcrt
import threading
import time


class InterruptHandler:
    """Detect Enter/Space during tutor speech to skip playback."""

    def __init__(self):
        self._requested = threading.Event()
        self._active = threading.Event()
        self._thread = threading.Thread(target=self._poll_keyboard, daemon=True)
        self._thread.start()

    def _poll_keyboard(self):
        while True:
            if self._active.is_set() and msvcrt.kbhit():
                key = msvcrt.getch()
                if key in (b"\r", b"\n", b" "):
                    self._requested.set()
            time.sleep(0.03)

    def arm(self):
        self._requested.clear()
        self._active.set()

    def disarm(self):
        self._active.clear()
        self._requested.clear()

    def check(self) -> bool:
        if self._requested.is_set():
            self._requested.clear()
            return True
        return False
