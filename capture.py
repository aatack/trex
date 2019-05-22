import pyautogui as pg
import keyboard


class Capture:
    def capture(self):
        """Capture some aspect of the current state of the computer."""
        raise NotImplementedError()


class ScreenRegion(Capture):
    def __init__(self, x, y, w, h):
        """Captures the pixels of the screen within a certain region."""
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def capture(self):
        """Capture some aspect of the current state of the computer."""
        return pg.screenshot(region=(self.x, self.y, self.w, self.h))


class KeyState(Capture):
    def __init__(self, key):
        """Records whether or not a key is pressed."""
        self.key = key

    def capture(self):
        """Capture some aspect of the current state of the computer."""
        return keyboard.is_pressed(self.key)


class CaptureGroup(Capture):
    def __init__(self, **kwargs):
        """Capture a number of different state features all in one."""
        self.captures = kwargs

    def capture(self):
        """Capture some aspect of the current state of the computer."""
        return {k: v.capture() for k, v in self.captures.items()}
