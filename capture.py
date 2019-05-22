import pyautogui as pg
import keyboard


class CaptureBacklog:
    def __init__(self, capturer, backlog):
        """Creates a backlog of the last n time steps of captures."""
        self.capturer = capturer
        self.backlog = backlog
        self.full = False

        self.captures = []
        self.index = 0

    def update(self):
        """Take a new capture and replace any old ones."""
        new_capture = self.capturer.capture()
        if not self.full:
            self.captures.append(new_capture)
            self.full = len(self.captures) >= self.backlog
        else:
            for i in range(self.backlog - 1):
                self.captures[i] = self.captures[i + 1]
            self.captures[-1] = new_capture
        return new_capture

    def at(self, step):
        """Index the backlog; 0 is the current step, -1 the one before, and so on."""
        return self.captures[step - 1]


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


class CaptureList(Capture):
    def __init__(self, *args):
        """Capture an ordered list of different state features at once."""
        self.captures = args

    def capture(self):
        """Capture some aspect of the current state of the computer."""
        return [capture.capture() for capture in self.captures]
