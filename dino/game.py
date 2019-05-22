import capture
import images
import pickle
import dino.score as score
import numpy as np


class RawGameCapturer(capture.Capture):
    def __init__(self, reduction_factor_x=1, reduction_factor_y=None):
        """Capture the whole game environment."""
        self.reduction_factor_x = reduction_factor_x
        self.reduction_factor_y = reduction_factor_y or reduction_factor_x
        self.capturer = capture.ScreenRegion(664, 173, 593, 102)

    def capture(self):
        """Capture the game environment."""
        return images.greyscale(self.capturer.capture())[
            :: self.reduction_factor_x, :: self.reduction_factor_y
        ]


def collect_raw_footage(backlog_size=10):
    """Collect raw images of the game being played."""
    backlog = capture.CaptureBacklog(
        capture.CaptureGroup(score=score.ScoreCapturer(2), game=RawGameCapturer(4)), 20
    )
    key_capturer = capture.KeyState("up")
    frames = []

    try:
        while True:
            await_game_start(key_capturer)
            await_game_finish(frames, backlog)
    except KeyboardInterrupt:
        save_raw_footage([frame["game"] for frame in frames])
        return None


def get_raw_footage_location():
    """Return the location of the raw footage save file."""
    return "data/rawframes.obj"


def save_raw_footage(frames):
    """Saves raw footage as a tensor under data/rawframes.obj."""
    file_handler = open(get_raw_footage_location(), "wb")
    pickle.dump(np.array(frames), file_handler)
    file_handler.close()


def load_raw_frames():
    """Load the numpy array containing the raw game frames."""
    file_handler = open(get_raw_footage_location(), "rb")
    object = pickle.load(file_handler)
    file_handler.close()
    return object


def await_game_start(up_capturer):
    """Wait until the game starts, and then return."""
    while True:
        if up_capturer.capture():
            print("Game started.")
            return None


def await_game_finish(frames, backlog):
    """Wait for the game to finish, recording frames while doing so."""
    while True:
        frames.append(backlog.update())
        if backlog.full:
            mean = np.mean(np.square(backlog.at(0)["game"] - backlog.at(-1)["game"]))
            if mean == 0.0:
                print("Game finished.")
                return None
