import capture
import images


class RawGameCapturer(capture.Capture):
    def __init__(self, reduction_factor_x=1, reduction_factor_y=None):
        """Capture the whole game environment."""
        self.reduction_factor_x = reduction_factor_x
        self.reduction_factor_y = reduction_factor_y or reduction_factor_x
        self.capturer = capture.ScreenRegion(664, 173, 593, 102)

    def capture(self):
        """Capture the game environment."""
        return images.pixels(self.capturer.capture())[
            :: self.reduction_factor_x, :: self.reduction_factor_y, :
        ]
