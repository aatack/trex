import capture
import images
import time
import pickle
import numpy as np


class RawScoreCapturer(capture.Capture):
    def __init__(self, reduction_factor_x=1, reduction_factor_y=None):
        """Capture the pixels representing the game score."""
        self.reduction_factor_x = reduction_factor_x
        self.reduction_factor_y = reduction_factor_y or reduction_factor_x
        self.capturer = capture.ScreenRegion(1193, 147, 55, 13)

    def capture(self):
        """Capture the pixels representing the game score."""
        pixels = images.greyscale(self.capturer.capture())
        return [
            pixels[
                :: self.reduction_factor_y,
                i * 11 : (i + 1) * 11 : self.reduction_factor_x,
            ]
            for i in range(5)
        ]


def get_raw_score_capture():
    """Return a capture instances that screenshots the current score."""
    return capture.CaptureList(
        *[capture.ScreenRegion(1193 + 11 * i, 147, 11, 13) for i in range(5)]
    )


def get_digit_images_location():
    return "data/digit_images.obj"


def collect_digit_images(n):
    """Collects the images to be used to determine scores."""
    time.sleep(2)
    digits = []
    digit_images = [None] * 10
    raw_score_captures = RawScoreCapturer()

    for _ in range(n):
        raw = raw_score_captures.captures[-1].capture()
        digits.append(images.greyscale(raw))
        time.sleep(0.1)

    for digit in digits:
        if len([i for i in digit_images if i is None]) == 0:
            break

        images.show(digit)
        user_input = input("Save? ")
        if user_input in "0123456789" and len(user_input) > 0:
            digit_images[int(user_input)] = digit
        elif user_input == "end":
            break

    file_handler = open(get_digit_images_location(), "wb")
    pickle.dump(np.array(digit_images), file_handler)
    file_handler.close()


def load_digit_images(reduction_factor_x, reduction_factor_y):
    """Load the numpy array containing the digit images."""
    file_handler = open(get_digit_images_location(), "rb")
    object = pickle.load(file_handler)
    file_handler.close()
    return object[:, ::reduction_factor_x, ::reduction_factor_y]


def digit_classifier(reduction_factor_x, reduction_factor_y):
    """Return a function that classifies images as digits."""
    digit_images = load_digit_images(reduction_factor_x, reduction_factor_y)

    def classify(image):
        """Classify the given image as a digit from the score line."""
        tiled = np.tile(image, (10, 1, 1))
        error = np.square(tiled - digit_images)
        accumulated_errors = np.mean(error, axis=(1, 2))
        return np.argmin(accumulated_errors)

    return classify


class ScoreCapturer(capture.Capture):
    def __init__(self, reduction_factor_x=1, reduction_factor_y=None):
        """Create a capturer that gets the current score."""
        self.reduction_factor_x = reduction_factor_x
        self.reduction_factor_y = reduction_factor_y or reduction_factor_x

        self.classify_digit = digit_classifier(
            self.reduction_factor_x, self.reduction_factor_y
        )
        self.raw_score_capture = RawScoreCapturer(
            self.reduction_factor_x, self.reduction_factor_y
        )

    def capture(self):
        """Capture some aspect of the current state of the computer."""
        digits = [
            self.classify_digit(digit) for digit in self.raw_score_capture.capture()
        ]
        return sum(
            [
                digit * (10 ** base)
                for digit, base in zip(digits, range(len(digits))[::-1])
            ]
        )
