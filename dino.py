import capture
import images
import time
import pickle
import numpy as np


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
    raw_score_captures = get_raw_score_capture()

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


def load_digit_images():
    """Load the numpy array containing the digit images."""
    file_handler = open(get_digit_images_location(), "rb")
    object = pickle.load(file_handler)
    file_handler.close()
    return object


def digit_classifier():
    """Return a function that classifies images as digits."""
    digit_images = load_digit_images()

    def classify(image):
        """Classify the given image as a digit from the score line."""
        tiled = np.tile(images.greyscale(image), (10, 1, 1))
        error = np.square(tiled - digit_images)
        accumulated_errors = np.mean(error, axis=(1, 2))
        return np.argmin(accumulated_errors)

    return classify


class ScoreCapturer(capture.Capture):
    def __init__(self):
        """Create a capturer that gets the current score."""
        self.classify_digit = digit_classifier()
        self.raw_score_capturers = get_raw_score_capture()

    def capture(self):
        """Capture some aspect of the current state of the computer."""
        digits = [self.classify_digit(d) for d in self.raw_score_capturers.capture()]
        return sum(
            [
                digit * (10 ** base)
                for digit, base in zip(digits, range(len(digits))[::-1])
            ]
        )
