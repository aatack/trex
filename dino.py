import capture
import images
import time
import pickle
import numpy as np


score_capture = capture.CaptureList(
    *[capture.ScreenRegion(1193 + 11 * i, 147, 11, 13) for i in range(5)]
)

digit_images_location = "data/digit_images.obj"


def collect_digit_images(n):
    """Collects the images to be used to determine scores."""
    time.sleep(2)
    digits = []
    digit_images = [None] * 10

    for _ in range(n):
        raw = score_capture.captures[-1].capture()
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

    file_handler = open(digit_images_location, "wb")
    pickle.dump(np.array(digit_images), file_handler)
    file_handler.close()


def load_digit_images():
    """Load the numpy array containing the digit images."""
    file_handler = open(digit_images_location, "rb")
    object = pickle.load(file_handler)
    file_handler.close()
    return object
