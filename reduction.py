from keras.layers import Dense, Flatten
from keras.models import Sequential, load_model
import os


def make_encoder(
    feature_dimension, latent_dimension, inner_layers, latent_activation="tanh"
):
    """Create an encoder."""
    encoder = Sequential()
    first_layer = True

    for units, activation in (
        inner_layers if isinstance(inner_layers, list) else [inner_layers]
    ) + [(latent_dimension, latent_activation)]:
        if first_layer:
            first_layer = False
            encoder.add(
                Dense(units, activation=activation, input_shape=(feature_dimension,))
            )
        else:
            encoder.add(Dense(units, activation=activation))
    return encoder


def make_decoder(
    latent_dimension,
    feature_dimension,
    inner_layers,
    reconstruction_activation="sigmoid",
):
    """Create a decoder."""
    return make_encoder(
        latent_dimension,
        feature_dimension,
        inner_layers,
        latent_activation=reconstruction_activation,
    )


def make_mirrored_autoencoder(
    feature_dimension,
    latent_dimension,
    inner_layers,
    inner_activation="relu",
    latent_activation="tanh",
    reconstruction_activation="sigmoid",
):
    """Create an autoencoder with mirrored layers."""
    layers = [
        (layer, inner_activation)
        for layer in (
            inner_layers if isinstance(inner_layers, list) else [inner_layers]
        )
    ]
    encoder = make_encoder(
        feature_dimension, latent_dimension, layers, latent_activation=latent_activation
    )
    decoder = make_decoder(
        latent_dimension,
        feature_dimension,
        layers[::-1],
        reconstruction_activation=reconstruction_activation,
    )
    return Sequential([encoder, decoder])


def save_autoencoder(autoencoder, name):
    """Save the given autoencoder."""
    save_encoder(autoencoder.layers[0], name)
    save_decoder(autoencoder.layers[1], name)


def save_encoder(encoder, name):
    """Save the given encoder."""
    directory = "data/models/{}/".format(name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    encoder.save(directory + "encoder.h5")


def save_decoder(decoder, name):
    """Save the given decoder."""
    directory = "data/models/{}/".format(name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    decoder.save(directory + "decoder.h5")


def load_autoencoder(name):
    """Load an autoencoder with the given name."""
    return Sequential([load_encoder(name), load_decoder(name)])


def load_encoder(name, trainable=True):
    """Load an encoder and optionally make it untrainable."""
    encoder = load_model("data/models/{}/encoder.h5".format(name))
    encoder.trainable = trainable
    return encoder


def load_decoder(name):
    """Load a decoder."""
    return load_model("data/models/{}/decoder.h5".format(name))


def product(values):
    """Calculate the product of a list or tuple of integers."""
    if isinstance(values, int):
        return values
    total = 1
    for value in values:
        total *= value
    return total
