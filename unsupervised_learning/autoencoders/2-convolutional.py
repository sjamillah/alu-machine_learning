#!/usr/bin/env python3
"""
Defines function that creates a convolutional autoencoder
"""


import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder
    """
    if type(input_dims) is not tuple:
        raise TypeError(
            "input_dims must be tuple of ints containing dimensions of \
            model input")
    for dim in input_dims:
        if type(dim) is not int:
            raise TypeError("input_dims must be tuple of ints containing \
            dimensions of model input")
    if type(filters) is not list:
        raise TypeError("filters must be a list of ints \
        representing number of filters for each convolutional layer")
    for number_filter in filters:
        if type(number_filter) is not int:
            raise TypeError("filters must be a list of ints \
            representing number of filters for each convolutional layer")
    if type(latent_dims) is not tuple:
        raise TypeError("latent_dims must be an int containing \
        dimensions of latent space representation")
    for dim in latent_dims:
        if type(dim) is not int:
            raise TypeError("latent_dims must be an int containing \
            dimensions of latent space representation")

    # encoder
    encoder_inputs = keras.Input(shape=(input_dims))
    encoder_value = encoder_inputs
    for i in range(len(filters)):
        encoder_layer = keras.layers.Conv2D(filters[i],
                                            activation='relu',
                                            kernel_size=(3, 3),
                                            padding='same')
        encoder_value = encoder_layer(encoder_value)
        encoder_pooling_layer = keras.layers.MaxPooling2D((2, 2),
                                                          padding='same')
        encoder_value = encoder_pooling_layer(encoder_value)
    encoder_outputs = encoder_value
    encoder = keras.Model(inputs=encoder_inputs, outputs=encoder_outputs)

    # decoder
    decoder_inputs = keras.Input(shape=(latent_dims))
    decoder_value = decoder_inputs
    for i in range(len(filters) - 1, 0, -1):
        decoder_layer = keras.layers.Conv2D(filters[i],
                                            activation='relu',
                                            kernel_size=(3, 3),
                                            padding='same')
        decoder_value = decoder_layer(decoder_value)
        decoder_upsample_layer = keras.layers.UpSampling2D((2, 2))
        decoder_value = decoder_upsample_layer(decoder_value)
    decoder_last_layer = keras.layers.Conv2D(filters[0],
                                             kernel_size=(3, 3),
                                             padding='valid',
                                             activation='relu')
    decoder_value = decoder_last_layer(decoder_value)
    decoder_upsample_layer = keras.layers.UpSampling2D((2, 2))
    decoder_value = decoder_upsample_layer(decoder_value)
    decoder_output_layer = keras.layers.Conv2D(input_dims[2],
                                               activation='sigmoid',
                                               kernel_size=(3, 3),
                                               padding='same')
    decoder_outputs = decoder_output_layer(decoder_value)
    decoder = keras.Model(inputs=decoder_inputs, outputs=decoder_outputs)

    # autoencoder
    inputs = encoder_inputs
    auto = keras.Model(inputs=inputs, outputs=decoder(encoder(inputs)))
    auto.compile(optimizer='adam',
                 loss='binary_crossentropy')

    return encoder, decoder, auto
