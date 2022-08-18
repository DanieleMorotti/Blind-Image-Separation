from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPooling2D, Conv2DTranspose, concatenate, Concatenate
from tensorflow.keras.models import Model


def encoder_block(inputs, id_layer, n_blocks, n_ch, conv_per_b, backup_layers, decoder=False):
    """
    Args:
        inputs: image in the first layer and the output of a layer in the following one.
        id_layer: (int) the number of the current block in the structure. E.g. the first block has 'id_layer' equal to 0
                        and so on.
        n_blocks: (int) the number of blocks in the encoder part of the model. Actually if 'n_blocks' is n the model
                        create n-1 encoder and decoder blocks because this number also includes the bridge.
        n_ch: (int) the number of filters to use in the Conv2D layer.
        conv_per_b: (int) the number of Conv2D layers used in each block.
        backup_layers: the list of encoder layers to connect through skip connections.
        decoder: (bool) This variable is False by default and can be set to True to avoid the MaxPooling and other operations
                        not required if the function is used in the decoder part of the model.
    Returns:
        keras.models.Model instance
    """
    h = inputs
    for i in range(conv_per_b):
        h = Conv2D(n_ch, 3, 1, padding='same')(h)
        h = BatchNormalization()(h)
        h = ReLU()(h)

    if not(decoder) and (id_layer < n_blocks-1):
        backup_layers.append(h)
        h = MaxPooling2D(padding='same')(h)
        n_ch = n_ch * 2

    return h, n_ch


def decoder_block(inputs, id_layer, n_blocks, n_ch, conv_per_b, backup_layers):
    n_ch = n_ch // 2
    h = Conv2DTranspose(n_ch, 3, 2, padding='same')(inputs)
    # Skip connection
    h = concatenate([h, backup_layers.pop(-1)])
    # Convolutions
    h, _ = encoder_block(h, id_layer, n_blocks, n_ch, conv_per_b, backup_layers, decoder=True)
    
    return h, n_ch


def build_model(input_shape, n_ch=32, blocks=4, conv_per_b=2):
    """
    Args:
        inputs_shape: (tuple) A tuple that represents the shape of the images in input.
        n_ch: (int) the number of filters to start from in the Conv2D layers.
        blocks: (int) the number of blocks in the encoder and decoder parts of the model. This number also includes the
                      bridge, so the number of actual implemented blocks is 'blocks'- 1.
        conv_per_b:  the number of Conv2D layers used in each block. By default is 2 like in the original implementation.
    Returns:
        keras.models.Model instance
    """
    x = Input(shape=input_shape)
    # Encoder part
    backup_layers = []
    h = x
    for b in range(blocks):
        # Create the encoder blocks
        h, n_ch = encoder_block(h, b, blocks, n_ch, conv_per_b, backup_layers, decoder=False)

    # Decoder part
    for b in range(blocks-1):
        h, n_ch = decoder_block(h, b, blocks, n_ch, conv_per_b, backup_layers)

    # Output
    out1 = Conv2D(1, 1, activation='relu')(h)
    out2 = Conv2D(1, 1, activation='relu')(h)
    y = Concatenate(axis=2, dtype="float32")([out1, out2])
    return Model(x, y)
