import numpy as np
import matplotlib.pyplot as plt

from transformer import PositionalEncoder


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model
    )

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return pos_encoding.squeeze()


pos_enc = PositionalEncoder(max_sequence_length=10, d_model=64).joint_pos.numpy()

plt.imshow(pos_enc)
plt.show()

# Comparing with the "ground truth" from
# https://github.com/jalammar/jalammar.github.io/blob/master/notebookes/transformer/transformer_positional_encoding_graph.ipynb
ref_positional_encoding = positional_encoding(position=10, d_model=64)

plt.imshow(ref_positional_encoding)
plt.show()
