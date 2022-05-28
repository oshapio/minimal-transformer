from math import sqrt

import numpy as np
import torch
import torch.nn as nn

## Define the main components: Encoder, Decoder, and the Transformer

from utils.utils import get_encoder_mask, get_decoder_mask, get_enc_dec_mask

""" Attention layer.

    This layer has the following flavors:
        - Attention (used in encoder)
        - Masked attention (used in decoder)
        - Encoder-decoder attention (used in ecoder, only queries produced)
"""


class Attention(nn.Module):
    def __init__(self, d_model, d_k, d_v, d_q, type):
        super().__init__()

        self.d_model = d_model

        self.d_k = d_k
        self.d_v = d_v
        self.d_q = d_q

        # We pre-define the type mostly for didactic reasons, though this could be
        # deduced from the parameters of the `forward` function
        assert (
            type == "self_attention" or type == "enc_dec_attention"
        ), "Invalid flavor of attention given, found {type}"

        self.type = type

        # if type == "default" or type == "masked":
        self.W_Q = (
            torch.nn.Parameter(torch.rand(d_model, d_q), requires_grad=True) - 0.5
        )
        self.W_K = (
            torch.nn.Parameter(torch.rand(d_model, d_k), requires_grad=True) - 0.5
        )
        self.W_V = (
            torch.nn.Parameter(torch.rand(d_model, d_v), requires_grad=True) - 0.5
        )

        # elif type == "enc_dec":
        #     self.W_Q = torch.nn.Parameter(torch.rand(d_model, d_q), requires_grad=True)

    def forward(self, sequence_repr, mask=None, encoder_repr=None):
        """

        args:
            sequence_repr [batch_size, max_sequence_elements, token_dim]: input to the model. This can either be
            input token sequence (in the case of self-attention in the encoder), or output token sequence (in the case
            of self-attention in the decoder), or self-attended representation (in the case of encoder-decoder attention
            in the decoder, in which case `encoder_repr` is also passed)
            mask [batch_size, max_seq_len, token_dim]: a mask indicating presence of elements (in the case of encoder)
            and additionally indicating access of context (in sequence deocder case, only context from the left is
            available).
        """

        if self.type == "self_attention":
            Q, K, V = (
                sequence_repr @ self.W_Q,
                sequence_repr @ self.W_K,
                sequence_repr @ self.W_V,
            )
        elif self.type == "enc_dec_attention":
            # Only queries are computed from the output sequence [batch_size, max_output_seq_len, d_q]
            Q = sequence_repr @ self.W_Q
            # While keys and values are computed from the encoder representation
            K = encoder_repr @ self.W_K
            V = encoder_repr @ self.W_V

        attention_weights = Q @ torch.transpose(K, 1, 2)

        # Mask the sequences that are shorter than the bounds
        if mask is not None:
            attention_weights[~mask] = -1e9

        attention_matrix = torch.softmax(
            attention_weights / sqrt(self.d_k), dim=2
        )  # B x max_seq x max_seq

        attended_vals = attention_matrix @ V  # B x max_seq x d_v

        return attended_vals


# %%
""" Multi-head attention layer.
    This runs multiple attention blocks independently (ideally in parallel, not done here),
    then stacks the representations and produces representation matrix for the sequence. 

    This layer has the following flavors:
        - Multi-head attention (used in encoder)
        - Masked multi-head attention (used in decoder)
        - Encoder-decoder multi-head attention (used ind ecoder, only queries produced)
 """


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads=8, d_model=512, type="self_attention"):
        super().__init__()

        self.n_heads = n_heads
        self.d_model = d_model

        # d_k, d_v, d_q are taken as in the paper equaling to d_model / n_heads
        self.d_k = self.d_v = self.d_q = (
            d_model // n_heads
        )  # , d_model // n_heads, d_model // n_heads

        self.type = type
        assert (
            type == "self_attention" or type == "enc_dec_attention"
        ), f"Invalid flavor of attention given, found {type}"

        self.attention_heads = []
        for i in range(n_heads):
            self.attention_heads.append(
                Attention(self.d_model, self.d_k, self.d_v, self.d_q, type=type)
            )

        self.C = (
            torch.nn.Parameter(
                torch.rand(n_heads * self.d_v, d_model), requires_grad=True
            )
            - 0.5
        )

    def forward(self, sequence, mask=None, encoder_repr=None):
        """
        args:
            input_sequence [batch_size, max_sequence_elements, token_dim]
            K [batch_size, d_k], is only given in encoder-decoder attention
            V [batch_size, d_v], is only given in encoder-decoder attention
            encoder_mask [batch_size, max_sequence_length]:
        returns:
            pooled_attention [batch_size, max_sequence_elements, d_v]
        """

        representations = []

        for i in range(self.n_heads):
            representation_i = self.attention_heads[i](
                sequence, mask=mask, encoder_repr=encoder_repr
            )
            representations.append(representation_i)

        representations_cated = torch.cat(representations, dim=2)

        pooled_representation = representations_cated @ self.C

        return pooled_representation


class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.input_dim = input_dim
        self.d_ff = d_ff

        self.layer1 = torch.nn.Linear(input_dim, d_ff)
        self.layer2 = torch.nn.Linear(d_ff, d_model)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.layer2(self.relu(self.layer1(x)))


"""
Encoder module. The main goal here is to acquire context representations at each position.
This is achieved by passing input embeddings (optionally with positional encoding) through 
attention and feed-forward layers.
"""


class TransformerEncoder(nn.Module):
    def __init__(self, n_heads=8, d_model=512, d_ff=128):
        super().__init__()
        self.d_k = self.d_v = self.d_q = d_model // n_heads

        self.self_attention = MultiHeadAttention(
            n_heads, d_model, type="self_attention"
        )

        self.layer_norm1 = torch.nn.LayerNorm(d_model)
        self.feed_forward = FeedForwardNetwork(d_model, d_model, d_ff)

        self.layer_norm2 = torch.nn.LayerNorm(d_model)

        # Only relevant for the last encoder
        # if output_enc_dec_K_V:
        #     # Matrix for outputting keys and values for encoder-decoder attention
        #     self.enc_dec_W_K, self.enc_dec_W_V = torch.nn.Parameter(
        #         torch.rand((d_model, d_k)), requires_grad=True
        #     ), torch.nn.Parameter(torch.rand((d_model, d_v)), requires_grad=True)

    def forward(self, x, encoder_mask=None):
        """
        args:
            x [batch_size, max_sequence_length, d_model]

        returns:
            z [batch_size, max_sequence_length, d_model]: the representations of all the elements
        """

        attended_inputs = self.self_attention(x, mask=encoder_mask)
        layer_normed1 = self.layer_norm1(attended_inputs + x)
        feed_forwarded = self.feed_forward(layer_normed1)
        layer_normed2 = self.layer_norm2(feed_forwarded + layer_normed1)

        return layer_normed2


"""
Decoder module in case only a classification of the whole sequence needs to be produced.
In this case, this is pretty much just a linear head. The only interesting part about this
is that it first averages out all the states, and then makes the classification. 
"""


class TransformerClassificationDecoder(nn.Module):
    def __init__(self, d_model, d_output):
        super().__init__()
        self.d_output = d_output

        self.linear = torch.nn.Linear(d_model, d_output)

    def forward(self, states, encoder_mask=None):
        """
        Average out the non-masked states and return classify it
        args:
            input [batch_size, max_sequence_length, d_model]
            encoder_repr [batch_size, max_sequence_length, d_model]

        """
        if encoder_mask is None:
            # Average out the non-masked representations
            averaged_state = torch.mean(states, dim=1)  # B x d_model
        else:
            # Average out the masked representations;
            # the sum determines the positions that are masked
            active_states = encoder_mask[..., 0:1]
            averaged_state = torch.sum((active_states * states), dim=1) / torch.sum(
                active_states, dim=1
            )  #
        lineared_avg_state = self.linear(averaged_state)  # B x d_output
        return lineared_avg_state


"""
Decoder module in case whole sequence needs to be produced.
"""


class TransformerSequenceDecoder(nn.Module):
    def __init__(
        self,
        d_model,
        d_output,
        max_seq_len,
        n_heads_self_attn=1,
        n_heads_enc_dec_attn=1,
        d_ff=128,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_output = d_output
        self.max_seq_len = max_seq_len

        self.self_attention = MultiHeadAttention(
            n_heads_self_attn, d_model, type="self_attention"
        )
        self.layer_norm1 = torch.nn.LayerNorm(d_model)

        self.enc_dec_attention = MultiHeadAttention(
            n_heads_enc_dec_attn, d_model, type="enc_dec_attention"
        )
        self.layer_norm2 = torch.nn.LayerNorm(d_model)

        self.feed_forward = FeedForwardNetwork(d_model, d_model, d_ff)
        self.layer_norm3 = torch.nn.LayerNorm(d_model)

    def forward(
        self, decoded_repr, encoded_repr, mask_self_attn=None, mask_enc_dec_attn=None
    ):
        """
        args:
            x [batch_size, max_seq_len, d_model]: position-encoded representation of the output sequence
            mask [batch_size, max_seq_len, d_model]:
        """

        # causal self-attention, where only the context from the left is visible
        self_attended_output = self.self_attention(decoded_repr, mask_self_attn)
        layer_normed1 = self.layer_norm1(self_attended_output + decoded_repr)

        enc_dec_attended = self.enc_dec_attention(
            layer_normed1, mask_enc_dec_attn, encoder_repr=encoded_repr
        )
        layer_normed2 = self.layer_norm2(enc_dec_attended + layer_normed1)

        feed_forwarded = self.feed_forward(layer_normed2)
        layer_normed3 = self.layer_norm3(feed_forwarded + layer_normed2)

        return layer_normed3

    # def forward(self, prev_inputs=[]):
    #     pass


""" Positional encoding """


class PositionalEncoder(nn.Module):
    def __init__(self, max_sequence_length, d_model):
        super().__init__()

        self.periodic_signal = torch.zeros((max_sequence_length, d_model))

        # max_seq x d_model
        pos_idx = np.repeat(
            np.linspace(0.0, max_sequence_length - 1, max_sequence_length)[:, None],
            d_model,
            axis=1,
        )
        # max_seq x d_model
        i_idx = np.repeat(
            np.linspace(0.0, d_model - 1, d_model)[None, :], max_sequence_length, axis=0
        )

        sines = np.sin(pos_idx / (10000 ** (i_idx / d_model)))
        cosines = np.cos(pos_idx / (10000 ** (i_idx / d_model)))

        joint_pos = np.empty_like(sines)
        joint_pos[:, ::2] = sines[:, ::2]
        joint_pos[:, 1::2] = cosines[:, ::2]

        self.joint_pos = torch.from_numpy(joint_pos).float()

    def forward(self, x):
        return x + self.joint_pos


""" Main Transformer module """


class Transformer(nn.Module):
    def __init__(
        self,
        d_input,
        d_output,
        max_input_seq_length,
        max_output_seq_length=None,
        n_heads_enc=8,
        n_heads_dec=8,
        n_encoders=6,
        n_decoders=6,
        d_model=512,
        d_ff=128,
        dec_type="classifier",
    ):
        """
        args:
            dec_type: type of decoder to use: either `classifier` or `sequence`
        """
        super().__init__()

        self.d_input = d_input
        self.d_output = d_output

        if d_input != d_output:
            raise NotImplementedError(
                "Target sequences with dimensionality different than input sequences are not"
                "supported."
            )

        self.n_encoders = n_encoders
        self.n_decoders = n_decoders

        self.n_heads_enc = n_heads_enc
        self.n_heads_dec = n_heads_dec

        self.dec_type = dec_type

        self.d_model = d_model

        self.input_embedding = nn.Linear(d_input, d_model, bias=False)
        self.output_embedding = nn.Linear(d_model, d_input)

        # if d_input == d_output: # TODO: fix this; for now just pass transposed output sequence to the input embedding
        #     self.output_embedding.weight = self.input_embedding.weight.T
        #     self.output_embedding.bias = self.input_embedding.bias

        self.max_input_seq_length = max_input_seq_length
        self.input_positional_encoding = PositionalEncoder(
            max_sequence_length=max_input_seq_length, d_model=d_model
        )

        # If `max_output_seq_length` is not provided, we take it to be idenntical to the input length
        self.max_output_seq_length = max_output_seq_length or max_input_seq_length

        if dec_type == "sequence":
            self.output_positional_encoding = PositionalEncoder(
                max_sequence_length=self.max_output_seq_length, d_model=d_model
            )

        self.encoder_stack = nn.ModuleList([])
        self.decoder_stack = nn.ModuleList([])

        # Encoder stack
        for i in range(n_encoders):
            encoder_i = TransformerEncoder(n_heads_enc, d_model=self.d_model, d_ff=d_ff)
            self.encoder_stack.append(encoder_i)

        assert (
            dec_type == "classifier" or dec_type == "sequence"
        ), f"Unrecognized decoder block type given! Found {dec_type}"
        if dec_type == "classifier" and n_decoders != 1:
            raise NotImplementedError(
                "Using multiple decoding heads in classification mode is not supported"
            )

        # Decoder stack
        for i in range(n_decoders):
            if dec_type == "classifier":
                decoder_i = TransformerClassificationDecoder(
                    d_model=d_model, d_output=d_output
                )
            elif dec_type == "sequence":
                decoder_i = TransformerSequenceDecoder(
                    d_model=d_model,
                    d_output=d_output,
                    max_seq_len=max_input_seq_length,
                    d_ff=d_ff,
                )
                # raise NotImplementedError()
            self.decoder_stack.append(decoder_i)

    def forward(self, x_encoder, x_decoder=None):
        """
        args:
            x [batch_size, max_sequence_length, d_input]
        returns:
            lineared (if the transformer type is `classifier` then [batch_size, d_output], (if type is `sequence`
            then [batch_size, max_seq_len, d_output]
        """
        if self.dec_type == "sequence" and x_decoder is None:
            raise ValueError("In seq2seq setting `x_decoder` cannot be `None`")

        # Figure out the mask in the encoder based on the input sequence.
        # This is only to allow for variable-length sequences

        encoder_mask = get_encoder_mask(x_encoder)

        embedded_input = self.input_embedding(x_encoder)  # B x max_seq_len x d_model
        pos_encoded_input = self.input_positional_encoding(embedded_input)

        encoded_repr = pos_encoded_input
        # Encode with all the encoders
        for encoder_i in range(self.n_encoders):
            encoded_repr = self.encoder_stack[encoder_i](
                encoded_repr, encoder_mask=encoder_mask
            )

        if self.dec_type == "classifier":
            lineared = self.decoder_stack[0](encoded_repr, encoder_mask=encoder_mask)
        elif self.dec_type == "sequence":
            embedded_output = x_decoder @ self.input_embedding.weight.T  #   + self.input_embedding.bias #   self.input_embedding(x_decoder.transpose(1,2)).transpose(1,2) # to reuse the input embedding weight
            pos_encoded_output = self.output_positional_encoding(embedded_output)

            # Prepare the inputs and do teacher-forcing
            # [batch_size, max_target_seq_len, max_target_seq_len]
            decoder_mask = get_decoder_mask(x_decoder)
            enc_dec_mask = get_enc_dec_mask(x_encoder, x_decoder)
            decoded_repr = pos_encoded_output

            for decoder_i in range(self.n_decoders):
                decoded_repr = self.decoder_stack[decoder_i](
                    decoded_repr,
                    encoded_repr,
                    mask_self_attn=decoder_mask,
                    mask_enc_dec_attn=enc_dec_mask,
                )

            # Pass through Linear (transpose of the input embedding weight matrix)
            lineared = self.output_embedding(decoded_repr)
        return lineared
