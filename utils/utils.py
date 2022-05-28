import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


def get_encoder_mask(x):
    """Outputs encoder mask to support variable-length sequences.

    args:
        input [B, max_len, input_dim]: tokenized sequences. It assumes that an empty token consists of all-zero vector.
        A mask simply indicates whether the entry is empty.
    returns:
        encoder_mask [B, max_len, max_len]: square matrix per batch indicating whether a pair of indices can interact
    """

    encoder_flat_mask = torch.sum(x, dim=-1) == 1.0

    full_mask = encoder_flat_mask[..., None] * torch.transpose(
        encoder_flat_mask[..., None], 1, 2
    )

    return full_mask


def get_decoder_mask(x):
    """Outputs decoder mask appropriate for teacher-forcing. This basically works like encoder mask, except that
    additionally, each point in a sequence is masked in a way that allows for only left-context to be available.

    """

    base_mask = get_encoder_mask(x)  #
    triangulated_mask = torch.tril(base_mask)
    return triangulated_mask


def get_enc_dec_mask(x_encoder, x_decoder):
    encoder_flat_mask = torch.sum(x_encoder, dim=-1) == 1.0
    decoder_flat_mask = torch.sum(x_decoder, dim=-1) == 1.0

    full_mask = encoder_flat_mask[..., None] * torch.transpose(
        decoder_flat_mask[..., None], 1, 2
    )

    return full_mask


def plot_grad_flow(named_parameters):
    # Code of @RoshanRane modified from
    # https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            if p.grad is None:
                ave_grads.append(torch.zeros(1))
                max_grads.append(torch.zeros(1))
            else:
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())

    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend(
        [
            Line2D([0], [0], color="c", lw=4),
            Line2D([0], [0], color="b", lw=4),
            Line2D([0], [0], color="k", lw=4),
        ],
        ["max-gradient", "mean-gradient", "zero-gradient"],
    )
    plt.show()
    # plt.savefig(
    #     os.path.join(
    #         global_state.working_dir,
    #         "extras",
    #         f"grad_flow_e={epoch}_gates.png",
    #     ),
    #     bbox_inches="tight",
    # )
