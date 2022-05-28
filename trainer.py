from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

# %% md
## Data for task #1: classify if first element in the sequence is the same as the last one
## Data for task #2: reverse the sequence
# %%
from data import get_data
## Data for task #1: classify if first element in the sequence is the same as the last one
## Data for task #2: reverse the sequence
from transformer import Transformer

input_dim = 12
E_max = 128

data_x_torch, data_y_torch_t1, data_y_torch_t2, data_x_decoder_torch_t2 = get_data()

## Sanity check the data
""" Print some samples of the data (task 1), make sure it makes sense """
idxed_data = torch.argmax(data_x_torch, dim=2)
print(idxed_data[0:10])
print(data_y_torch_t1[:10])

""" Print some samples of the data (task 2), make sure it makes sense """
padded_data_to_idx = torch.argmax(data_x_torch, dim=2)
print(padded_data_to_idx[0:2])
print(padded_data_to_idx.shape)

""" Define the dataloaders for both tasks """
import torch.utils.data as data_utils

train_dataset_t1 = data_utils.TensorDataset(data_x_torch, data_y_torch_t1)
train_dataset_t2 = data_utils.TensorDataset(data_x_torch, data_x_decoder_torch_t2, data_y_torch_t2)

train_dataloader_t1 = data_utils.DataLoader(train_dataset_t1, batch_size=16, shuffle=True)
train_dataloader_t2 = data_utils.DataLoader(train_dataset_t2, batch_size=16, shuffle=True)

""" Training loop """
task = "seq2seq"  # either `classification` for classification or `seq2seq` for outputting sequences

if task == "classification":
    train_dataloader = train_dataloader_t1
    n_epochs = 1000

    # Define the model for the first task

    transformer = Transformer(d_input=input_dim, d_output=2, max_input_seq_length=E_max, n_heads_enc=2,
                              n_heads_dec=2, n_encoders=1, n_decoders=2, d_model=128,
                              dec_type="classifier").train()

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-2, eps=1e-4)
elif task == "seq2seq":
    train_dataloader = train_dataloader_t2

    n_epochs = 1000
    # Define the model for the second task

    transformer = Transformer(d_input=input_dim, d_output=input_dim, max_input_seq_length=E_max, n_heads_enc=2,
                              n_heads_dec=2, n_encoders=1, n_decoders=1, d_model=128, d_ff=128,
                              dec_type="sequence").train()

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, eps=1e-4)
else:
    raise NotImplementedError(f"`task = {task}` is not supported.")

print(transformer)

metrics = defaultdict(list)
for epoch in (pbar := tqdm(range(n_epochs))):
    for batch in train_dataloader:
        if task == "classifier":
            x, y = batch

            out = transformer(x)

            truth_label = torch.argmax(y, dim=-1)
            preds_label = torch.argmax(out, dim=-1)

            accuracy = torch.count_nonzero(truth_label == preds_label) / truth_label.shape[0]

            CE_loss = loss_fn(out, y)

            optimizer.zero_grad()
            CE_loss.backward()
            optimizer.step()

            metrics["CE_loss"].append(CE_loss.item())
            metrics["accuracy"].append(accuracy.item())

            # plot_grad_flow(classifier_transformer.named_parameters())

            pbar.set_description(
                f"Accuracy - {np.mean(metrics['accuracy'][-10:]):.2f}, CE loss - {np.mean(metrics['CE_loss'][-10:]):.2f}")
        elif task == "seq2seq":
            x_encoder, y, x_decoder = batch

            out = transformer(x_encoder, x_decoder)

            # Take the loss only wrt present inputs
            valid_elements = (torch.sum(y, dim=-1) == 1)

            CE_loss = torch.mean(loss_fn(out.view(-1, input_dim), y.view(-1, input_dim))[valid_elements.view(-1)])

            truth_label = torch.argmax(y, dim=-1)[valid_elements]
            preds_label = torch.argmax(out, dim=-1)[valid_elements]

            accuracy = torch.count_nonzero(truth_label == preds_label) / truth_label.shape[0]

            metrics["CE_loss"].append(CE_loss.item())
            metrics["accuracy"].append(accuracy.item())

            optimizer.zero_grad()
            CE_loss.backward()
            optimizer.step()

            pbar.set_description(
                f"Accuracy - {np.mean(metrics['accuracy'][-10:]):.2f}, CE loss - {np.mean(metrics['CE_loss'][-10:]):.2f}")
