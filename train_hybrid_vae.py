# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from data_utils import preprocess_train_test, MEAL_COVARIATES, DEMOGRAPHICS_COVARIATES
from models import MechanisticAutoencoder, count_params
from utils import seed_everything

seed = 4
batch_size = 64
lr = 1e-2
beta_hat = 0.01
num_epochs = 100

seed_everything(seed)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

from torch.utils.data import DataLoader, TensorDataset
train_arrays, test_arrays, patient_info, (train_mean, train_std) = preprocess_train_test(seed=21, domain_adaptation=False)
train_tensors = list(map(lambda x: torch.as_tensor(x, dtype=torch.float, device=device), train_arrays))
train_dataset = TensorDataset(*train_tensors)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_tensors = list(map(lambda x: torch.as_tensor(x, dtype=torch.float, device=device), test_arrays))

# %%
G_mean = torch.as_tensor(train_mean[0], dtype=torch.float, device=device)
G_std = torch.as_tensor(train_std[0], dtype=torch.float, device=device)
def remove_scale(G, mean=G_mean, std=G_std):
    return (G - mean) / std
def add_scale(G, mean=G_mean, std=G_std):
    return G * std + mean
# %%
model = MechanisticAutoencoder(
    meal_size=len(MEAL_COVARIATES),
    demographics_size=len(DEMOGRAPHICS_COVARIATES),
    embedding_size=8,
    hidden_size=32,
    num_layers=2,
    encoder_dropout_prob=0.,
    decoder_dropout_prob=0.5,
).to(device)
print(f"Number of encoder parameters: {count_params(model)}")
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

from typing import NamedTuple
LossOutput = NamedTuple('LossOutput', [('loss', torch.Tensor), ('mse', torch.Tensor), ('kl', torch.Tensor)])

def loss_fn(output, seq_q, nonseq_q, seq_p, nonseq_p, cgm, beta_hat=beta_hat):
    states = output.states
    pred_cgm = remove_scale(states[..., 0:1])
    mse = (pred_cgm - cgm).pow(2).sum((0, 1, 2))  # sum over all 3 axes

    seq_kl = torch.distributions.kl.kl_divergence(seq_q, seq_p).sum((0, 1, 2))
    nonseq_kl = torch.distributions.kl.kl_divergence(nonseq_q, nonseq_p).sum((0, 1))

    N, T = pred_cgm.shape[:2]
    M = states.shape[-1]
    mse = mse / (N * T)
    kl = seq_kl  / (N * T) + M * nonseq_kl / (N * T)

    loss = mse + beta_hat * kl
    return LossOutput(loss=loss, mse=mse, kl=kl)

# %%
#@torch.compile
def update(model, cgm, timestamps, meals, demographics, step=True):
    optimizer.zero_grad()
    output, seq_q, nonseq_q = model(cgm, timestamps, meals, demographics)

    loss_output = loss_fn(output, seq_q, nonseq_q, model.seq_p, model.nonseq_p, cgm)
    loss = loss_output.loss
    if step:
        loss.backward()
        optimizer.step()
    return loss_output

batch_losses = []
epoch_losses = []
val_losses = []
# %%
for epoch in range(num_epochs):
    model.train()
    loss_this_epoch = 0
    for i, (cgm, timestamps, meals, demographics, _) in enumerate(train_loader):
        loss_output = update(model, cgm, timestamps, meals, demographics)
        loss = loss_output.loss
        loss_this_epoch += loss.item() * len(cgm)
        batch_losses.append(loss.item())
    loss_this_epoch /= len(train_dataset)
    with torch.no_grad():
        val_loss = update(model, *test_tensors[:-1], step=False).loss
        val_losses.append(val_loss.item())

    epoch_losses.append(loss_this_epoch)
    print(f'Epoch {epoch+1}: Train Loss: {loss_this_epoch:.3e} ==== Last Batch Loss: {loss.item():.3e} ===  MSE: {loss_output.mse.item():.3e} === KL: {loss_output.kl.item():.3e} === Val Loss: {val_loss.item():.3e}')
# %%
fig, ax = plt.subplots()
ax.plot(epoch_losses)
ax.plot(val_losses)
ax.set(yscale="log")
# %%
# %%
torch.save(model.state_dict(), "autoencoder.pt")
# %%
model.load_state_dict(torch.load("autoencoder.pt"))
# %%
def plot_states(i, model, param, states, carb_rate, meals, cgm=None):
    param, states, carb_rate, meals = param.cpu().numpy(), states.cpu().numpy(), carb_rate.cpu().numpy(), meals.cpu().numpy()
    if cgm is not None:
        cgm = cgm.cpu().numpy()
    if states.ndim == 3:
        print(f"Passed in a batch. Plotting {i}")
        param = param[i]
        states = states[i]
        carb_rate = carb_rate[i]
        reported_total = meals[i, :, 0]
        reported_carbs = meals[i, :, 1]
        if cgm is not None:
            cgm = cgm[i, :, 0]
    assert states.ndim == 2
    timesteps = np.arange(states.shape[-2]) * model.dt / 60.0 # hours
    num_dims = states.shape[-1]
    fig, axes = plt.subplots(figsize=(4,2 * num_dims), ncols=1, nrows=num_dims+1, sharey=False, layout="constrained")
    for i, field in enumerate(model.state_lims):
        ax = axes[i]
        state = states[:, i]
        ax.plot(timesteps, state, label=field)
        ax.legend()

    Iendo = param[-1] * (states[:, 0] - param[2])
    
    if cgm is not None:
        ax = axes[0]
        ax.plot(timesteps, cgm)

    ax = axes[2]
    ax2 = ax.twinx()
    ax2.stem(timesteps, reported_total, linefmt="C5-", markerfmt="", basefmt="C5")
    ax2.stem(timesteps, reported_carbs, linefmt="C3-", markerfmt="", basefmt="C3")
    ax2.plot(timesteps, carb_rate / 1000 * model.dt, color="C4")
    # TODO: plot missing carb data
    ax2.set(ylim=[0, 10])
    ax2.set(ylabel="Equivalent carbs eaten (g)")

    ax = axes[-1]  # Plot Iendo
    ax.plot(timesteps, Iendo, label="I_endo")
    ax.legend()

    axes[0].set(ylim=[0, 300])
    axes[2].set(ylim=[0, 300])
    axes[3].set(ylim=[0, 300])
    return fig, axes, timesteps
with torch.no_grad():
    model.eval()
    output, seq_q, nonseq_q = model(*train_tensors[:-1])
plot_states(5, model, output.param, output.states, output.carb_rate, train_tensors[-3], add_scale(train_tensors[0]))
# %%
def pairplot(param, diagnosis):
    import pandas as pd
    import seaborn as sns
    param = param.cpu().numpy()
    diagnosis = diagnosis.cpu().numpy()
    param_df = pd.DataFrame(param, columns=["tau_m", "Gb", "sg", "si", "p2", "mi"])
    param_df["si_mi"] = param_df["si"] * param_df["mi"]
    param_df["diagnosis"] = diagnosis
    #param_df = param_df.drop(columns=["si", "mi", "vg"])
    sns.pairplot(param_df, hue="diagnosis", corner=True, diag_kind="kde", plot_kws={"alpha": 0.5}, diag_kws={"alpha": 0.5})
with torch.no_grad():
    model.eval()
    output, seq_q, nonseq_q = model(*train_tensors[:-1])
    pairplot(output.param, train_tensors[-1])
# %%
