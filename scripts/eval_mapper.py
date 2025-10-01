# %%
# imports
import os
import torch
import numpy as np
import pandas as pd
from cmmr25_isudt.models.models import PlMapper
from cmmr25_isudt.datasets import White_Square_dataset
from cmmr25_isudt.utils.misc import midi2frequency, frequency2midi
from cmmr25_isudt.utils.matrix import square_over_bg_falloff
from cmmr25_isudt.utils.dsp import amp2db
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
from sklearn.neighbors import KDTree
from pythonosc import udp_client, dispatcher, osc_server

# %%
# find the latest mapper checkpoint
ckpt_path = '../ckpt/mapper'
model_version = '9'
ckpt_name = 'v' + model_version
ckpt_path = os.path.join(ckpt_path, ckpt_name)
# list files, find the one that has "last" in it
ckpt_files = [f for f in os.listdir(ckpt_path) if 'last' in f]
if len(ckpt_files) == 0:
    raise ValueError(f"No checkpoint file found in {ckpt_path} with 'last' in the name")
ckpt_file = ckpt_files[0]
ckpt_path = os.path.join(ckpt_path, ckpt_file)
print(f"Checkpoint file found: {ckpt_path}")

# %%
# load checkpoint and extract saved args
# HACK: remap old library name to new one to load old checkpoint
# sorry, this is because the model was originally trained from my "sonification" repo: https://github.com/balintlaczko/sonification
# and turns out the pickled models remember the original module name
import sys
import cmmr25_isudt
sys.modules['sonification'] = cmmr25_isudt
mapper_ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
args = mapper_ckpt["hyper_parameters"]['args']

# %%
# create mapper model from checkpoint
model = PlMapper(args)
model.load_state_dict(mapper_ckpt['state_dict'])
model.eval()
print("Mapper model created")

# %%
# create image dataset
dataset = White_Square_dataset(img_size=model.in_model.args.img_size, square_size=model.in_model.args.square_size)

# create image dataloader
batch_size=128
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
print("Image dataloader created")

# %%
# get the cuda/mps device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
# move model to device
model = model.to(device)
model.eval()
model.in_model.eval()
model.out_model.eval()
print(f"Using device: {device}")

# %%
# iterate through the dataloader and encode, map, decode, re-encode
z_1_all = torch.zeros(len(dataset), model.in_model.args.latent_size).to(device) # img encoded
z_2_all = torch.zeros(len(dataset), model.out_model.args.latent_size).to(device) # mapped
with torch.no_grad():
    for batch_idx, data in tqdm(enumerate(dataloader)):
        x, _ = data
        mu, logvar = model.in_model.model.encode(x.to(device))
        z_1 = model.in_model.model.reparameterize(mu, logvar)
        # store
        z_1_all[batch_idx * batch_size: batch_idx * batch_size + batch_size, :] = z_1
        # map to audio latent space
        z_2 = model(z_1)
        z_2_all[batch_idx * batch_size: batch_idx * batch_size + batch_size, :] = z_2


# %%
# create a scatter plot of the latent space
z_all = z_2_all.cpu().numpy()
num_dims = z_all.shape[1]
dim_pairs = list(itertools.combinations(range(num_dims), 2))
num_pairs = len(dim_pairs)
df = dataset.df

if num_pairs > 0:
    fig, axes = plt.subplots(num_pairs, 2, figsize=(20, 10 * num_pairs), squeeze=False)

    epoch_idx = ckpt_file.split('_')[-1].split('.')[0].split("=")[-1]
    fig.suptitle(f"Mapper Model v{model_version} | Epoch {epoch_idx} | Mapped Latent Space")

    for i, (dim1, dim2) in enumerate(dim_pairs):
        ax1 = axes[i, 0]
        ax2 = axes[i, 1]

        # Plot 1: Colored by x position
        sc1 = ax1.scatter(z_all[:, dim1], z_all[:, dim2], c=df['x'], cmap='viridis', s=3)
        fig.colorbar(sc1, ax=ax1, label='X Position', shrink=0.8)
        ax1.set_title(f"Latent Dims {dim1} vs {dim2} by X Position")
        ax1.set_xlabel(f"Latent Dim {dim1}")
        ax1.set_ylabel(f"Latent Dim {dim2}")
        ax1.set_aspect('equal', adjustable='box')

        # Plot 2: Colored by y position
        sc2 = ax2.scatter(z_all[:, dim1], z_all[:, dim2], c=df['y'], cmap='viridis', s=3)
        fig.colorbar(sc2, ax=ax2, label='Y Position', shrink=0.8)
        ax2.set_title(f"Latent Dims {dim1} vs {dim2} by Y Position")
        ax2.set_xlabel(f"Latent Dim {dim1}")
        ax2.set_ylabel(f"Latent Dim {dim2}")
        ax2.set_aspect('equal', adjustable='box')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# %%
# create a meshgrid of parameters to synthesize

# create ranges for each parameter
pitch_steps = 128
amp_steps = 128
pitches = np.linspace(38, 86, pitch_steps)  # MIDI note numbers
freqs = midi2frequency(pitches)
amps = np.linspace(0.01, 1.0, amp_steps)  # linear amplitude

# create a meshgrid
freqs, amps = np.meshgrid(freqs, amps)
print(f"Meshgrid shapes: freqs: {freqs.shape}, amps: {amps.shape}")

# create a dictionary where each key-value pair corresponds to a column in the dataframe
data = {
    "x": np.tile(np.arange(pitch_steps), amp_steps),
    "y": np.repeat(np.arange(amp_steps), pitch_steps),
    "pitch": frequency2midi(freqs.flatten()),
    "freq": freqs.flatten(),
    "amp": amps.flatten(),
    "db": amp2db(amps.flatten()),
}

# Create the dataframe
df_sine = pd.DataFrame(data)

# %%
# iterate the dataframe in batches and synthesize
batch_size = 128
n_batches = len(df_sine) // batch_size
print(f"Number of batches: {n_batches}, batch size: {batch_size}")
out_spec_all = torch.zeros((len(df_sine), 1, model.out_model.args.n_mels)).to(device)
with torch.no_grad():
    for i in tqdm(range(n_batches)):
        batch_df = df_sine.iloc[i * batch_size: (i + 1) * batch_size]
        # convert to tensor
        batch_z = torch.tensor(batch_df[['freq', 'amp']].values, dtype=torch.float32)
        freqs = batch_z[:, 0]
        amps = batch_z[:, 1]
        # repeat in samples dimension
        freqs = freqs.unsqueeze(1).repeat(1, model.out_model.args.length_samps).to(device)
        # synthesize
        x = model.out_model.input_synth(freqs) * amps.unsqueeze(1).to(device)
        # predict
        out_spec, mu, logvar, z = model.out_model(x)
        # store
        out_spec_all[i * batch_size: i * batch_size + batch_size, :, :] = out_spec

# %%
# fit KDTree to the rendered spectrograms
out_spec_all = out_spec_all.squeeze(1).cpu().numpy()
tree = KDTree(out_spec_all)

# %%
# create osc client
ip = "127.0.0.1"
port = 12341
client = udp_client.SimpleUDPClient(ip, port)

# function to generate an input image based on xy coordinates
def handle_pictslider(unused_addr, x, y):
    # create the image
    img = square_over_bg_falloff(x, y, model.in_model.args.img_size, model.in_model.args.square_size)
    # add a channel dimension
    img = img.unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        # encode the image
        mu, logvar = model.in_model.model.encode(img.to(device))
        # z_1 = model.in_model.model.reparameterize(mu, logvar)
        z_1 = mu # use mean only for stability
        # project to audio latent space
        z_2 = model(z_1)
        # decode the audio
        out_spec = model.out_model.model.decoder(z_2)
        # convert to numpy
        out_spec = out_spec.squeeze(1).cpu().numpy()
    # query the KD tree
    _, idx = tree.query(out_spec, k=1)
    idx = idx[0][0]
    # look up pitch and loudness from the sine dataset
    row = df_sine.iloc[idx]
    pitch = row["pitch"]
    loudness = row["db"]
    # send pitch and loudness to Max
    client.send_message("/sineparams", [pitch, loudness])

# create an OSC receiver and start it
# create a dispatcher
d = dispatcher.Dispatcher()
d.map("/pictslider", handle_pictslider)
# create a server
ip = "127.0.0.1"
port = 12342
server = osc_server.ThreadingOSCUDPServer(
    (ip, port), d)
print("Serving on {}".format(server.server_address))

try:
    server.serve_forever()
except KeyboardInterrupt:
    server.shutdown()
    server.server_close()
    print("Server stopped.")

# %%
