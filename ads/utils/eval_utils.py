import matplotlib.pyplot as plt
import pandas as pd
from utils.global_variables import DS_INFO_DICT
from utils.data_utils import set_index_fields, load_data


def plot_latent_effect(val_scores, latent_dims):

  # latent_dims = sorted(k for k in model_dict)
  # val_scores = [model_dict[k]["result"]["val"][0]["test_loss"] for k in latent_dims]

  fig = plt.figure(figsize=(6, 4))
  plt.plot(
    latent_dims, val_scores, "--", color="#000", marker="*", markeredgecolor="#000", markerfacecolor="y",
    markersize=16
  )

  plt.xscale("log")
  plt.xticks(latent_dims, labels=latent_dims)
  plt.title("Reconstruction error over latent dimensionality", fontsize=14)
  plt.xlabel("Latent dimensionality")
  plt.ylabel("Reconstruction error")
  plt.minorticks_off()
  plt.ylim(0, 1)
  plt.show()

