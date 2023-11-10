import matplotlib.pyplot as plt
import pandas as pd
from dataset_paper_repo.utils.normalize_funcs import standardize_per_catX
from utils.global_variables import DS_INFO_DICT
from utils.data_utils import set_index_fields, load_data


def load_zscores(methods,base_dir,dataset, profile_type,normalize_by_all=False,by_dose=False,z_trim=10):
# def load_zscores(methods,configs,by_dose=False):

    for m in methods.keys():
      print(f'loading zscores for method: {m}')
      if m == 'l1k':
          methods[m]['modality'] ='L1000'
        #   def load_data(base_dir,dataset, profile_type,plate_normalize_by_all = False, modality = 'CellPainting'):

          zscores, features = load_data(base_dir,dataset, profile_type,modality=methods[m]['modality'], plate_normalize_by_all =normalize_by_all)
          methods[m]['features']=list(features)

          zscores = zscores.query(f"{DS_INFO_DICT[dataset]['L1000']['role_col']} != '{DS_INFO_DICT[dataset]['L1000']['mock_val']}' ")
          
          # methods[m]['zscores'] = zscores.loc[:,methods[m]['features']]
      else:
          methods[m]['modality']='CellPainting'
          zscores = pd.read_csv(methods[m]['path'], compression = 'gzip')     
          if m == 'anomaly_emb':
              meta_features = [c for c in zscores.columns if 'Metadata_' in c]
              features = [c for c in zscores.columns if 'Metadata_' not in c]
          else:
              features = zscores.columns[zscores.columns.str.contains("Cells_|Cytoplasm_|Nuclei_")]
          methods[m]['features']=list(features)

          if normalize_by_all:
              zscores = standardize_per_catX(zscores,DS_INFO_DICT[dataset]['CellPainting']['plate_col'],methods[m]['features'])
      zscores = zscores.query(f"{DS_INFO_DICT[dataset][methods[m]['modality']]['role_col']} != '{DS_INFO_DICT[dataset][methods[m]['modality']]['mock_val']}' ")
      # zscores = zscores.query('Metadata_ASSAY_WELL_ROLE == "treated"')
      zscores = set_index_fields(zscores,dataset,by_dose=by_dose, modality=methods[m]['modality'])
      if z_trim is not None:
          zscores[features] = zscores[features].clip(-z_trim, z_trim)
      methods[m]['zscores'] = zscores.loc[:,methods[m]['features']]

    return methods



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

