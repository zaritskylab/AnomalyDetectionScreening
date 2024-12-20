# Anomaly detection for high-content image-based phenotypic cell profiling

High-content image-based phenotypic profiling combines automated microscopy and analysis to identify phenotypic alterations in cell morphology and provide insight into the cell's physiological state. Classical representations of the phenotypic profile can not capture the full underlying complexity in cell organization, while recent weakly machine-learning based representation-learning methods are hard to biologically interpret. We used the abundance of control wells to learn the in-distribution of control experiments and use it to formulate a self-supervised reconstruction anomaly-based representation that encodes the intricate morphological inter-feature dependencies while preserving the representation interpretability. The performance of our anomaly-based representations was evaluated for downstream tasks with respect to two classical representations across four public Cell Painting datasets. Anomaly-based representations improved reproducibility, Mechanism of Action classification, and complemented classical representations. Unsupervised explainability of autoencoder-based anomalies identified specific inter-feature dependencies causing anomalies. The general concept of anomaly-based representations can be adapted to other applications in cell biology.

<p align="center">
<img src="figures/fig1.png" width=80%>
</p>


## Downloading data

All augmented per-well aggregated Cell Painting datasets were downloaded from the Cell Painting Gallery (CPG) (https://registry.opendata.aws/cellpainting-gallery/).

## Project setup and run:

1. Clone this repository.
2. Open cmd/shell/terminal and go to project folder: `cd AnomalyDetectionScreening`
3. Create a conda environment: `conda create -n pytorch_anomaly python=3.10.9`
4. Activate the conda environment `conda activate pytorch_anomaly`
5. Install the required packages: `pip install -r requirements.txt`
5. Run `python ads/main.py`. This script will run the anomaly detection pipeline and subsequent analyses: train the anomaly detection model, calculate percent replicating, train mechanism of action (MoA) and generate SHAP-based anomaly explanations.
6. Run additional notebooks under 'notebooks/' to generate figures and tables.


## Repository Structure

│
├── ads/
│   ├── utils/
│   │   ├── general.py
│   │   ├── configuration.py
│   │   └── logger.py
│   ├── pipeline/
│   │   ├── anomaly_pipeline.py
│   │   └── eval_pipeline.py
│   │── data/
│   │   ├── data_processing.py
|   │   └── data_utils.py
│   └── eval/
│       ├── calc_reproducibility.py
│       ├── classify_moa.py
│       |── shap_anomalies.py
|       └── eval_utils.py
│
├── configs/
│   ├── default.yaml
│   └── experiment.yaml
│
├── requirements.txt
├── main.py
└── README.md'''


## Usage

```python
import torch
from frequency_dropout import FrequencyDropout

# Initialize the layer
freq_dropout = FrequencyDropout(
    p=0.1,                # dropout probability
    preserve_energy=True, # maintain signal energy
    preserve_dc=True     # retain DC component
)

# For uncertainty estimation
model.eval()  # Keep dropout active for MC sampling
predictions = []
for _ in range(num_samples):
    pred = model(input)  # Multiple forward passes
    predictions.append(pred)
uncertainty = torch.std(torch.stack(predictions), dim=0)
```

## Repository Structure
```
frequency-dropout/
├── README.md           # This file
├── setup.py            # Package configuration
├── frequency_dropout/  # Main package
    ├── __init__.py    # Package initialization
    └── module.py      # FrequencyDropout implementation
```

## Features

- Frequency-domain dropout for uncertainty estimation
- Optional energy preservation using Parseval's theorem
- DC component preservation option
- Support for 2D and 3D inputs
- Compatible with Monte Carlo dropout inference

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{shpigler2024anomaly,
  title={Anomaly detection for high-content image-based phenotypic cell profiling},
  author={Shpigler, Alon and Kolet, Naor and Golan, Shahar and Weisbart, Erin and Zaritsky, Assaf},
  journal={bioRxiv},
  pages={2024--06},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```

## License

MIT