# Anomaly-Based Cell Profiling

A PyTorch implementation for anomaly detection in cellular images, based on [this paper](https://www.biorxiv.org/content/10.1101/2024.06.01.595856v1). The module learns the "normal" patterns from control experiments using self-supervised reconstruction, enabling fast and efficient detection and interpretation of cellular anomalies.

### Key Features
- Self-supervised learning from control data
- Complex morphological pattern recognition
- Interpretable anomaly detection
- Fast training and inference times
- PyTorch-based implementation

<br>

<p align="center">
<img src="figures/fig1.png" width=80%>
</p>

---

## Downloading data

All augmented per-well aggregated Cell Painting datasets were downloaded from the Cell Painting Gallery (CPG) (https://registry.opendata.aws/cellpainting-gallery/). To download the datasets used in the paper, run:

`aws s3 cp --no-sign-request s3://cellpainting-gallery/cpg0003-rosetta/broad/workspace/ <YOUR_PATH>`

AWS CLI is required for donwnload, for installation see https://docs.aws.amazon.com/cli/v1/userguide/install-windows.html.

## Setup Instructions

### Quick Start
```bash
# 1. Clone and navigate
git clone [repository-url]
cd AnomalyDetectionScreening

# 2. Create and activate environment
conda create -n pytorch_anomaly python=3.10.9
conda activate pytorch_anomaly

# 3. Install dependencies
pip install -r requirements.txt

# 4. install the package in development mode
pip install -e .
```

### Configuration
Configure your experiment through YAML files in /configs:
- Use `/configs/default_config.yaml` for reference of all parameters
- Create custom config `/configs/<YOUR_CONFIG>.yaml` to override defaults
- Set required `base_dir=<YOUR_PATH>` for data location
- Output paths:
  - Output repsentations: output_dir (default: base_dir/anomaly_output/)
  - Results: res_dir (default: base_dir/results)

Parameters can be set through:
- Command line arguments (highest priority)
- Custom config file
- Default config file (lowest priority)

### Run

```bash
# Train anomaly detection model
python main.py --flow train --exp_name <exp_name> --config configs/<config>.yaml

# Evaluate results (calculates replication %, MoA classification, SHAP explanations)
python main.py --exp_name <exp_name> --flow eval
```

---

## Repository Structure

```
AnomalyDetectionScreening/
├── README.md                   # This file
├── main.py                     # main_script
├── requirement.txt             # Package requirements
├── configs/                    # directory for .yaml configuration files
├── notebooks/                  # directory for .yaml configuration files
    ├── analyze_moa_res.ipynb               # Visualize and compare MoA results 
    └── interpret_feature_dists.ipynb       #  Analysis of top features by compound \ MoA st 
├── src/                        # Main package
    ├── __init__.py                         # Package initialization
    ├── data/                               #  directory for data related ops
    ├── model/                              #  directory for model related ops
    ├── eval/                               #  directory for eval related ops
    ├── utils/                              #  directory for utils
    └── ProfilingAnomalyDetector.py         #  anomaly detection module implementation
    
```

---
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