# Anomaly detection for high-content image-based phenotypic cell profiling

High-content image-based phenotypic profiling combines automated microscopy and analysis to identify phenotypic alterations in cell morphology and provide insight into the cell's physiological state. Classical representations of the phenotypic profile can not capture the full underlying complexity in cell organization, while recent weakly machine-learning based representation-learning methods are hard to biologically interpret. We used the abundance of control wells to learn the in-distribution of control experiments and use it to formulate a self-supervised reconstruction anomaly-based representation that encodes the intricate morphological inter-feature dependencies while preserving the representation interpretability. The performance of our anomaly-based representations was evaluated for downstream tasks with respect to two classical representations across four public Cell Painting datasets. Anomaly-based representations improved reproducibility, Mechanism of Action classification, and complemented classical representations. Unsupervised explainability of autoencoder-based anomalies identified specific inter-feature dependencies causing anomalies. The general concept of anomaly-based representations can be adapted to other applications in cell biology.

<p align="center">
<img src="figures/fig1.png" width=80%>
</p>


## Downloading data

All augmented per-well aggregated Cell Painting datasets were downloaded from the Cell Painting Gallery (CPG) (https://registry.opendata.aws/cellpainting-gallery/). To download the datasets used in the paper, run:

`aws s3 cp --no-sign-request s3://cellpainting-gallery/cpg0003-rosetta/broad/workspace/ <YOUR_PATH>`

For installation of AWS CLI, see https://docs.aws.amazon.com/cli/v1/userguide/install-windows.html.

## Project setup and run:

1. Clone this repository.
2. Open cmd/shell/terminal and go to project folder: `cd AnomalyDetectionScreening`
3. Create a conda environment: `conda create -n pytorch_anomaly python=3.10.9`
4. Activate the conda environment `conda activate pytorch_anomaly`
5. Install the required packages: `pip install -r requirements.txt`
6. Configure parameters through .yaml file in `/configs`
5. Run `python main.py --flow train --config configs/experiment.yaml`. This script will train the anomaly detection model.
6. Run `python main.py --flow eval`. This script will run the subsequent analyses: calculate percent replicating, train mechanism of action (MoA) and generate SHAP-based anomaly explanations.


## Repository Structure

```
AnomalyDetectionScreening/
├── README.md           # This file
├── main.py           # main_script
├── requirement.txt            # Package requirements
├── configs/            # directory for .yaml configuration files
├── src/  # Main package
    ├── __init__.py     # Package initialization
    ├── data/           #  directory for data related ops
    ├── model/          #  directory for model related ops
    ├── eval/           #  directory for eval related ops
    ├── utils/          #  directory for utils
    └── ProfilingAnomalyDetector.py      #  anomaly detection module implementation
    
```


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