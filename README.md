# Seismic Facies Interpretation

## Project Overview
This project intend to explore the use of conditioned diffusion model compared to general segmentation model like U-Net in segmenting multi-facies seismic labels.



## Data and Model Weights

Please put you data inside of `seismic_facies/data` folder. The pretrained weights of each models is already provided in `seismic_facies/models_weights`. For some big model weights, can be downloaded in the same link as dataset.
The `seismic_facies/data` folder should contain the following structure:
    
---
seismic_facies/
├── data/
│   ├── data
│   │   ├── test_once
│   │   └── train

---

p.s. data for training can be downloaded in [here](https://drive.google.com/drive/folders/1fFbanBLDqQn6Cu9QYzdRgm92YiixV1X-?usp=drive_link)


## Software Installation Guide
## Installation Instructions
```bash
# Clone project
git clone https://github.com/ada-my619/Seismic_Facies_Interpretation.git
cd Seismic_Facies_Interpretation
```

## Create and Activate a Virtual Environment
Install conda to run the following cell:
```bash
conda create -n seismic_facies-env python=3.13
conda activate seismic_facies-env
```

### Install Dependencies/the Package
```bash
pip install -e .
```

## Minimum Sample Implementation

To see a minimum working example of training and inference, it is provided in the `notebooks` folder.
