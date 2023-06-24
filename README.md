# A Framework for Global Sensitivity Analysis with Polynomial Chaos Expansion on the Grassmann Manifold

This Git repository accompanies the paper titled "Global Sensitivity Analysis using Polynomial Chaos Expansion on the Grassmann Manifold".

## General Overview

## Contents
Files in the `notebooks` folder:

- `GSA_results`
- `input_data`
- `pce_accuracy`
- `plots`
- `DeepABM_PCE-GSA.ipynb`
- `DeepABM_SobolGSA.ipynb`
- `GDMaps_PCE_LV.ipynb`
- `GDMaps_unit_sphere.ipynb`

`Snellius_DeepABM` folder is ... and contains:
- `main_uq_no_interventions.py`
- `model_uq_no_interventions.py`

## Data
To run with the original data, download it from [https://figshare.com/articles/dataset/data_zip/23515965](https://figshare.com/articles/dataset/output_data_zip/22216921) and add the unzipped folder named `output_data` into `notebooks` folder.

## Prerequisites and Installation

1. Clone the repository:
```
git clone git@github.com:bazvalya/GSA_using_GDMaps_PCE.git
```
and navigate to it on the local machine:
```
cd GSA_using_GDMaps_PCE
```
2. Create a virtual environment (Python 3.10):
```
python3.10 -m venv new_environment_name
```
and activate it with:
```
source new_environment_name/bin/activate
```
3. Install the required packages with:
```bash
pip install -r requirements.txt
```

## Development

This repository is being actively developed. Our main objective is to offer a collection of reusable code that enables researchers to effortlessly replicate results and leverage our framework for global sensitivity analysis. Whether you are aiming to reproduce our findings or explore novel applications, this repository provides the necessary tools and resources to support your endeavors.

## References 

A substantial portion of our implementation stems from [GDM-PCE](https://github.com/katiana22/GDM-PCE). We express our gratitude to the contributors of the original repository for their valuable resources.
