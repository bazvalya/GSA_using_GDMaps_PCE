# A Manifold Learning-Based Framework for Global Sensitivity Analysis

This Git repository contains the code implementation for the "Global Sensitivity Analysis using Polynomial Chaos Expansion on the Grassmann Manifold" paper. The paper introduces a novel approach to global sensitivity analysis (GSA) that is specifically designed to address the challenges posed by models with diverse timescales, structural complexity, and other dynamics inherent in complex systems, such as agent-based models (ABMs).

## Table of contents
- [Abstract Summary](#abstract-summary)
- [Key Features](#key-features)
- [Methodology](#methodology)
- [Repository Contents](#repository-contents)
- [Data](#data)
- [Getting Started](#getting-started)
- [Development](#development)
- [References ](#references)

## Abstract Summary
Traditional GSA techniques, including variance- and density-based approaches, have limitations when it comes to comprehensively understanding temporal dynamics in complex spatiotemporal systems within the context of complex systems theory. To overcome these limitations, the proposed method combines manifold learning techniques with polynomial chaos expansion (PCE) to assess parameter sensitivities. By reducing the dimensionality of the data using Grassmannian diffusion maps (GDMaps) and mapping stochastic input parameters to diffusion coordinates in the reduced space, the method provides a more comprehensive estimation of sensitivities by considering multiple outputs and their entire trajectories.

## Key Features

1. **Benefits of the proposed GSA method:** 
  - The method provides a more informative quantification of parametric sensitivities by:
    - Aggregating multiple outputs and their entire trajectories for a comprehensive analysis.
    - Reducing the dimensionality of the data using Grassmannian Diffusion Maps (GDMaps) for improved understanding.
  - The framework is designed to handle non-linearities and capture interaction effects in agent-based models (ABMs) and complex systems.
2. **Successful application:**
  - The method has been successfully applied to both a classic Lotka-Volterra dynamical system and a large-scale ABM model.
  - Application of the framework revealed important parameter relations and relative influences on the model outputs.
3. **Influence of hyper-parameters:**
  - Sensitivity measures are affected by the choice of Grassmann manifold dimension and maximal polynomial degree.
  - Fine-tuning these hyper-parameters is recommended.
4. **Potential impacts:**
  - Deepening the understanding of systems with complex spatiotemporal dynamics by providing insights into parameter sensitivities.
  - Expanding the application of manifold-based approaches in ABMs and other complex systems, enabling more comprehensive analyses.

## Methodology and Application Setup

A detailed overview of the proposed methodology can be found in [SI Section B.1](https://doi.org/10.5281/zenodo.8050579), and descriptions of the two models used to illustrate the application of the proposed framework and the corresponding setup used for evaluation are presented in [SI Sections B.2 and B.3](https://doi.org/10.5281/zenodo.8050579).

## Repository Contents

- `GSA_GDM_PCE`: Contains the source code for the proposed method.

- `notebooks`: Provides examples demonstrating the application of the method and includes the following files:
  - `GSA_results`: Contains the resulting data from applying the framework on Lotka-Volterra (LV) model (in `LV` folder) and DeepABM COVID-19 (in `parsim` and `nonparsim` folders)
  - `input_data`: Contains the input data of parameter combinations sampled with low-discrepancy sequence required for running the code.
  - `pce_accuracy`: When assessing the accuracy of PCE, the resulting plots are saved here.
  - `plots`: Contain all plots used in the paper (main text) and SI.
  - `DeepABM_PCE-GSA.ipynb`: Jupyter notebook containing the application of the framework on DeepABM COVID-19.
  - `DeepABM_SobolGSA.ipynb`: Jupyter notebook with GSA for DeepABM COVID-19 model employing conventional Sobol’ index calculation methods over multiple time steps. Contains code for writing datafile from the original data.
  - `GDMaps_PCE_LV.ipynb`: Jupyter notebook with the application of the proposed framework on LV model.
  - `GDMaps_unit_sphere.ipynb`: Jupyter notebook containing a simple example illustrating GDMaps’ capability to capture intrinsic geometric structures following the example previously used by [Dos Santos et al.](https://arxiv.org/abs/2009.07547)

`Snellius_DeepABM`: Cannot be treated as part of a stand-alone repository and contains files used to write data for DeepABM COVID-19 together with [this repository](https://github.com/ayushchopra96/deepabm-covid).

## Data
To run with the original data, download it from [https://figshare.com/articles/dataset/data_zip/23515965](https://figshare.com/articles/dataset/output_data_zip/22216921) and add the unzipped folder named `output_data` into `notebooks` folder.

## Getting Started

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

This repository is being actively developed. Our objective is to offer a collection of reusable code that enables researchers to replicate results effortlessly and leverage our framework for global sensitivity analysis. Whether you are aiming to reproduce our findings or explore novel applications, this repository provides the necessary tools and resources to support your endeavours.

## References 

A substantial portion of our implementation stems from [GDM-PCE](https://github.com/katiana22/GDM-PCE). We express our gratitude to the contributors of the original repository for their valuable resources.
