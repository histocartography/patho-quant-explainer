# Quantifying Explainers of Graph Neural Networks in Computational Pathology

Code for replicating results presented in the paper [Quantifying Explainers of Graph Neural Networks in Computational Pathology](https://arxiv.org/pdf/2011.12646.pdf). 

The code heavily relies on the [`histocartography`](https://github.com/histocartography/histocartography) library, a python-based package for modeling and learning with graphs of pathology images. 

All the experiments are based on the BRACS dataset. The data needs to be downloaded separately (see Installation steps). 

## Installation 

### Cloning and handling ependencies 

Clone the repo:

```
git clone <> && cd quant-gnn-explainers
```

Create a conda environment and activate it:

```
conda env create -f environment.yml
conda activate pathoexplainer
```

### Downloading the BRACS dataset 

BRACS is a dataset of Hematoxylin and Eosin (H&E) histopathological images for automated detection/classification of breast tumors. BRACS includes >4k tumor regions-of-interest labeled in 7 categories (Normal, Benign, UDH, ADH, FEA, DCIS, Invasive). 

In order to download the BRACS dataset, you need to create an account [there](https://www.bracs.icar.cnr.it/). Then, go to `Data Collection`, `Download`, and hit the `Regions of Interest Set` button to access the data. Download the `v1` data. The data are stored on an FTP server. 

## Running the code 

The proposed approach for explainability of histology images is based on 3 steps: cell graph generation, post-hoc graph explainers, quantitative analysis. 

### Step 1: Cell graph generation 

Each image needs to be transformed into a cell graph where nodes represent nuclei and edges nuclei-nuclei interactions. The cell graph for the BRACS test set can be generated with: 

```
cd core
python generate_cell_graphs.py --data_path <PATH-TO-BRACS>/BRACS/test/ --save_path <SOME-SAVE-PATH>/quant-gnn-explainers-data
```

The script will automatically create a directory containing a cell graph as a `.bin` file for each image. Should be 626 files. 

### Step 2: Explaining the cell graphs

We benchmark 4 different explainers: GraphLRP, GNNExplainer, GraphGradCAM and GraphGradCAM++, that returns a different explanation, ie nuclei-level importance scores. The system will automatically download a pre-trained checkpoint that reaches 74% accuracy on the test set. 

Generating explanation with:

```

```


### Step 3: Quantifying explainers


If you use this code, please consider citing our work:

```
@inproceedings{jaume2021,
    title = "Quantifying Explainers of Graph Neural Networks in Computational Pathology",
    author = "Guillaume Jaume, Pushpak Pati, Behzad Bozorgtabar, Antonio Foncubierta-Rodríguez, Florinda Feroce, Anna Maria Anniciello, Tilman Rau, Jean-Philippe Thiran, Maria Gabrani, Orcun Goksel",
    booktitle = "IEEE CVPR",
    url = "https://arxiv.org/abs/2011.12646",
    year = "2021"
} 
```
