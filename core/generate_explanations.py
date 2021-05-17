"""Explain cell graphs using an explainer."""

import numpy as np
import torch 
import yaml
from tqdm import tqdm  
import glob 
import os
import numpy as np
import argparse
from dgl.data.utils import load_graphs
from sklearn.metrics import f1_score
import h5py
import warnings

from histocartography.ml import CellGraphModel
from histocartography.interpretability import (
    GraphPruningExplainer,
    GraphGradCAMExplainer,
    GraphGradCAMPPExplainer,
    GraphLRPExplainer
)
from histocartography.utils import set_graph_on_cuda
from constants import TUMOR_TYPE_TO_LABEL


IS_CUDA = torch.cuda.is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'

EXPLAINER_TYPE_TO_OBJ = {
    'graphgradcam': GraphGradCAMExplainer,
    'graphgradcampp': GraphGradCAMPPExplainer,
    'graphlrp': GraphLRPExplainer,
    'gnnexplainer': GraphPruningExplainer
}
NODE_DIM = 514


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--explainer',
        choices=['graphgradcam', 'graphgradcampp', 'graphlrp', 'gnnexplainer'],
        help='Explainer type.',
        required=True
    )
    parser.add_argument(
        '--cell_graphs',
        type=str,
        help='Path to the cell graphs.',
        default='../data/cell_graphs/',
    )
    parser.add_argument(
        '--save_path',
        type=str,
        help='Path to save the cell graphs.',
        default='../data/explanations',
        required=False
    )
    return parser.parse_args()


def check_io(cell_graph_path, save_path, explainer_type):

    # 1. check if the cell graph path exists
    if not os.path.isdir(cell_graph_path):
        download = input('Cell Graph path does not exist. Should we download the cell graphs? [y/n]') == 'y'
        if download:
            raise NotImplementedError('@TODO Automatic download of the cell graphs.')
        else:
            raise ValueError('Please provide a valid cell graph path.')

    # 2. create the save path if it doesn't exist 
    if not os.path.isdir(save_path):
        print('Could not find save path, creating it at: {}'.format(save_path))
        os.mkdir(save_path) 
    if not os.path.isdir(os.path.join(save_path, explainer_type)):
            os.mkdir(os.path.join(save_path, explainer_type)) 


def explain_cell_graphs(explainer, cg_path, save_path):

    cg_paths = glob.glob(os.path.join(cg_path, '*.bin'))

    all_labels = []
    all_predictions = []
    all_misclassifications = []

    for p in tqdm(cg_paths):

        # 1. load the cell graph 
        cell_graph, _ = load_graphs(p)
        cell_graph = set_graph_on_cuda(cell_graph[0]) if IS_CUDA else cell_graph[0]

        try:
            # 2. explain the graph & automatically save it in h5 file
            importance_score, logits = explainer.process(cell_graph)

            # 3. performance analysis
            pred = np.argmax(logits.squeeze(), axis=0)
            all_predictions.append(pred)
            label = TUMOR_TYPE_TO_LABEL[p.split('/')[-1].split('_')[2]]
            all_labels.append(label)

            # 4. save importance scores 
            out_fname = os.path.join(save_path, p.split('/')[-1].replace('.bin', '.h5'))
            with h5py.File(out_fname, "w") as f:
                f.create_dataset(
                    "importance_score",
                    data=importance_score,
                    compression="gzip",
                    compression_opts=9,
                )
                f.create_dataset(
                    "correct",
                    data=np.array([label == pred]),
                    compression="gzip",
                    compression_opts=9,
                )
        except:
            warnings.warn("An error occurred while creating explanation of sample. {}".format(p))

    print('Weighted F1 score:', f1_score(np.array(all_labels), np.array(all_predictions), average='weighted'))


def main(args):

    # 1. check io directories 
    check_io(args.cell_graphs, args.save_path, args.explainer)

    # 2. define CG-GNN model 
    config_fname = os.path.join(
        os.path.dirname(__file__),
        'config',
        'cg_bracs_cggnn_3_classes_gin.yml')
    with open(config_fname, 'r') as file:
        config = yaml.load(file)

    model = CellGraphModel(
        gnn_params=config['gnn_params'],
        classification_params=config['classification_params'],
        node_dim=NODE_DIM,
        num_classes=3,
        pretrained=True
    ).to(DEVICE)

    # 3. explain the cell graphs
    explainer = EXPLAINER_TYPE_TO_OBJ[args.explainer](
        model=model
    )

    explain_cell_graphs(
        explainer=explainer,
        cg_path=args.cell_graphs,
        save_path=os.path.join(args.save_path, args.explainer)
    )


if __name__ == "__main__":
    main(args=parse_arguments())
