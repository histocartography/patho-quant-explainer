import os 
import argparse
import glob
import h5py
import numpy as np
from PIL import Image
from dgl.data.utils import load_graphs

from histocartography.visualization import OverlayGraphVisualization, InstanceImageVisualization


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cell_graphs',
        type=str,
        help='Path to the cell graphs.',
        required=True
    )
    parser.add_argument(
        '--importance_scores',
        type=str,
        help='Path to the importance scores.',
        required=True
    )
    parser.add_argument(
        '--images',
        type=str,
        help='Path to the images.',
        required=True
    )
    parser.add_argument(
        '--save_path',
        type=str,
        help='Path to save the images with explanations.',
        required=False,
        default='../data/visualization'
    )
    return parser.parse_args()


def h5_to_numpy(h5_path, key):
    h5_object = h5py.File(h5_path, 'r')
    out = np.array(h5_object[key])
    return out


def main(args):

    os.makedirs(args.save_path, exist_ok=True)

    image_fnames = glob.glob(os.path.join(args.images, '*.png'))
    image_fnames.sort()
    images = [np.array(Image.open(p)) for p in image_fnames]

    cg_fnames = [os.path.join(args.cell_graphs, p.split('/')[-1].replace('.png', '.bin')) for p in image_fnames]
    cell_graphs = [load_graphs(p)[0][0] for p in cg_fnames]

    imp_fnames = [os.path.join(args.importance_scores, p.split('/')[-1].replace('.png', '.h5')) for p in image_fnames]
    importance_scores = [h5_to_numpy(p, key='importance_score') for p in imp_fnames]

    visualizer = OverlayGraphVisualization(
        instance_visualizer=InstanceImageVisualization(
            instance_style="filled+outline"
        ),
        colormap='jet'
    )

    for fname, image, cell_graph, importance in zip(image_fnames, images, cell_graphs, importance_scores):

        node_attributes = {}
        node_attributes["thickness"] = 15
        node_attributes["radius"] = 8
        node_attributes["color"] = importance

        out = visualizer.process(
            canvas=image,
            graph=cell_graph,
            node_attributes=node_attributes
        )
        out_fname = fname.split('/')[-1]
        out.save(os.path.join(args.save_path, out_fname))


if __name__ == "__main__":
    main(args=parse_arguments())
