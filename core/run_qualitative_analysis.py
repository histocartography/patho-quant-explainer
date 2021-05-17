import os 
import argparse
import itertools
import glob

from typing import Any, Dict, List

import numpy as np
from numpy import inf
from itertools import compress

import pandas as pd
from sklearn.preprocessing import minmax_scale
from scipy.stats import wasserstein_distance
from sklearn.metrics import auc
from dgl.data.utils import load_graphs

from histocartography.preprocessing.feature_extraction import HANDCRAFTED_FEATURES_NAMES

from constants import PATHO_PRIOR, RISK, CONCEPT_GROUPING, TUMOR_TYPE_TO_LABEL
from utils import plot_histogram, h5_to_numpy


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
        '--misclassification',
        help='If the analysis should include misclassified samples.',
        action='store_true',
    )
    parser.add_argument(
        '--save_path',
        type=str,
        help='Path to save histograms.',
        required=False,
        default='../data/visualization'
    )
    return parser.parse_args()


class AttributeSeparability():
    def __init__(
        self,
        keep_nuclei: str = '5,10,15,20,25,30,35,40,45,50',
        tumor_classes: str = '0,1,2',
    ) -> None:
        """
        AttributeSeparability constructor. 

        Args:
            keep_nuclei (str): Number of nuclei to retain each time. Default to '5,10,15,20,25,30,35,40,45,50'. 
            tumor_classes (str): Default to '0,1,2'.
        """

        self.keep_nuclei_list = [int(x) for x in keep_nuclei.split(',')]
        self.n_keep_nuclei = len(self.keep_nuclei_list)
        self.tumor_classes = [int(x) for x in tumor_classes.split(',')]
        self.n_tumors = len(self.tumor_classes)
        self.class_pairs = list(itertools.combinations(self.tumor_classes, 2))
        self.n_class_pairs = len(self.class_pairs)

    def process(
        self,
        nuclei_importance_list: List[np.ndarray], 
        nuclei_concept_list: List[np.ndarray],
        tumor_label_list: List[int]
    ) -> Any:
        """
        Derive metrics based on the explainer importance
        scores and nuclei-level concepts. 

        Args:
            nuclei_importance_list (List[np.ndarray]): List of nuclei importance scores output by explainers. 
            nuclei_concept_list (List[np.ndarray]): List of nuclei-level concepts. 
            tumor_label_list (List[int]): List of tumor-level labels.
        """

        # 1. extract number of concepts
        n_attrs = nuclei_concept_list[0].shape[1]

        # 2. min max normalize the importance scores
        nuclei_importance_list = self.normalize_node_importance(nuclei_importance_list)

        # 3. extract all the histograms
        all_histograms = self._compute_attr_histograms(nuclei_importance_list, nuclei_concept_list, tumor_label_list, n_attrs)

        # 4. compute the Wasserstein distance for all the class pairs
        all_distances = self._compute_hist_distances(all_histograms, n_attrs)

        # 5. compute the AUC over the #k: output will be Omega x #c
        all_aucs = {}
        for class_pair_id in range(self.n_class_pairs):
            all_aucs[class_pair_id] = {}
            for attr_id in range(n_attrs):
                attr_name = [key for key, val in HANDCRAFTED_FEATURES_NAMES.items() if val == attr_id][0]
                all_aucs[class_pair_id][attr_name] = auc(
                    np.array(self.keep_nuclei_list) / np.max(self.keep_nuclei_list),
                    all_distances[:, class_pair_id, attr_id]
                )

        return all_aucs, all_histograms

    def _compute_hist_distances(
        self,
        all_histograms: Dict,
        n_attr: int
    ) -> np.ndarray:
        """
        Compute all the pair-wise histogram distances. 

        Args:
             all_histograms (Dict): all the histograms. 
             n_concepts (int): number of concepts. 
        """
        all_distances = np.empty((self.n_keep_nuclei, self.n_class_pairs, n_attr))
        for k_id , k in enumerate(self.keep_nuclei_list):
            omega = 0
            for tx in range(self.n_tumors):
                for ty in range(self.n_tumors):
                    if tx < ty:
                        for attr_id in range(n_attr):
                            all_distances[k_id, omega, attr_id] = wasserstein_distance(
                                all_histograms[k][tx][attr_id],
                                all_histograms[k][ty][attr_id]
                            )
                        omega += 1
        return all_distances

    def _compute_attr_histograms(
        self, 
        importance_list: List[np.ndarray], 
        concept_list: List[np.ndarray],
        label_list: List[int],
        n_attrs: int
    ) -> Dict:
        """
        Compute histograms for all the attributes. 

        Args:
            importance_list (List[np.ndarray]): List of nuclei importance scores output by explainers. 
            concept_list (List[np.ndarray]): List of nuclei-level concepts. 
            label_list (List[int]): List of tumor-level labels.
        Returns:
            all_histograms (Dict[Dict[np.ndarray]]): Dict with all the histograms
                                                     for each thresh k (as key),
                                                     tumor type (as key) and 
                                                     attributes (as np array).
        """
        all_histograms = {}
        for k in self.keep_nuclei_list:
            all_histograms[k] = {}

            attrs = [c[np.argsort(s)[-k:]] for c, s in zip(concept_list, importance_list)]
            attrs = np.concatenate(attrs, axis=0)  # (#samples x k) x #attrs 
            attrs[attrs == inf] = 0  # ensure no weird values in attributes 
            attrs = minmax_scale(attrs)   
            attrs = np.reshape(attrs, (-1, k, n_attrs))  # #samples x k x #attrs 
            attrs = list(attrs)

            for t in range(self.n_tumors):

                # i. extract the samples of type t
                selected_attrs = [a for l, a in zip(label_list, attrs) if l==t]
                selected_attrs = np.concatenate(selected_attrs, axis=0)

                # iii. build the histogram for all the attrs (dim = #nuclei x attr_types)
                all_histograms[k][t] = np.array(
                    [self.build_hist(selected_attrs[:, attr_id]) for attr_id in range(selected_attrs.shape[1])]
                )
        return all_histograms

    @staticmethod
    def normalize_node_importance(node_importance: List[np.ndarray]) -> List[np.ndarray]:
        """
        Normalize node importance. Min-max normalization on each sample. 

        Args:
            node_importance (List[np.ndarray]): node importance output by an explainer. 
        Returns:
            node_importance (List[np.ndarray]): Normalized node importance. 
        """
        node_importance = [minmax_scale(x) for x in node_importance] 
        return node_importance

    @staticmethod
    def build_hist(concept_values: np.ndarray, num_bins: int = 100) -> np.ndarray:
        """
        Build a 1D histogram using the concept_values. 

        Args:
            concept_values (np.ndarray): All the nuclei-level values for a concept. 
            num_bins (int): Number of bins in the histogram. Default to 100. 
        Returns:
            hist (np.ndarray): Histogram
        """
        hist, _ = np.histogram(concept_values, bins=num_bins, range=(0., 1.), density=True)
        return hist


class SeparabilityAggregator:

    def __init__(
        self,
        separability_scores: Dict,
    ) -> None:
        """
            SeparabilityAggregator constructor. 

        Args:
            separability_score (Dict[Dict][float]): Separability score for all the class pairs
                                                    (as key) and attributes (as key). 
        """
        self.separability_scores = self._group_separability_scores(separability_scores)

    def _group_separability_scores(self, sep_scores: Dict) -> Dict:
        """
        Group the individual attribute-wise separability scores according
        to the grouping concept. 

        Args:
            sep_scores (Dict): Separability scores 
        Returns:
            grouped_sep_scores (Dict): Grouped separability scores 
        """
        grouped_sep_scores = {}

        for class_pair_key, class_pair_val in sep_scores.items():
            grouped_sep_scores[class_pair_key] = {}
            for concept_key, concept_attrs in CONCEPT_GROUPING.items():
                val = sum([class_pair_val[attr] for attr in concept_attrs]) / len(concept_attrs)
                grouped_sep_scores[class_pair_key][concept_key] = val
        return grouped_sep_scores

    def compute_max_separability_score(self) -> Dict:
        """
        Compute maximum separability score for each class pair. Then the 
        aggregate max sep score w/ and w/o risk. 

        Returns:
            max_sep_score (Dict): Maximum separability score. 
        """
        max_sep_score = {}
        for class_pair_key, class_pair_val in self.separability_scores.items():
            max_sep_score[class_pair_key] = max([val for _, val in class_pair_val.items()])
        max_sep_score['agg_with_risk'] = sum(
            np.array([val for _, val in max_sep_score.items()]) *
            RISK
        ) 
        max_sep_score['agg'] = sum([val for key, val in max_sep_score.items() if type(key)==int]) 
        return max_sep_score

    def compute_average_separability_score(self) -> Dict:
        """
        Compute average separability score for each class pair. Then the 
        aggregate avg sep score w/ and w/o risk. 

        Returns:
            avg_sep_score (Dict): Average separability score. 
        """
        avg_sep_score = {}
        for class_pair_key, class_pair_val in self.separability_scores.items():
            avg_sep_score[class_pair_key] = np.mean(np.array([val for _, val in class_pair_val.items()]))
        avg_sep_score['agg_with_risk'] = sum(
            np.array([val for _, val in avg_sep_score.items()]) *
            RISK
        ) 
        avg_sep_score['agg'] = sum([val for key, val in avg_sep_score.items() if type(key)==int]) 
        return avg_sep_score

    def compute_correlation_separability_score(self) -> float:
        """
        Compute correlation separability score between the prior 
        and the concept-wise separability scores.  

        Returns:
            corr_sep_score (Dict): Correlation separability score. 
        """
        sep_scores = pd.DataFrame.from_dict(self.separability_scores).to_numpy()
        sep_scores = minmax_scale(sep_scores)
        corrs = {}
        for tumor_pair in range(sep_scores.shape[1]):
            corr_sep_score = np.corrcoef(PATHO_PRIOR[:, tumor_pair], sep_scores[:, tumor_pair])
            corrs[tumor_pair] = corr_sep_score[1, 0]
        corrs['agg_with_risk'] = sum(
            np.array([val for _, val in corrs.items()]) *
            RISK
        ) 
        corrs['agg'] = sum([val for key, val in corrs.items() if type(key)==int]) 
        return corrs



def main(args):

    os.makedirs(args.save_path, exist_ok=True)

    # 1. load the cell graphs, labels and importance scores 
    imp_fnames = glob.glob(os.path.join(args.importance_scores, '*.h5'))
    imp_fnames.sort()
    cg_fnames = [os.path.join(args.cell_graphs, p.split('/')[-1].replace('.h5', '.bin')) for p in imp_fnames]

    cell_graphs = [load_graphs(p)[0][0] for p in cg_fnames]
    concepts = [cg.ndata['concepts'].cpu().detach().numpy() for cg in cell_graphs]
    labels = [TUMOR_TYPE_TO_LABEL[p.split('/')[-1].split('_')[2]] for p in cg_fnames]
    importance_scores = [h5_to_numpy(p, key='importance_score') for p in imp_fnames]

    # 2. prune the misclassifed samples
    if not args.misclassification: 
        correct = [h5_to_numpy(p, key='correct')[0] for p in imp_fnames]
        concepts = list(compress(concepts, correct))
        labels = list(compress(labels, correct))
        importance_scores = list(compress(importance_scores, correct))

    # 3. compute separability scores
    separability_calculator = AttributeSeparability()
    separability_scores, all_histograms = separability_calculator.process(
        nuclei_importance_list=importance_scores, 
        nuclei_concept_list=concepts,
        tumor_label_list=labels
    )

    # 4. plot histograms 
    for attr_id, attr_name in zip([0, 2, 13, 22, 16], ['area', 'eccentricity', 'shape_factor', 'crowdedness', 'contrast']):
        plot_histogram(all_histograms, args.save_path, attr_id, attr_name, k=25)

    # 5. compute final qualitative metrics 
    metric_analyser = SeparabilityAggregator(separability_scores)
    average = metric_analyser.compute_average_separability_score()
    maximum = metric_analyser.compute_max_separability_score()
    correlation = metric_analyser.compute_correlation_separability_score()

    # 6. print output
    print('*** Separability scores ***')
    for task_id, val in metric_analyser.separability_scores.items():
        print('\nClass pair ID:', task_id)
        for concept_name, score in val.items():
            print(concept_name, round(score, 3))

    print('\n\n*** Average separability scores ***')
    for task_id, score in average.items():
        print(task_id, round(score, 3))

    print('\n\n*** Maximum separability scores ***')
    for task_id, score in maximum.items():
        print(task_id, round(score, 3))

    print('\n\n*** Correlation separability scores ***')
    for task_id, score in correlation.items():
        print(task_id, round(score, 3))


if __name__ == "__main__":
    main(args=parse_arguments())
