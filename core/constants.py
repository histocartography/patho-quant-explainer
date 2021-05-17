import numpy as np


PATHO_PRIOR = np.asarray([
    [0.617283951, 0.672131148, 0.529069767],
    [0.839506173, 0.896174863, 0.872093023],
    [1., 1., 1.],
    [0., 0., 0.],
    [0.25308642, 0.404371585, 0.377906977],
])

RISK = np.array([1, 2, 1])

CONCEPT_GROUPING = {
    'size': ['area'],
    'shape': ['perimeter', 'roughness', 'eccentricity', 'roundness'],
    'shape_variation': ['shape_factor'],
    'spacing': ['mean_crowdedness', 'std_crowdedness'],
    'chromatin': ['glcm_dissimilarity', 'glcm_contrast', 'glcm_homogeneity', 'glcm_ASM', 'glcm_dispersion']
}

TUMOR_TYPE_TO_LABEL = {
    'N': 0,
    'PB': 0,
    'UDH': 0,
    'ADH': 1,
    'FEA': 1,
    'DCIS': 2,
    'IC': 2
}