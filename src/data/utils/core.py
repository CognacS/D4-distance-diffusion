import numpy as np

def get_dict_histogram(array: np.ndarray):

    unique, counts = np.unique(
        array,
        return_counts=True
    )

    return dict(zip(unique.tolist(), counts.tolist()))