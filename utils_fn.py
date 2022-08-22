import os

from typing import List

import numpy as np


def enumerate_images(images_dir: str) -> List[str]:
    images_list: List[str] = []

    for dirs, _, files in os.walk(images_dir):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png", ".ifjf")):
                images_list.append(os.path.join(dirs, file))

    return images_list


def compare_faces(features_1: np.ndarray, feature_2: np.ndarray, threshold: float=0.8) -> bool:
    assert len(feature_2.shape) == 1
    assert len(features_1.shape) == 2
    assert features_1.shape[1] == feature_2.shape[0]

    '''similarity = np.sum(features_1 * feature_2[np.newaxis, :], axis=1)

    return (similarity > threshold)'''
    similarity = np.sqrt(np.sum((features_1 - feature_2[np.newaxis, :])**2, axis=1))
    return (similarity < threshold)