# @Author  :   Snehit
# @E-mail  :   snehitc@gmail.com


import numpy as np

def cosine_similarity(vector1, vector2):
    """
    Calculates the cosine similarity between two vectors.
    """
    dot_product = np.sum(vector1 * vector2, axis=1)
    magnitude_vector1 = np.linalg.norm(vector1, axis=1)
    magnitude_vector2 = np.linalg.norm(vector2, axis=1)

    cosine_similarity = dot_product / (magnitude_vector1 * magnitude_vector2)
    cosine_similarity = np.nan_to_num(cosine_similarity)
    cosine_similarity = np.expand_dims(cosine_similarity, axis=1)
    return cosine_similarity


def angular_distance(vector1, vector2):
    """
    Calculates the cosine angular distance between two vectors.
    """
    dot_product = np.sum(vector1 * vector2, axis=1)
    magnitude_vector1 = np.linalg.norm(vector1, axis=1)
    magnitude_vector2 = np.linalg.norm(vector2, axis=1)

    cosine_similarity = dot_product / (magnitude_vector1 * magnitude_vector2)
    cosine_similarity = np.nan_to_num(cosine_similarity)

    angle_radians = 1 - np.arccos(cosine_similarity) / np.pi
    angle_radians = np.expand_dims(angle_radians, axis=1)

    return angle_radians


def L2_normalized(vector1, vector2):
    """
    Calculates the L2 normalized distance between two vectors.
    """
    l2_distance = np.linalg.norm(vector1 - vector2, axis=1, keepdims=True)
    l2_normalized = l2_distance / np.max(l2_distance)
    return l2_normalized


def L1_normalized(vector1, vector2):
    """
    Calculates the L1 normalized distance between two vectors.
    """
    l1_distance = np.sum(np.abs(vector1 - vector2), axis=1, keepdims=True)
    l1_normalized = l1_distance / np.max(l1_distance)
    return l1_normalized


def Bundle_Similarity_Angle_L2_L1(vector1, vector2):
    """
    Combines cosine similarity, angular distance, L2 normalized distance,
    and L1 normalized distance into a single feature set.
    """
    cosine_sim = calculate_cosine_similarity(vector1, vector2)
    angular_dist = calculate_angular_distance(vector1, vector2)
    l2_norm = L2_normalized(vector1, vector2)
    l1_norm = L1_normalized(vector1, vector2)

    combined_features = np.hstack((cosine_sim, angular_dist, l2_norm, l1_norm))
    return combined_features
