
import numpy as np
from collections import Counter
import math  

def euclidean_distance(point1, point2):
    
    """
    Calculate the Euclidean distance between two points.

    Parameters
    ----------
    point1 : array
        The first point.
    point2 : array
        The second point.

    Returns
    -------
    distance : float
        The Euclidean distance between the two points.
    """
    
    distance = 0  # Initialize distance
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return distance ** 0.5  # Return the square root of the sum


def calculate_hypersphere_volume(radius, dimensions):
    
    """
    Calculate the volume of a hypersphere with a given radius and number of dimensions.

    Parameters
    ----------
    radius : float
        The radius of the hypersphere.
    dimensions : int
        The number of dimensions of the hypersphere.

    Returns
    -------
    volume : float
        The volume of the hypersphere.
    """
    if dimensions < 1:
        raise ValueError("Dimensions must be a positive integer.")
    
    # Use the formula for hypersphere volume
    volume = (math.pi ** (dimensions / 2)) * (radius ** dimensions) / math.gamma(dimensions / 2 + 1)
    return volume

def calculate_density(data, radius):
   
    """
    Calculate the density of each point in the dataset.

    Parameters
    ----------
    data : array-like
        The input dataset.
    radius : float
        The radius of the hypersphere around each point.

    Returns
    -------
    densities : array
        The density of each point in the dataset.
    """
    densities = []
    n = len(data)
    dimensions = data.shape[1]  # Number of features (dimensions)
    
    for i in range(n):
        # Compute distances from data[i] to all other points in the dataset
        distances = np.linalg.norm(data - data[i], axis=1)  # Vectorized distance computation
        count_within_radius = np.sum(distances <= radius)  # Count neighbors within radius
        volume = calculate_hypersphere_volume(radius, dimensions)  # Calculate hypersphere volume
        densities.append(count_within_radius / volume)
    
    return np.array(densities)

def calculate_degree_of_certainty(class_scores):
    
    """
    Calculate the degree of certainty given a dictionary of class scores.
    
    Parameters
    ----------
    class_scores : dict
        A dictionary mapping class labels to their corresponding scores.
    
    Returns
    -------
    degree_of_certainty : float
        The degree of certainty that a given point belongs to a class.
    """
    max_score = max(class_scores.values())
    total_score = sum(class_scores.values())
    return max_score / total_score if total_score != 0 else 0

def calculate_class_scores(neighbor_indices, y_train, densities_normalized, distances):
    """
    Calculate scores for each class based on the k nearest neighbors.

    Parameters
    ----------
    neighbor_indices : array
        The indices of the k nearest neighbors.
    y_train : array
        The labels of the k nearest neighbors.
    densities_normalized : array
        The normalized densities of the k nearest neighbors.
    distances : array
        The distances from the current point to the k nearest neighbors.

    Returns
    -------
    class_scores : dict
        A dictionary mapping class labels to their corresponding scores.
    """
    class_scores = {}
    for idx in neighbor_indices:
        label = y_train[idx]
        density = densities_normalized[idx]
        score = density / (distances[idx] + 1e-9)  # Avoid division by zero
        if label not in class_scores:
            class_scores[label] = 0
        class_scores[label] += score
    return class_scores

def db_knn_predict(train_data, y_train, test_data, k, radius):
    X_train = train_data[:, :-1]  # Features of training data
    y_train = train_data[:, -1]  # Labels of training data
    X_test = test_data[:, :-1]   # Features of testing data
    
    # Precompute densities for training points
    densities = calculate_density(X_train, radius)
    densities_normalized = densities / np.max(densities)
    
    predictions = []
    for test_point in X_test:
        # Compute distances from the test point to all training points
        distances = np.linalg.norm(X_train - test_point, axis=1)
        
        # Find the k nearest neighbors
        neighbor_indices = np.argsort(distances)[:k]
        
        # Calculate scores for each class
        class_scores = calculate_class_scores(neighbor_indices, y_train, densities_normalized, distances)
        
        # Calculate Degree of Certainty
        dc = calculate_degree_of_certainty(class_scores)
        print("Degree of Certainty:", dc)
        if dc < 0.667:
            print("\nFalling back to classic kNN")

            # Fallback to classic kNN
            nearest_label_counts = Counter(y_train[neighbor_indices])
            predicted_class = nearest_label_counts.most_common(1)[0][0]
        else:
            # Predict the class with the highest score
            predicted_class = max(class_scores, key=class_scores.get)
        
        predictions.append(predicted_class)
    
    return np.array(predictions)
