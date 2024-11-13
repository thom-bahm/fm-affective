from typing import List, Dict
import numpy as np
from datetime import datetime

def generate_confusion_matrix(true_labels: List[str], predicted_labels: List[str], class_names: List[str]) -> List[List[int]]:
    """
    Generates a confusion matrix given the true and predicted labels and saves 
    it to an .npy file as a numpy array.
    
    Parameters:
    - true_labels: List of actual labels.
    - predicted_labels: List of predicted labels.
    - class_names: List of all possible class names.
    
    Returns:
    - A 2D list representing the confusion matrix.
    """
    # Initialize the confusion matrix with zeros
    d = len(class_names)
    confusion_matrix = [[0 for _ in range(d)] for _ in range(d)]
    
    # Create a mapping from class name to index
    class_to_index = {name: idx for idx, name in enumerate(class_names)}
    
    # Populate the confusion matrix
    for true, pred in zip(true_labels, predicted_labels):
        true_index = class_to_index[true]
        pred_index = class_to_index[pred]
        confusion_matrix[true_index][pred_index] += 1
    
    # sae the confusion matrix
    filename = f"affectnet_confusion_matrix{datetime.now().strftime('%Y-%m-%d_%H')}.npy"
    np.save(filename, np.array(confusion_matrix))

    return confusion_matrix

def print_confusion_matrix(confusion_matrix: List[List[int]], class_names: List[str]):
    """
    Prints the confusion matrix in a readable format.
    """
    print("Confusion Matrix:")
    # Print header row
    print(" " * 15 + "\t".join(class_names))
    
    for i, row in enumerate(confusion_matrix):
        row_str = "\t".join(str(x) for x in row)
        print(f"{class_names[i]:<15} {row_str}")


