import torch
import torch.nn.functional as F

def focal_loss():
    """
    
    """
    
    
def create_focal_matrix():
    """
    Creates a diagonal matrix H such that
    H_{i,j} = f_i / f_{min} when i = j, and 0 if i != j.
    where f_i is the number of samples for the ith class
    and f_min is the number of samples for the least represented class (Disgust in this case)
    """