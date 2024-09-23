import numpy as np
from sklearn import metrics
from math import sqrt
from tqdm import tqdm
from typing import Tuple, List
import tensorflow as tf
from extra_types import PSRSequence


def run_epoch(
    model: tf.keras.Model, 
    data: PSRSequence, 
    num_skills: int, 
    num_steps: int, 
    batch_size: int = 128, 
    is_training: bool = True
) -> Tuple[float, float]:
    """
    Runs one epoch of training or evaluation.

    Args:
        model (tf.keras.Model): The model to be trained or evaluated.
        data (PSRSequence): Processed dataset containing sequences of problem IDs, skill IDs, and correctness scores.
        num_skills (int): The number of unique skills (or problems) in the dataset.
        num_steps (int): The number of steps (sequence length) for the model.
        batch_size (int): Number of examples per batch.
        is_training (bool): Flag to indicate whether the model is in training mode.
    
    Returns:
        Tuple[float, float]: Root mean squared error (RMSE) and AUC score for the current epoch.
    """
    actual_labels: List[int] = []
    pred_labels: List[float] = []
    index: int = 0
    
    with tqdm(total=len(data) // batch_size) as pbar:
        while index + batch_size < len(data):
            x: np.ndarray = np.zeros((batch_size, num_steps - 1), dtype=np.int32)
            problems: np.ndarray = np.zeros((batch_size, num_steps - 1), dtype=np.int32)
            target_id: List[int] = []
            target_correctness: List[int] = []

            for i in range(batch_size):
                problem_ids: List[int]
                skill_ids: List[int]
                correctness: List[int]
                problem_ids, skill_ids, correctness = data[index + i]
                
                for j in range(num_steps - 1):
                    problem_id: int = int(problem_ids[j])
                    label_index: int = problem_id + (num_skills if int(correctness[j]) else 0)
                    x[i, j] = label_index
                    problems[i, j] = problem_ids[j + 1] if j + 1 < num_steps else 0
                    target_id.append(i * (num_steps - 1) + j)
                    target_correctness.append(correctness[j + 1] if j + 1 < num_steps else 0)
                    actual_labels.append(correctness[j + 1] if j + 1 < num_steps else 0)

            index += batch_size

            inputs: Tuple[np.ndarray, np.ndarray, List[int], List[int]] = (x, problems, target_id, target_correctness)
            pred: tf.Tensor
            pred, _, _ = model.call(inputs, training=is_training)

            pred_labels.extend(pred.numpy().tolist())

            pbar.update(1)

    rmse: float = sqrt(metrics.mean_squared_error(actual_labels, pred_labels))
    fpr: np.ndarray
    tpr: np.ndarray
    _ : np.ndarray
    fpr, tpr, _ = metrics.roc_curve(actual_labels, pred_labels)
    auc: float = metrics.auc(fpr, tpr)
    
    return rmse, auc
