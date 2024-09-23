import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np

def plot_attention(attention_weights, head = 0):
    """
    Plots the attention weights for a specific attention head.

    Args:
        attention_weights (np.ndarray): A numpy array representing the attention weights from the model.
        head (int): The index of the attention head to visualize. Defaults to the first head (0).
    
    Returns:
        None: Displays a heatmap of the attention weights.
    """
    plt.figure(figsize=(12, 6))
    sns.heatmap(attention_weights[head], cmap = 'coolwarm')
    plt.title('Attention Weights for Head {head}')
    plt.show()

def plot_latent_topics(problem_embeddings, skill_embeddings):
    """
    Plots the latent topic mapping of problem and skill embeddings using t-SNE for dimensionality reduction.

    Args:
        problem_embeddings (np.ndarray): Numpy array of problem embeddings.
        skill_embeddings (np.ndarray): Numpy array of skill embeddings.
    
    Returns:
        None: Displays a scatter plot of the latent topics for problems and skills.
    """
    tsne = TSNE(n_components = 2, random_state = 69)
    problem_latent = tsne.fit_transform(problem_embeddings)
    skill_latent = tsne.fit_transform(skill_embeddings)
    plt.figure(figsize=(12, 6))
    plt.scatter(problem_latent[:, 0], problem_latent [:, 1], label = "Problems", alpha = 0.6, marker = 'o')
    plt.scatter(skill_latent[:, 0], skill_latent[:, 1], label = 'Skills', alpha = 0.6, marker = '+')
    plt.legend()
    plt.title('t-SNE of Problems and Skills')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.show()