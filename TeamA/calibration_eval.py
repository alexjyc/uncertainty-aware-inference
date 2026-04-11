import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

output_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(output_dir, exist_ok=True)

# ==============================================================================
# NEW: PROBABILITY SCORING FIXES (Priorities 1 & 2)
# ==============================================================================

def score_multiple_choice(choice_log_probs_list):
    """
    PRIORITY 1 FIX: Multiple-Choice Length Bias
    Computes confidence using the average log-likelihood per token rather 
    than the sum, preventing the pathological bias toward shorter answers.
    
    Args:
        choice_log_probs_list: A list of 1D tensors, where each tensor contains 
                               the log probabilities of the tokens for one MC choice.
    Returns:
        predicted_confidence (float): The confidence of the chosen answer.
        predicted_idx (int): The index of the chosen answer.
    """
    # Use .mean() instead of .sum() to normalize by sequence length
    mean_log_probs = torch.stack([probs.mean() for probs in choice_log_probs_list])
    
    # Softmax across the normalized choices to get a valid probability distribution
    confidences = F.softmax(mean_log_probs, dim=0)
    
    predicted_confidence, predicted_idx = torch.max(confidences, dim=0)
    return predicted_confidence.item(), predicted_idx.item()

def score_generative_qa(generated_token_log_probs):
    """
    PRIORITY 2 FIX: Sequence-Level Confidence for Generative QA
    Replaces first-token confidence with the geometric mean over the 
    entire generated sequence to reflect true answer uncertainty.
    
    Args:
        generated_token_log_probs: A 1D tensor of log probabilities for the generated tokens.
    Returns:
        confidence (float): The sequence-level confidence.
    """
    num_tokens = generated_token_log_probs.size(0)
    
    if num_tokens == 0:
        return 0.0
        
    seq_log_prob = generated_token_log_probs.sum()
    
    # Geometric mean: exp(sum(log_probs) / N)
    confidence = torch.exp(seq_log_prob / num_tokens)
    return confidence.item()

# ==============================================================================
# EXISTING: CALIBRATION METRICS & PLOTTING
# ==============================================================================

def compute_calibration_metrics(confidences, accuracies, entropies, num_bins=15):
    """Computes ECE, MCE, Brier Score, and Average Entropy."""
    bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=confidences.device)

    ece = torch.tensor(0.0, device=confidences.device)
    mce = torch.tensor(0.0, device=confidences.device)
    
    # Brier score requires float comparisons
    brier_score = torch.mean((confidences - accuracies.float()) ** 2).item()

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()

            calibration_error = torch.abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += calibration_error * prop_in_bin
            if calibration_error > mce:
                mce = calibration_error

    return {
        "ECE": ece.item(),
        "MCE": mce.item(),
        "Brier_Score": brier_score,
        "Avg_Entropy": entropies.mean().item()
    }

def plot_reliability_diagram(confidences, accuracies, title="Reliability Diagram", num_bins=15):
    confidences = confidences.cpu().numpy()
    accuracies = accuracies.cpu().numpy()
    bins = np.linspace(0, 1, num_bins + 1)
    
    # digitize returns 1-indexed bins, subtract 1 for 0-indexed array access
    bin_indices = np.digitize(confidences, bins) - 1
    # Handle edge case where confidence is exactly 1.0
    bin_indices[bin_indices == num_bins] = num_bins - 1

    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    
    for i in range(num_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_accs[i] = np.mean(accuracies[mask])
            bin_confs[i] = np.mean(confidences[mask])

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')
    
    # Only plot bars for bins that actually have data
    plt.bar(bins[:-1], bin_accs, width=1/num_bins, align='edge', alpha=0.5, edgecolor='black', label='Accuracy')
    
    valid_bins = bin_confs > 0
    if np.any(valid_bins):
        plt.plot(bin_confs[valid_bins], bin_accs[valid_bins], 'ro-', label='Confidence')
        
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    safe_title = title.replace(' ', '_').replace('/', '-')
    plt.savefig(os.path.join(output_dir, f"{safe_title}_reliability_diagram.png"), bbox_inches='tight')
    plt.close()

def plot_entropy(entropies, accuracies, title="Entropy Distribution"):
    e = entropies.cpu().numpy()
    a = accuracies.cpu().numpy()
    
    plt.figure(figsize=(6, 4))
    
    correct_e = e[a == 1]
    incorrect_e = e[a == 0]
    
    if len(correct_e) > 0:
        plt.hist(correct_e, bins=20, alpha=0.6, color='green', label='Correct', density=True)
    if len(incorrect_e) > 0:
        plt.hist(incorrect_e, bins=20, alpha=0.6, color='red', label='Incorrect', density=True)
        
    plt.xlabel('Shannon Entropy (bits)')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    
    safe_title = title.replace(' ', '_').replace('/', '-')
    plt.savefig(os.path.join(output_dir, f"{safe_title}_entropy_distribution.png"), bbox_inches='tight')
    plt.close()
