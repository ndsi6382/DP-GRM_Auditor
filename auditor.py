import math, random, pickle, os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as beta_dist
from tqdm import tqdm
from collections.abc import Callable
from typing import Literal
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typeguard import typechecked
import warnings
import shlex
import subprocess


class DPGRMAuditor():

    @typechecked
    def __init__(self, 
        pair_info: list[tuple[str, str | Literal[0, 1]]] | tuple[tuple[str, str], tuple[str, str], int],
        epsilon_claimed: float, 
        delta_claimed: float, 
        confidence: float, 
        metric_fn: Callable[..., float | int],
        selection_evaluation_split: float = 0.5,
        seed: int | None = None,
    ):
        if seed is not None:
            random.seed(seed)

        if isinstance(pair_info, list): # [(path, path)] OR [(path, 0/1)]
            self.n_trials = len(pair_info)
            if isinstance(pair_info[0][1], str): # [(path, path)]
                for (g0_file, g1_file) in pair_info:
                    if not os.path.exists(g0_file):
                        raise FileNotFoundError(f"Cannot find file {g0_file}")
                    if not os.path.exists(g1_file):
                        raise FileNotFoundError(f"Cannot find file {g1_file}")
            else: # [(path, 0/1)]
                for (g0_file, label) in pair_info:
                    if not os.path.exists(g0_file):
                        raise FileNotFoundError(f"Cannot find file {g0_file}")
                    assert label in [0, 1]
        else: #((cmd_1, out), (cmd_2, out), n_trials) (NOT LIST)
            warnings.warn(
                "Commands for DP mechanism execution have been provided." + \
                "The mechanism will be executed for each of the specified trials." + \
                "Note this may take a while.",
                UserWarning
            )
            self.n_trials = pair_info[2]
        
        self.pairs = pair_info
        self.epsilon_claimed = epsilon_claimed
        self.delta_claimed = delta_claimed
        self.confidence = confidence
        self.metric_fn = metric_fn
        if 0 <= selection_evaluation_split <= 1:
            self.selection_evaluation_split = selection_evaluation_split
        else:
            raise ValueError("selection_evaluation_split must be 0 <= x <= 1.")
        self.results = None
        self.evaluation_labels = None
        self.evaluation_scores = None

    
    @typechecked
    def __compute_roc__(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_pos  = int((labels == 0).sum())
        n_neg  = int((labels == 1).sum())
        thresholds = np.concatenate([[np.inf], np.sort(np.unique(scores))[::-1], [-np.inf]])
        fprs, tprs = [], []
        for t in thresholds:
            tp = int(((scores >= t) & (labels == 0)).sum())
            fp = int(((scores >= t) & (labels == 1)).sum())
            tprs.append(tp / n_pos if n_pos else 0.0)
            fprs.append(fp / n_neg if n_neg else 0.0)
        return np.array(fprs), np.array(tprs), thresholds    


    def __selection_phase__(self, labels, scores):
        n0 = int((labels == 0).sum())
        n1 = int((labels == 1).sum())
        _, _, thresholds = DPGRMAuditor.__compute_roc__(labels, scores)
        best_eps, best_t, best_dir = -np.inf, None, None
        for t in thresholds:
            k_tp = int(((scores >= t) & (labels == 0)).sum())
            k_fp = int(((scores >= t) & (labels == 1)).sum())
            tpr = k_tp / n0 if n0 else 0.0
            fpr = k_fp / n1 if n1 else 0.0
            if fpr > 0 and (tpr - self.delta_claimed) > 0: # direction 1: (tpr, fpr)
                eps = math.log((tpr - self.delta_claimed) / fpr)
                if eps > best_eps:
                    best_eps, best_t, best_dir = eps, t, "fwd"
            omfpr, omtpr = 1.0 - fpr, 1.0 - tpr 
            if omtpr > 0 and (omfpr - self.delta_claimed) > 0: # direction 2: (1-fpr, 1-tpr)
                eps = math.log((omfpr - self.delta_claimed) / omtpr)
                if eps > best_eps:
                    best_eps, best_t, best_dir = eps, t, "rev"
        return best_t, best_dir

    
    def __evaluation_phase__(self, t_star, direction):
        n0 = int((self.evaluation_labels == 0).sum())
        n1 = int((self.evaluation_labels == 1).sum())
        gamma = 1 - self.confidence
        k_tp = int(((self.evaluation_scores >= t_star) & (self.evaluation_labels == 0)).sum())
        k_fp = int(((self.evaluation_scores >= t_star) & (self.evaluation_labels == 1)).sum())
        tpr_cp_lo = 0.0 if k_tp == 0 else float(beta_dist.ppf(gamma / 2, k_tp, n0 - k_tp + 1))
        fpr_cp_hi = 1.0 if k_fp == n1 else float(beta_dist.ppf(1 - gamma / 2, k_fp + 1, n1 - k_fp))
        if direction == "fwd":
            tpr, fpr = tpr_cp_lo, fpr_cp_hi
        else:  # "rev"
            tpr, fpr = 1.0 - fpr_cp_hi, 1.0 - tpr_cp_lo  # note: bound direction flips too
        if fpr > 0 and (tpr - self.delta_claimed) > 0:
            eps_lower = math.log((tpr - self.delta_claimed) / fpr)
        else:
            eps_lower = -np.inf  # no violation certifiable at this operating point
        eps_lower = max(0, eps_lower)
        return {
            'eps_lower_bound': float(eps_lower),
            'threshold': float(t_star),
            'direction': direction,
            'tpr_cp': tpr, 
            'fpr_cp': fpr, 
            'n0': n0, 
            'n1': n1
        }    

    
    @typechecked
    def run(self, verbose: bool = True) -> dict:
        # STEP 1: measure phase
        labels = np.zeros(self.n_trials)
        scores = np.zeros(self.n_trials)

        if isinstance(self.pairs, list):
            if isinstance(self.pairs[0][1], str):
                for i, (g0_file, g1_file) in tqdm(enumerate(self.pairs), total=len(self.pairs), desc="Scoring trials", disable=not verbose):
                    b = random.choice([0, 1]) # secret bit, balanced prior
                    if b == 0:
                        g_tilde_file = g0_file
                    else: # b == 1
                        g_tilde_file = g1_file
                    labels[i] = b
                    scores[i] = self.metric_fn(g_tilde_file)
            else:
                warnings.warn(
                    "Labels have been provided." + \
                    "Note we cannot confirm that for each trial, G, G' were selected independently and uniformly at random." + \
                    "This needed to have been ensured prior to this execution.",
                    UserWarning
                )
                for i, (g_file, label) in tqdm(enumerate(self.pairs), total=len(self.pairs), desc="Scoring trials", disable=not verbose):
                    labels[i] = label
                    scores[i] = self.metric_fn(g_file)
        else: # tuple -- live execution version
            for i in range(self.n_trials):
                b = random.choice([0, 1]) # secret bit, balanced prior
                command = shlex.split(self.pairs[b][0])
                subprocess.run(command, check=True)
                labels[i] = b
                scores[i] = self.metric_fn(self.pairs[b][1]) # Note: here, the PATH is given as the argument to the metric_fn.
            
        selection_n = int(self.selection_evaluation_split * self.n_trials)
        l_selection, s_selection, =  labels[:selection_n], scores[:selection_n]
        self.evaluation_labels, self.evaluation_scores = labels[selection_n:], scores[selection_n:]
        # STEP 2: selection phase
        t_star, direction = self.__selection_phase__(l_selection, s_selection)
        # STEP 3: evaluation phase
        self.results = self.__evaluation_phase__(t_star, direction)
        # Results        
        if verbose:
            print(f"RESULTS")
            print(f"    Trials: {self.n_trials} ({self.results['n0']} x G, {self.results['n1']} x G')")
            print(f"    Claimed Epsilon: {self.epsilon_claimed}")
            print(f"    Empirical Epsilon (lower bound, {int(self.confidence*100)}% confidence): {self.results['eps_lower_bound']:.4f}")
        if self.results['eps_lower_bound'] > self.epsilon_claimed:
            print("\033[91m✗ " + f"DP claim falsified, with {int(self.confidence*100)}% confidence." + "\033[0m")
        else:
            print("\033[92m✓ " + f"DP claim not falsified." + "\033[0m")
        print()
        return self.results


    @typechecked
    def plot(self, metric_name: str = "", title: str = "") -> tuple[Figure, Axes | np.ndarray]:
        if not self.results:
            raise Exception("Results not yet obtained. Run this object first (with .run()) to obtain them.")
        fig, axes = plt.subplots(1, 3, figsize=(9.5, 4), gridspec_kw={"width_ratios": [1, 1, 0.55]})
        fig.suptitle(f"{title + ' ' if title else ''}($\\epsilon = ${self.epsilon_claimed}, $\\delta = ${self.delta_claimed})", y=0.92)
        
        # Panel 1: score distributions
        s_G = self.evaluation_scores[self.evaluation_labels == 0]
        s_Gp = self.evaluation_scores[self.evaluation_labels == 1]
        bins = np.linspace(self.evaluation_scores.min(), self.evaluation_scores.max(), 40)
        axes[0].hist(s_G, bins=bins, alpha=1, color='steelblue', label=r'$G$', density=True)
        axes[0].hist(s_Gp, bins=bins, alpha=1, color='tomato', label=r"$G'$", density=True)
        axes[0].set_xlabel(r'$\tilde{G}$' + f" {metric_name if metric_name else 'Score'}")
        axes[0].set_ylabel('Density')
        axes[0].set_title("Distributions from\nUnderlying $G, G'$")
        axes[0].legend(framealpha=0.65)
        
        # Panel 2 – ROC curve vs DP-permitted region. (boundary: TPR = e^eps * FPR + delta)
        fprs, tprs, _ = DPGRMAuditor.__compute_roc__(self.evaluation_labels, self.evaluation_scores)
        fpr_line = np.linspace(0, 1, 400)
        tpr_dp_bnd = np.minimum(1.0, np.exp(self.epsilon_claimed) * fpr_line + self.delta_claimed)
        axes[1].plot(fprs, tprs, color='steelblue', linewidth=2.5, label='Empirical ROC', zorder=10)
        axes[1].plot(fpr_line, tpr_dp_bnd, color='tomato', linewidth=1.5, linestyle='--', label='DP Boundary')
        axes[1].fill_between(fpr_line, tpr_dp_bnd, 1.0, alpha=0.12, color='tomato', label='DP-Forbidden\nRegion')
        axes[1].plot([0, 1], [0, 1], color='gray', linestyle=':', linewidth=1.0)#, label='Random Classifier')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('Empirical ROC Curve vs.\nDP-Permitted Region')
        leg = axes[1].legend(loc='lower right', framealpha=0.65)
        leg.set_zorder(100)
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
        
        # Panel 3 - Claimed vs empirical epsilon. Bar chart comparing epsilon_claimed with the lower bound.
        labels_bar = ['Claimed $\\epsilon$', f'Empirical $\\epsilon$\n(lower bound)']#, {int(confidence*100)}% CI)']
        values_bar = [self.epsilon_claimed, self.results['eps_lower_bound']]
        colors_bar = ['mediumseagreen', 'tomato']
        bars = axes[2].bar(labels_bar, values_bar, color=colors_bar, edgecolor='black', linewidth=0.5, width=0.25)
        axes[2].axhline(self.results['eps_lower_bound'], color='tomato', linestyle='--', linewidth=1.5)
        for bar, val in zip(bars, values_bar):
            axes[2].text(bar.get_x() + bar.get_width() / 1.25, max(val, 0) + 0.025 * max(values_bar), f'{val:.3f}', ha='center', va='bottom',
                         bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
        axes[2].set_ylabel(r'Privacy Budget ($\epsilon$)')
        axes[2].set_title(f'Claimed vs.\nEmpirical $\\epsilon$')
        axes[2].set_ylim(bottom=min(0, min(values_bar)), top=(max(values_bar)*1.2))
        plt.tight_layout()
        plt.savefig(f"{title if title else 'plot'}.pdf", bbox_inches='tight')
        plt.close()
        return fig, axes
