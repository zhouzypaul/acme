"""Policy behavior tracker for analyzing agent decision-making in RL training."""

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from typing import Optional, List, Tuple


class PolicyBehaviorTracker:
    """Tracks and visualizes policy behavior metrics during training."""
    
    def __init__(self, save_dir: str, window_size: int = 1000):
        """Initialize the policy behavior tracker.
        
        Args:
            save_dir: Directory to save visualization plots
            window_size: Number of recent steps to keep in memory
        """
        self.save_dir = save_dir
        self.window_size = window_size
        
        # Track action entropy (measure of policy randomness)
        self.action_entropies = deque(maxlen=window_size)
        self.entropy_steps = deque(maxlen=window_size)
        
        # Track value estimates (Q-values or V-values)
        self.value_estimates = deque(maxlen=window_size)
        self.value_steps = deque(maxlen=window_size)
        
        self.step_count = 0
        
        os.makedirs(save_dir, exist_ok=True)
        print(f'[PolicyBehaviorTracker] Saving plots to {save_dir}')
    
    def log_action_entropy(self, entropy: float):
        """Log action entropy from policy distribution.
        
        Args:
            entropy: Entropy of action distribution (higher = more random)
        """
        self.step_count += 1
        self.action_entropies.append(entropy)
        self.entropy_steps.append(self.step_count)
    
    def log_value_estimate(self, value: float):
        """Log value function estimate.
        
        Args:
            value: Value estimate (Q-value or V-value)
        """
        if not self.action_entropies or len(self.value_steps) < len(self.entropy_steps):
            # Sync with entropy logging
            self.value_estimates.append(value)
            self.value_steps.append(self.step_count)
    
    def save_entropy_plot(self, filename: str = 'action_entropy.png'):
        """Save plot of action entropy over time.
        
        Args:
            filename: Name of the file to save
        """
        if not self.action_entropies:
            print('[PolicyBehaviorTracker] No entropy data to plot')
            return
        
        plt.figure(figsize=(12, 6))
        plt.plot(list(self.entropy_steps), list(self.action_entropies), alpha=0.6, linewidth=0.5)
        
        # Add moving average
        if len(self.action_entropies) > 100:
            window = min(100, len(self.action_entropies) // 10)
            moving_avg = np.convolve(self.action_entropies, np.ones(window)/window, mode='valid')
            avg_steps = list(self.entropy_steps)[window-1:]
            plt.plot(avg_steps, moving_avg, 'r-', linewidth=2, label=f'{window}-step MA')
            plt.legend()
        
        plt.xlabel('Training Step')
        plt.ylabel('Action Entropy')
        plt.title(f'Policy Action Entropy (step={self.step_count})')
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f'[PolicyBehaviorTracker] Saved entropy plot to {save_path}')
    
    def save_value_plot(self, filename: str = 'value_estimates.png'):
        """Save plot of value estimates over time.
        
        Args:
            filename: Name of the file to save
        """
        if not self.value_estimates:
            print('[PolicyBehaviorTracker] No value data to plot')
            return
        
        plt.figure(figsize=(12, 6))
        plt.plot(list(self.value_steps), list(self.value_estimates), alpha=0.6, linewidth=0.5)
        
        # Add moving average
        if len(self.value_estimates) > 100:
            window = min(100, len(self.value_estimates) // 10)
            moving_avg = np.convolve(self.value_estimates, np.ones(window)/window, mode='valid')
            avg_steps = list(self.value_steps)[window-1:]
            plt.plot(avg_steps, moving_avg, 'r-', linewidth=2, label=f'{window}-step MA')
            plt.legend()
        
        plt.xlabel('Training Step')
        plt.ylabel('Value Estimate')
        plt.title(f'Value Function Estimates (step={self.step_count})')
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f'[PolicyBehaviorTracker] Saved value plot to {save_path}')
    
    def save_combined_plot(self, filename: str = 'policy_behavior.png'):
        """Save combined plot with both entropy and value.
        
        Args:
            filename: Name of the file to save
        """
        if not self.action_entropies and not self.value_estimates:
            print('[PolicyBehaviorTracker] No data to plot')
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Entropy plot
        if self.action_entropies:
            ax1.plot(list(self.entropy_steps), list(self.action_entropies), alpha=0.6, linewidth=0.5)
            if len(self.action_entropies) > 100:
                window = min(100, len(self.action_entropies) // 10)
                moving_avg = np.convolve(self.action_entropies, np.ones(window)/window, mode='valid')
                avg_steps = list(self.entropy_steps)[window-1:]
                ax1.plot(avg_steps, moving_avg, 'r-', linewidth=2, label=f'{window}-step MA')
                ax1.legend()
            ax1.set_xlabel('Training Step')
            ax1.set_ylabel('Action Entropy')
            ax1.set_title('Policy Action Entropy')
            ax1.grid(True, alpha=0.3)
        
        # Value plot
        if self.value_estimates:
            ax2.plot(list(self.value_steps), list(self.value_estimates), alpha=0.6, linewidth=0.5)
            if len(self.value_estimates) > 100:
                window = min(100, len(self.value_estimates) // 10)
                moving_avg = np.convolve(self.value_estimates, np.ones(window)/window, mode='valid')
                avg_steps = list(self.value_steps)[window-1:]
                ax2.plot(avg_steps, moving_avg, 'r-', linewidth=2, label=f'{window}-step MA')
                ax2.legend()
            ax2.set_xlabel('Training Step')
            ax2.set_ylabel('Value Estimate')
            ax2.set_title('Value Function Estimates')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f'[PolicyBehaviorTracker] Saved combined plot to {save_path}')
    
    def get_stats(self) -> dict:
        """Get statistics about policy behavior.
        
        Returns:
            Dictionary with behavior statistics
        """
        stats = {
            'step_count': self.step_count,
            'num_entropy_samples': len(self.action_entropies),
            'num_value_samples': len(self.value_estimates),
        }
        
        if self.action_entropies:
            stats['mean_entropy'] = np.mean(self.action_entropies)
            stats['std_entropy'] = np.std(self.action_entropies)
        
        if self.value_estimates:
            stats['mean_value'] = np.mean(self.value_estimates)
            stats['std_value'] = np.std(self.value_estimates)
        
        return stats
