"""Classifier firing frequency tracker for analyzing goal classifier usage."""

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List


class ClassifierFiringTracker:
    """Tracks which classifiers fire and how often during training."""
    
    def __init__(self, save_dir: str):
        """Initialize the classifier firing tracker.
        
        Args:
            save_dir: Directory to save visualization plots
        """
        self.save_dir = save_dir
        
        # Track firing counts per classifier
        self.firing_counts: Dict[int, int] = defaultdict(int)
        
        # Track firing history over time (classifier_id, step)
        self.firing_history: List[tuple] = []
        
        self.step_count = 0
        
        os.makedirs(save_dir, exist_ok=True)
        print(f'[ClassifierFiringTracker] Saving plots to {save_dir}')
    
    def log_firing(self, classifier_id: int):
        """Log a classifier firing event.
        
        Args:
            classifier_id: ID of the classifier that fired
        """
        self.step_count += 1
        self.firing_counts[classifier_id] += 1
        self.firing_history.append((classifier_id, self.step_count))
    
    def save_frequency_plot(self, filename: str = 'classifier_frequency.png'):
        """Save bar plot of classifier firing frequencies.
        
        Args:
            filename: Name of the file to save
        """
        if not self.firing_counts:
            print('[ClassifierFiringTracker] No firing data to plot')
            return
        
        # Sort by classifier ID
        classifier_ids = sorted(self.firing_counts.keys())
        counts = [self.firing_counts[cid] for cid in classifier_ids]
        
        plt.figure(figsize=(14, 6))
        plt.bar(classifier_ids, counts, alpha=0.7)
        plt.xlabel('Classifier ID')
        plt.ylabel('Number of Firings')
        plt.title(f'Classifier Firing Frequency (total firings={sum(counts)}, step={self.step_count})')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Highlight top 5 most fired classifiers
        if len(counts) > 0:
            top_5_threshold = sorted(counts, reverse=True)[min(4, len(counts)-1)]
            for i, (cid, count) in enumerate(zip(classifier_ids, counts)):
                if count >= top_5_threshold:
                    plt.bar(cid, count, alpha=0.9, color='red')
        
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f'[ClassifierFiringTracker] Saved frequency plot to {save_path}')
    
    def save_timeline_plot(self, filename: str = 'classifier_timeline.png', max_classifiers: int = 20):
        """Save timeline showing when each classifier fires.
        
        Args:
            filename: Name of the file to save
            max_classifiers: Maximum number of classifiers to show
        """
        if not self.firing_history:
            print('[ClassifierFiringTracker] No firing history to plot')
            return
        
        # Get top N most frequently fired classifiers
        top_classifiers = sorted(self.firing_counts.items(), 
                                key=lambda x: x[1], reverse=True)[:max_classifiers]
        top_ids = set([cid for cid, _ in top_classifiers])
        
        # Filter history to only include top classifiers
        filtered_history = [(cid, step) for cid, step in self.firing_history if cid in top_ids]
        
        if not filtered_history:
            return
        
        classifier_ids = [cid for cid, _ in filtered_history]
        steps = [step for _, step in filtered_history]
        
        plt.figure(figsize=(14, 8))
        plt.scatter(steps, classifier_ids, alpha=0.5, s=10)
        plt.xlabel('Training Step')
        plt.ylabel('Classifier ID')
        plt.title(f'Classifier Firing Timeline (top {len(top_ids)} classifiers)')
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f'[ClassifierFiringTracker] Saved timeline plot to {save_path}')
    
    def save_distribution_plot(self, filename: str = 'classifier_distribution.png'):
        """Save histogram of firing count distribution.
        
        Args:
            filename: Name of the file to save
        """
        if not self.firing_counts:
            print('[ClassifierFiringTracker] No firing data to plot')
            return
        
        counts = list(self.firing_counts.values())
        
        plt.figure(figsize=(10, 6))
        plt.hist(counts, bins=min(50, len(set(counts))), alpha=0.7, edgecolor='black')
        plt.xlabel('Number of Firings')
        plt.ylabel('Number of Classifiers')
        plt.title(f'Distribution of Classifier Firing Counts (n={len(counts)} classifiers)')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        mean_firings = np.mean(counts)
        median_firings = np.median(counts)
        plt.axvline(mean_firings, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_firings:.1f}')
        plt.axvline(median_firings, color='g', linestyle='--', linewidth=2, label=f'Median: {median_firings:.1f}')
        plt.legend()
        
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f'[ClassifierFiringTracker] Saved distribution plot to {save_path}')
    
    def get_stats(self) -> Dict:
        """Get statistics about classifier firings.
        
        Returns:
            Dictionary with firing statistics
        """
        if not self.firing_counts:
            return {
                'step_count': self.step_count,
                'num_classifiers': 0,
                'total_firings': 0
            }
        
        counts = list(self.firing_counts.values())
        
        return {
            'step_count': self.step_count,
            'num_classifiers': len(self.firing_counts),
            'total_firings': sum(counts),
            'mean_firings': np.mean(counts),
            'median_firings': np.median(counts),
            'max_firings': max(counts),
            'min_firings': min(counts),
            'most_fired_classifier': max(self.firing_counts.items(), key=lambda x: x[1])[0]
        }
