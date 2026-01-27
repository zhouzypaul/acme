"""State visitation tracker for debugging agent exploration in Montezuma's Revenge."""

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Optional, Dict, List, Tuple


class StateVisitationTracker:
    """Tracks and visualizes where the agent visits during training."""
    
    def __init__(self, save_dir: str, max_positions: int = 10000):
        """Initialize the state visitation tracker.
        
        Args:
            save_dir: Directory to save visualization plots
            max_positions: Maximum number of positions to track (for memory efficiency)
        """
        self.save_dir = save_dir
        self.max_positions = max_positions
        self.positions: List[Tuple[int, int]] = []  # List of (x, y) tuples
        self.room_positions: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self.step_count = 0
        
        os.makedirs(save_dir, exist_ok=True)
        print(f'[StateVisitationTracker] Saving plots to {save_dir}')
    
    def log_position(self, x: int, y: int, room: Optional[int] = None):
        """Log a position visited by the agent.
        
        Args:
            x: X coordinate
            y: Y coordinate  
            room: Optional room number
        """
        self.step_count += 1
        
        # Add to overall positions (with memory limit)
        if len(self.positions) < self.max_positions:
            self.positions.append((x, y))
        
        # Add to room-specific positions
        if room is not None:
            if len(self.room_positions[room]) < self.max_positions:
                self.room_positions[room].append((x, y))
    
    def save_plot(self, filename: str = 'state_visitation.png'):
        """Save a scatter plot of all visited positions.
        
        Args:
            filename: Name of the file to save
        """
        if not self.positions:
            print('[StateVisitationTracker] No positions to plot')
            return
        
        x_coords = [pos[0] for pos in self.positions]
        y_coords = [pos[1] for pos in self.positions]
        
        plt.figure(figsize=(12, 10))
        plt.scatter(x_coords, y_coords, alpha=0.3, s=1)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title(f'State Visitation (n={len(self.positions)} positions, step={self.step_count})')
        plt.grid(True, alpha=0.3)
        
        # Set fixed axis limits for Montezuma's Revenge
        plt.xlim(0, 180)
        plt.ylim(0, 270)
        
        # Invert y-axis to match game coordinates
        plt.gca().invert_yaxis()
        
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f'[StateVisitationTracker] Saved plot to {save_path}')
    
    def save_room_plots(self, max_rooms: int = 10):
        """Save separate scatter plots for each room.
        
        Args:
            max_rooms: Maximum number of room plots to save
        """
        if not self.room_positions:
            print('[StateVisitationTracker] No room data to plot')
            return
        
        # Sort rooms by number of visits
        sorted_rooms = sorted(self.room_positions.items(), 
                            key=lambda x: len(x[1]), reverse=True)
        
        for room_num, positions in sorted_rooms[:max_rooms]:
            if not positions:
                continue
            
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            plt.figure(figsize=(10, 8))
            plt.scatter(x_coords, y_coords, alpha=0.5, s=2)
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.title(f'Room {room_num} Visitation (n={len(positions)} visits)')
            plt.grid(True, alpha=0.3)
            
            # Set fixed axis limits for Montezuma's Revenge
            plt.xlim(0, 180)
            plt.ylim(0, 270)
            plt.gca().invert_yaxis()
            
            save_path = os.path.join(self.save_dir, f'room_{room_num}_visitation.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f'[StateVisitationTracker] Saved {min(len(sorted_rooms), max_rooms)} room plots')
    
    def get_stats(self) -> Dict:
        """Get statistics about state visitation.
        
        Returns:
            Dictionary with visitation statistics
        """
        unique_positions = len(set(self.positions))
        rooms_visited = len(self.room_positions)
        
        return {
            'total_positions': len(self.positions),
            'unique_positions': unique_positions,
            'rooms_visited': rooms_visited,
            'step_count': self.step_count,
            'coverage': unique_positions / len(self.positions) if self.positions else 0
        }
