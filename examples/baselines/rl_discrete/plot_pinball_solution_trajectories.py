import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def plot_all_trajectories_with_background(trajectory_folder, background_image_path):
    """
    Plots all solution trajectories (player + goal) from the specified folder over a 
    500x500 background image of the pinball domain.

    Each trajectory file is a dictionary with keys:
      - "player_x", "player_y": the player's trajectory coordinates in [0,100].
      - "goal_x",   "goal_y":   the goal's trajectory coordinates in [0,100].
    
    We multiply all coordinates by 5 so that [0,100] maps to [0,500].
    Subgoals (all but the final goal) are shown in black, and the final goal is shown in red.
    """
    # Load the background image (assumed shape: 500 x 500 x 3).
    bg_img = plt.imread(background_image_path)
    
    # Create the figure and axis.
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Display the background image, covering [0,500] in both x and y.
    ax.imshow(bg_img, extent=[0, 500, 0, 500], origin='lower')
    
    # List all pickle files in the trajectory folder.
    traj_files = sorted([f for f in os.listdir(trajectory_folder) if f.endswith('.pkl')])
    
    # Create a colormap with as many distinct colors as there are trajectories.
    cmap = plt.cm.get_cmap('viridis', len(traj_files))
    
    for idx, filename in enumerate(traj_files):
        file_path = os.path.join(trajectory_folder, filename)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # Ensure we have a dictionary with the expected keys
        if not isinstance(data, dict):
            continue
        
        # Convert lists to NumPy arrays.
        # Original coordinates are in [0,100]; scale them by 5 to match [0,500].
        player_x = np.array(data["player_x"]) * 5.0
        player_y = np.array(data["player_y"]) * 5.0
        goal_x   = np.array(data["goal_x"])   * 5.0
        goal_y   = np.array(data["goal_y"])   * 5.0
        
        # Plot the player's trajectory as a thick line.
        ax.plot(player_x, player_y, lw=4, color=cmap(idx),
                label=f"Trajectory {idx}")
        
        # If there are any goals, split them into subgoals (all but last) and final goal (last).
        if len(goal_x) > 0:
            subgoal_x = goal_x[:-1]
            subgoal_y = goal_y[:-1]
            final_goal_x = goal_x[-1]
            final_goal_y = goal_y[-1]

            # Plot subgoals in black, above the line (zorder=3).
            # ax.scatter(subgoal_x, subgoal_y, s=100, color='black', marker='*', edgecolor='black', zorder=3)
            
            # Plot final goal in red, also above the line.
            ax.scatter(final_goal_x, final_goal_y, s=400, color='black', marker='*', zorder=3, edgecolor='white')
    
    # ax.set_title("Pinball Domain: Player and Goal Trajectories (Scaled)")
    ax.legend(loc='upper right', fontsize=16)
    
    # Remove axis ticks for a cleaner look.
    plt.xticks([])
    plt.yticks([])
    
    plt.tight_layout()
    plt.savefig('pinball_trajectories.png')
    plt.close()

if __name__ == '__main__':
    trajectory_folder = "pinball_solution_trajectories_xy"
    background_image_path = "pinball_environment_500x500x3.png"
    plot_all_trajectories_with_background(trajectory_folder, background_image_path)
