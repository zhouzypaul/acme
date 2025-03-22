import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import seaborn as sns
sns.set("talk", "white")

from comparison_plotting_utils import *
from plotting_utils import get_summary_data

BASE_DIR = os.path.expanduser('~/git-repos/acme/examples/baselines/rl_discrete/')

SOKOBAN_EXPERIMENTS = {
    os.path.join(BASE_DIR, 'local_testing/sweeps/Sokoban/dsg_sokoban_using_background_rext5'): "IM-DSG",
    os.path.join(BASE_DIR, 'local_results/sokoban-v0/cfn/spi_sweep1'): "CFN",
    os.path.join(BASE_DIR, 'local_testing/sweeps/Sokoban/r2d2_baseline1'): "R2D2",
}

MINIGRID_KC_EXPERIMENTS = {
    os.path.join(BASE_DIR, 'local_testing/sweeps/kc/dsg_kc_thesis1'): "IM-DSG",
    os.path.join(BASE_DIR, 'local_results/minigrid/kc/cfn/policy_spi_sweep2'): "CFN",
    os.path.join(BASE_DIR, 'local_testing/sweeps/kc/r2d2_baseline1'): "R2D2",
}

MINIGRID_DOORKEY_EXPERIMENTS = {
    os.path.join(BASE_DIR, 'local_testing/sweeps/doorkey/dsg_doorkey_thesis1'): "IM-DSG",
    os.path.join(BASE_DIR, 'local_results/minigrid/doorkey/cfn/spisweep_thesis2'): "CFN",
    os.path.join(BASE_DIR, 'local_testing/sweeps/doorkey/r2d2_baseline2'): "R2D2",
}

TAXI_EXPERIMENTS = {
    os.path.join(BASE_DIR, 'local_testing/sweeps/taxi/dsg_background_rewards_coeff_sweep1'): "IM-DSG",
    os.path.join(BASE_DIR, 'local_testing/sweeps/taxi/cfn_small_taxi_baseline1'): "CFN",
    os.path.join(BASE_DIR, 'local_testing/sweeps/taxi/r2d2_small_taxi_baseline1'): "R2D2",
}

# Change domain order: doorkey, keycorridor, visual-taxi, sokoban.
ordered_domains = [
    ("MiniGrid-DoorKey-16x16", MINIGRID_DOORKEY_EXPERIMENTS),
    ("MiniGrid-KeyCorridorS5R3", MINIGRID_KC_EXPERIMENTS),
    ("Visual-Taxi", TAXI_EXPERIMENTS),
    ("sokoban-v0", SOKOBAN_EXPERIMENTS),
]

domain2label = {
    "MiniGrid-DoorKey-16x16-v0": "MiniGrid-DoorKey-16x16",
    "MiniGrid-KeyCorridorS5R3-v0": "MiniGrid-KeyCorridorS5R3",
    "Visual-Taxi-v0": "Visual-Taxi",
    "sokoban-v0": "Sokoban-v0",
}


def get_config(acme_id, group_key):
    try:
        return re.search(f".*?([+-]+{group_key}|{group_key}_[^_]*)_.*", acme_id).group(1)
    except:  # if it's at the end of the id name
        return re.search(f".*?([+-]+{group_key}|{group_key}_[^_]*)$", acme_id).group(1)

def default_make_key(log_dir_name, group_keys):
    keys = [get_config(log_dir_name, group_key) for group_key in group_keys]
    key = "_".join(keys)
    return key


def extract_log_dirs(id_to_csv, group_keys=("rewardscale",), xkey='actor_steps', ykey='epifsode_return'):
    log_dir_map = defaultdict(list)
    for acme_id, csv_path in id_to_csv.items():
        try:
            keys = [get_config(acme_id, group_key) for group_key in group_keys]
            key = "_".join(keys)
            key = default_make_key(acme_id, group_keys)
            frames, returns = get_summary_data(csv_path, xkey=xkey, ykey=ykey)
            log_dir_map[key].append((frames, returns))
        except Exception as e:
            print(f"Could not extract {acme_id}")
            print(e)
    return log_dir_map

def extract_log_dirs_group_func(id_to_csv, group_func=lambda x: x, xkey='actor_steps', ykey='episode_return'):
    log_dir_map = defaultdict(list)
    for acme_id, csv_path in id_to_csv.items():
        try:
            key = group_func(acme_id)
            if key is None:
                continue
            frames, returns = get_summary_data(csv_path, xkey=xkey, ykey=ykey)
            log_dir_map[key].append((frames, returns))
        except Exception as e:
            print(f'Exception: {e}')
            print(f"Could not extract from {acme_id}")
    return log_dir_map

def plot_comparison_learning_curves(
    base_dir,
    selected_acme_ids=None,
    group_keys=("rewardscale",),
    group_func=None,
    filter_func=None,
    save_path=None,
    show=True,
    smoothen=10,
    log_dir_path_map=None,
    uniform_truncate=False,
    truncate_max_frames=-1,
    truncate_min_frames=-1,
    ylabel=False,
    legend_loc=None,
    linewidth=2,
    min_seeds=1,
    all_seeds=False,
    title=None,
    min_final_val=None,
    log_file_type='evaluator',
    xkey='actor_steps',
    ykey='episode_return',
    xmax=None,
):
    id_to_csv = gather_csv_files_from_base_dir(base_dir=base_dir, selected_acme_ids=selected_acme_ids, log_type=log_file_type)

    assert isinstance(group_keys, (tuple, list)), f"{type(group_keys)} should be tuple or list"

    ylabel = ylabel or "Average Undiscounted Return"

    if log_dir_path_map is None:
        if group_func is not None:
            log_dir_path_map = extract_log_dirs_group_func(id_to_csv=id_to_csv, group_func=group_func, xkey=xkey, ykey=ykey)
        else:
            log_dir_path_map = extract_log_dirs(id_to_csv=id_to_csv, group_keys=group_keys, xkey=xkey, ykey=ykey)

    sorted_configs = sorted(log_dir_path_map.keys())

    for idx, config in enumerate(sorted_configs):
        if config is None:
            continue
        if filter_func and not filter_func(config):
            continue
        curves = log_dir_path_map[config]
        print(config)
        curves = [remove_strings_from_scores(frames, returns) for (frames, returns) in curves]
        for curve in curves:
            print(f"\t[+] Num points in curve: {len(curve[0])}")
            print(f'\t[+] Max x-axis val: {np.max(curve[0])}')
        truncated_xs, truncated_all_ys = truncate_and_interpolate(curves, max_frames=truncate_max_frames, min_frames=truncate_min_frames)
        print(truncated_xs.shape)
        if len(truncated_all_ys) < min_seeds:
            continue
        if smoothen and smoothen > 0 and len(truncated_xs) < smoothen:
            continue

        if min_final_val is not None:
            if np.array(truncated_all_ys)[:, -1].mean() <= min_final_val:
                continue

        print(np.max(truncated_all_ys))
        generate_plot(
            truncated_xs,
            truncated_all_ys,
            label=EXPERIMENTS[base_dir],
            smoothen=smoothen,
            linewidth=linewidth,
            all_seeds=all_seeds,
        )
    
    if xmax:
        plt.xlim(0, xmax)

    # Always set these (we may override later in the main loop)
    plt.xlabel("Frames")
    plt.ylabel(ylabel)
    if title:
        plt.title(title)

    if show:
        if legend_loc:
            plt.legend(loc=legend_loc)
        else:
            plt.legend()
        plt.show()
    
    if save_path is not None:
        plt.legend()
        plt.savefig(save_path)
        plt.close()


def get_rmse_for_each_iteration(count_dict):
    exact, approx = get_true_vs_approx(count_dict, "bonus")
    assert len(exact) == len(approx)
    exact = np.asarray(exact)
    approx = np.asarray(approx)
    sq_errors = (exact-approx) ** 2
    root_mean_sq_errors = np.mean(sq_errors) ** 0.5
    return root_mean_sq_errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_filename', type=str, default='learning_curves')
    parser.add_argument('--smoothen', type=int, default=100)
    parser.add_argument('--hyperparams', nargs='+', default='nsigmasthresholdforgoalcreation')
    parser.add_argument('--all_seeds', action='store_true', default=False)
    parser.add_argument('--process_name', type=str, default='actor', help='actor or evaluator')
    parser.add_argument('--selected_acme_ids', nargs='+', default=None)
    # The --domain argument is no longer used since we plot all domains together.
    parser.add_argument('--max_frames', type=int, default=None)
    args = parser.parse_args()

    def lr_group_func(acme_id):
        if "spi_3" not in acme_id:
            return None
        return get_config(acme_id, "learningrate")
    def rc_group_func(acme_id):
        if "spi_3" not in acme_id:
            return None
        return get_config(acme_id, "rewardcoefficient")
    def new_group_func(acme_id):
        return default_make_key(acme_id, args.hyperparams)
    def algorithm_group_func(acme_id):
        if "dsg" in acme_id:
            return "IM-DSG"
        elif "r2d2" in acme_id:
            return "R2D2"
        elif "cfn" in acme_id:
            return "CFN"
        return new_group_func(acme_id)

    # Create a figure with 1 row x 4 columns of subplots.
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    axes = axes.flatten()

    # Loop over each domain in the new order.
    for idx, (ax, (domain, experiments)) in enumerate(zip(axes, ordered_domains)):
        plt.sca(ax)
        # Set the global EXPERIMENTS mapping for this subplot.
        EXPERIMENTS = experiments  
        # For each experiment in the domain, plot its learning curves.
        for base_dir, experiment_label in experiments.items():
            # For taxi, cap frames at 1e7.
            cur_xmax = 1e7 if domain == "Visual-Taxi" else args.max_frames
            plot_comparison_learning_curves(
                base_dir=base_dir,
                show=False,
                group_func=algorithm_group_func,
                smoothen=args.smoothen,
                all_seeds=args.all_seeds,
                log_file_type='evaluator' if ('cfn' in base_dir or 'r2d2' in base_dir) else 'actor',
                selected_acme_ids=args.selected_acme_ids,
                title=None,
                xmax=cur_xmax,
            )
        ax.set_title(domain)
        ax.set_xlabel("Frames")

        if idx == 0:
            ax.set_ylabel("Reward")
        else:
            ax.set_ylabel("")
        # Remove individual legends so we can add a common one later.

    # Add a common ylabel on the very left of the figure.
    # fig.text(0.04, 0.5, "Average Undiscounted Return", va='center', rotation='vertical')

    # Adjust layout to leave space at the bottom for the legend.
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.28)

    # Collect legend handles and labels from all subplots.
    handles = []
    labels = []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)
    # Create a common horizontal legend at the bottom.
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), frameon=False)

    combined_save_path = os.path.join("combined_learning_curves", f"combined_{args.save_filename}.png")
    plt.savefig(combined_save_path)
    plt.close()
