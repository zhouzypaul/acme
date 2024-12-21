import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from comparison_plotting_utils import *
from plotting_utils import get_summary_data

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'legend.fontsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'grid.color': 'gray',
    'grid.linestyle': '--',
    'grid.alpha': 0.5,
    'lines.linewidth': 2,
    'lines.markersize': 6
})


BASE_DIR = os.path.expanduser('~/git-repos/acme/examples/baselines/rl_discrete/')


SOKOBAN_EXPERIMENTS = {
    os.path.join(BASE_DIR, 'local_testing/sweeps/Sokoban/dsg_sokoban_using_background_rext5'): "IM-DSG",
    os.path.join(BASE_DIR, 'local_results/sokoban-v0/cfn/spi_sweep1'): "CFN",
    os.path.join(BASE_DIR, 'local_testing/sweeps/Sokoban/r2d2_baseline1'): "R2D2",
}

# Collecting CFN MiniGrid results again, confirmed that the R2D2 configs were correct.
# Made the CFN target_update_period same as DSG, but results are worse. Also tuned the policy learning rate.
# Those results are in local_results/minigrid/kc/cfn/r2d2_learning_rate_sweep1 and are worse.

MINIGRID_KC_EXPERIMENTS = {
    os.path.join(BASE_DIR, 'local_testing/sweeps/kc/developing_online_classifiers_target_update_period_sweep1/'): "Abstract Subgoals",
    os.path.join(BASE_DIR, 'local_results/minigrid/kc/cfn/policy_spi_sweep2'): "CFN",
    os.path.join(BASE_DIR, 'local_testing/sweeps/kc/r2d2_baseline1'): "R2D2",
}

MINIGRID_DOORKEY_EXPERIMENTS = {
    os.path.join(BASE_DIR, 'local_testing/sweeps/doorkey/dsg_doorkey_thesis1'): "IM-DSG",
    os.path.join(BASE_DIR, 'local_results/minigrid/doorkey/cfn/spisweep_thesis2'): "CFN",
    os.path.join(BASE_DIR, 'local_testing/sweeps/doorkey/r2d2_baseline2'): "R2D2",
}

LARGE_TAXI_EXPERIMENTS = {
    os.path.join(BASE_DIR, 'local_testing/sweeps/larger_taxi/developing_online_classifiers_target_update_period_sweep1'): "Abstract Subgoals",
    os.path.join(BASE_DIR, 'local_results/large_taxi/cfn/policy_spi_sweep1'): "CFN",
    os.path.join(BASE_DIR, 'local_testing/sweeps/kc/r2d2_baseline1'): "R2D2",
}

MONTE_EXPERIMENTS = {
    os.path.join(BASE_DIR, 'local_testing/sweeps/monte/dsg_monte_thesis1'): "IM-DSG",
}


domain2experiments = {
    'sokoban-v0': SOKOBAN_EXPERIMENTS,
    'MiniGrid-KeyCorridorS5R3-v0': MINIGRID_KC_EXPERIMENTS,
    'MiniGrid-DoorKey-16x16-v0': MINIGRID_DOORKEY_EXPERIMENTS,
    'VisualTaxi-10x10': LARGE_TAXI_EXPERIMENTS,
}


def get_config(acme_id, group_key):
    try:
        return re.search(f".*?([+-]+{group_key}|{group_key}_[^_]*)_.*", acme_id).group(1)
    except: # if its at the end of the id name
        return re.search(f".*?([+-]+{group_key}|{group_key}_[^_]*)$", acme_id).group(1)

def default_make_key(log_dir_name, group_keys):
    keys = [get_config(log_dir_name, group_key) for group_key in group_keys]
    key = "_".join(keys)
    return key


def extract_log_dirs(id_to_csv, group_keys=("rewardscale",), xkey='actor_steps', ykey='episode_return'):

    # Map config to a list of curves
    log_dir_map = defaultdict(list)

    for acme_id, csv_path in id_to_csv.items():
        try:
            keys = [get_config(acme_id, group_key) for group_key in group_keys]
            key = "_".join(keys)
            key = default_make_key(acme_id, group_keys)
            # key = get_config(log_dir, group_key)
            frames, returns = get_summary_data(csv_path, xkey=xkey, ykey=ykey)
            log_dir_map[key].append((frames, returns))
        except Exception as e:
            print(f"Could not extract {acme_id}")
            print(e)


            # import ipdb; ipdb.set_trace()
            # print('why')

    return log_dir_map

def extract_log_dirs_group_func(id_to_csv, group_func=lambda x: x, xkey='actor_steps', ykey='episode_return'):

    # Map config to a list of curves
    log_dir_map = defaultdict(list)

    for acme_id, csv_path in id_to_csv.items():
        try:
            key = group_func(acme_id)
            if key is None:
                continue
            # key = get_config(log_dir, group_key)
            # frames, returns = get_summary_data(csv_path, xkey='actor_steps', ykey='episode_return')
            frames, returns = get_summary_data(csv_path, xkey=xkey, ykey=ykey)
            log_dir_map[key].append((frames, returns))
        except Exception as e:
            print(f'Exception: {e}')
            print(f"Could not extract from {acme_id}")

    return log_dir_map


def plot_comparison_learning_curves(
    # id_to_csv, # dict
    base_dir, #str
    selected_acme_ids=None,
    # experiment_name=None,
    # stat='eval_episode_lengths',
    group_keys=("rewardscale",),
    group_func=None,
    filter_func=None, # Only include things that are "true" in filter
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

    # import seaborn as sns
    # NUM_COLORS=100
    # clrs = sns.color_palette('husl', n_colors=NUM_COLORS)
    # sns.set_palette(clrs)
    id_to_csv = gather_csv_files_from_base_dir(base_dir=base_dir, selected_acme_ids=selected_acme_ids, log_type=log_file_type)


    assert isinstance(group_keys, (tuple, list)), f"{type(group_keys)} should be tuple or list"

    ylabel = ylabel or "Average Return"

    if log_dir_path_map is None:
        if group_func is not None:
            log_dir_path_map = extract_log_dirs_group_func(id_to_csv=id_to_csv, group_func=group_func, xkey=xkey, ykey=ykey)
        else:
            log_dir_path_map = extract_log_dirs(id_to_csv=id_to_csv, group_keys=group_keys, xkey=xkey, ykey=ykey)

    for config in log_dir_path_map:
        if config is None:
            continue
        if filter_func and not filter_func(config):
            continue
        curves = log_dir_path_map[config]
        print(config)
        curves = [(remove_strings_from_scores(frames, returns)) for (frames, returns) in curves]
        for curve in curves:
            print(f"\t[+] Num points in curve: {len(curve[0])}")
            print(f'\t[+] Max x-axis val: {np.max(curve[0])}')
        truncated_xs, truncated_all_ys = truncate_and_interpolate(curves, max_frames=truncate_max_frames, min_frames=truncate_min_frames)
        print(truncated_xs.shape)
        # import ipdb; ipdb.set_trace()
        if len(truncated_all_ys) < min_seeds:
            continue
        if smoothen and smoothen > 0 and len(truncated_xs) < smoothen:
            continue

        if min_final_val is not None:
            if np.array(truncated_all_ys)[:, -1].mean() <= min_final_val:
                continue

        # score_array = np.array(truncated_all_ys)
        print(np.max(truncated_all_ys))
        generate_plot(
            # score_array,
            truncated_xs,
            truncated_all_ys,
            label=EXPERIMENTS[base_dir],
            smoothen=smoothen,
            linewidth=linewidth,
            all_seeds=all_seeds)
    
    if xmax:
        plt.xlim(0, xmax)

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
    parser.add_argument('--domain', type=str, default='sokoban-v0')
    parser.add_argument('--max_frames', type=int, default=None)
    args = parser.parse_args()

    EXPERIMENTS = domain2experiments[args.domain]
    
    save_path = os.path.join(
        BASE_DIR,
        args.save_filename if 'png' in args.save_filename else f"{args.process_name}_{args.save_filename}.png")
    
    def lr_group_func(acme_id):
        if "spi_3" not in acme_id:
            return None
        return get_config(acme_id, "learningrate")
    def rc_group_func(acme_id):
        if "spi_3" not in acme_id:
            return None
        return get_config(acme_id, "rewardcoefficient")
    def new_group_func(acme_id):
        # if "size3" not in acme_id or "size3_5" in acme_id:
        #     return None
        return default_make_key(acme_id, args.hyperparams)

    plt.figure(figsize=(10, 8))

    for base_dir, experiment in EXPERIMENTS.items():
        plot_comparison_learning_curves(
            base_dir=base_dir,
            # save_path=None,
            # show=True,
            # save_path=save_path,
            show=False,
            # group_keys=("cfnmaxreplaysize", "cfn_use_forgetting"),
            # group_keys=("learningrate", ),
            group_func=new_group_func,
            # filter_func=lambda acme_id: "size3" in acme_id and "size3_5" not in acme_id,
            smoothen=args.smoothen,
            # smoothen=False,
            # truncate_min_frames=50_000_000,
            # min_seeds=5,
            all_seeds=args.all_seeds,
            log_file_type='evaluator' if 'cfn' in base_dir or 'r2d2' in base_dir else 'actor',
            selected_acme_ids=args.selected_acme_ids,
            title=f"{args.domain}",
            xmax=args.max_frames,
            )
    plt.legend()
    plt.savefig(f"combined_learning_curves/{args.domain}.png")
    plt.close()
