import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from comparison_plotting_utils import *
from plotting_utils import get_summary_data, extract_exploration_amounts, load_count_dict, get_true_vs_approx


def get_config(acme_id, group_key):
    try:
        return re.search(f".*({group_key}_[^_]*)_.*", acme_id).group(1)
    except: # if its at the end of the id name
        return re.search(f".*({group_key}_[^_]*)$", acme_id).group(1)


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
        except:
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
    ykey='episode_return'):

    # import seaborn as sns
    # NUM_COLORS=100
    # clrs = sns.color_palette('husl', n_colors=NUM_COLORS)
    # sns.set_palette(clrs)
    id_to_csv = gather_csv_files_from_base_dir(base_dir=base_dir, selected_acme_ids=selected_acme_ids, log_type=log_file_type)

    assert isinstance(group_keys, (tuple, list)), f"{type(group_keys)} should be tuple or list"
    if save_path is not None:
        plt.figure(figsize=(24,12))


    ylabel = ylabel or "Average Return"

    if log_dir_path_map is None:
        if group_func is not None:
            log_dir_path_map = extract_log_dirs_group_func(id_to_csv=id_to_csv, group_func=group_func, xkey=xkey, ykey=ykey)
        else:
            log_dir_path_map = extract_log_dirs(id_to_csv=id_to_csv, group_keys=group_keys, xkey=xkey, ykey=ykey)

    for config in log_dir_path_map:
        if config is None:
            continue
        curves = log_dir_path_map[config]
        print(config)
        for curve in curves:
            print(f"\t{len(curve[0])}")
        truncated_xs, truncated_all_ys = truncate_and_interpolate(curves, max_frames=truncate_max_frames, min_frames=truncate_min_frames)
        if len(truncated_all_ys) < min_seeds:
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
            label=config,
            smoothen=smoothen,
            linewidth=linewidth,
            all_seeds=all_seeds)
    
    # plt.grid()
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
    base_dir = "/Users/slobal1/Code/ML/acme_testing/acme_paul/examples/baselines/rl_discrete/ccv_results/results/monte/spi_lr_rscale_sweep"
    def lr_group_func(acme_id):
        if "spi_3" not in acme_id:
            return None
        return get_config(acme_id, "learningrate")
    def rc_group_func(acme_id):
        if "spi_3" not in acme_id:
            return None
        return get_config(acme_id, "rewardcoefficient")

    plot_comparison_learning_curves(
        base_dir=base_dir,
        # save_path=None,
        # show=True,
        save_path="/Users/slobal1/Downloads/matplotlib_plots/r2d2/rnd_rewardcoeff_spi3.png",
        show=False,
        # group_keys=("rewardcoefficient", "spi", "learningrate"),
        # group_keys=("learningrate", ),
        group_func=rc_group_func,
        # smoothen=10,
        smoothen=False,
        truncate_min_frames=50_000_000,
        # min_seeds=5,
        all_seeds=False,
        title="R2D2 RND sweep"
        )
