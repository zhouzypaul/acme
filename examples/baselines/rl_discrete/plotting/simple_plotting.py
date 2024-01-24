import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d


def gather_csv_files_from_base_dir(base_dir=None, selected_acme_ids=None, process_name='evaluator'):
    """
    Here the base_dir is assumed to be ~/acme, and under the base_dir are the dirs
    named with `acme_id`, under which has `logs/evaluator/logs.csv`.

    Args:
        base_dir (_type_): acme results dir, if None use the default acme setting
        selected_acme_ids (_type_): if None, use all acme_ids, otherwise use the selected ones
        
    Returns:
        a dict of {acme_id: csv_path}
    """
    if base_dir is None:
        base_dir = os.path.expanduser('~/acme')

    # find all subdir and get the acme_id of all
    id_to_csv = {}
    for acme_id in os.listdir(base_dir):
        if selected_acme_ids is not None and acme_id not in selected_acme_ids:
            continue
        exp_dir = os.path.join(base_dir, acme_id)
        if not os.path.isdir(exp_dir):
            continue
        csv_path = os.path.join(exp_dir, 'logs', process_name, 'logs.csv')
        id_to_csv[acme_id] = csv_path

    return id_to_csv


def plot_all_learning_curves(id_to_csv, step_to_frame_multiplier=1):
    """Given a dict of {acme_id: csv_path}, plot all learning curves on one figure

    Args:
        id_to_csv: a dict of {acme_id: csv_path}
    """
    for acme_id, csv_path in id_to_csv.items():
        print(f"Plotting {acme_id}")
        df = pd.read_csv(csv_path)
        try:
            steps = df['actor_steps']
        except:
            print('failed for id', acme_id)
            continue
        frames = step_to_frame_multiplier * steps
        returns = df['episode_return']
        
        # sparsify data for plotting
        every_n = 1
        frames = frames[frames.index % every_n == 0]
        returns = returns[returns.index % every_n == 0]

        # Convert frames and returns to numbers
        converted_frames = []
        converted_returns = []
        for frame, ret in zip(frames, returns):
            try:
                converted_frames.append(int(frame))
                converted_returns.append(float(ret))
            except:
                continue

        print('Unique pre-converted returns: ', np.unique(converted_returns))
        
        # Apply moving average filter
        converted_returns = pd.Series(converted_returns).rolling(1000).mean()

        # Print unique values in converted_returns
        print('Unique converted returns: ', np.unique(converted_returns))
        
        # Create a line by interpolating
        x = converted_frames
        y = converted_returns
        f = interp1d(x, y)
        xnew = np.linspace(x[0], x[-1], num=1000, endpoint=True)
        ynew = f(xnew)

        # Plot
        plt.plot(xnew, ynew, label=acme_id)
        plt.xlabel('Frames')
        plt.ylabel('Returns')

    plt.show()
    save_path = os.path.expanduser(
        f"{os.path.join(args.base_dir, args.acme_id, 'learning_curves.png')}"
    )
    print(f"Saving to {save_path}")
    plt.savefig(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--acme_id', type=str, default=None, help='if None, use all acme_ids, otherwise use the selected ones')
    parser.add_argument('--base_dir', type=str, default=None, help='acme results dir, if None use the default acme setting')
    parser.add_argument('--process_name', type=str, default='evaluator', help='actor or evaluator')
    args = parser.parse_args()

    if args.acme_id is None:
        selected_acme_ids = None
    else:
        selected_acme_ids = args.acme_id.split(',')
    
    id_to_csv = gather_csv_files_from_base_dir(base_dir=args.base_dir, selected_acme_ids=selected_acme_ids, process_name=args.process_name)
    print(id_to_csv)
    plot_all_learning_curves(id_to_csv)