"""Unit tests for the AMDP class in amdp.py."""

import time
import numpy as np
import matplotlib.pyplot as plt

from acme.agents.jax.r2d2 import amdp


def make_amdp_planner(transition_matrix,
                      hash2idx,
                      target_node,
                      reward_dict=None,
                      verbose=False,
                      max_vi_iterations=10,
                      amdp_reward_factor=200.0,
                      use_sparse_matrix=False):
    def make_fake_reward_dict(hash2idx):
        return {k: 0. for k in hash2idx}


    def make_fake_discount_dict(hash2idx):
        return {k: 1. for k in hash2idx}

    reward_dict = reward_dict if reward_dict else \
        make_fake_reward_dict(hash2idx)
    
    return amdp.AMDP(
        transition_tensor=transition_matrix,
        hash2idx=hash2idx,
        reward_dict=reward_dict,
        discount_dict=make_fake_discount_dict(hash2idx),
        count_dict={},
        target_node=target_node,
        verbose=verbose,
        max_vi_iterations=max_vi_iterations,
        rmax_factor=amdp_reward_factor,
        use_sparse_matrix=use_sparse_matrix)


def policy_equals(policy1, policy2, exclude_keys=()):
    for k in policy1:
        if k not in exclude_keys and policy1[k] != policy2[k]:
            return False
    return True


def generate_linear_graph(size):
    """Generates a simple linear graph of a given size."""
    transition_matrix = np.zeros((size, size), dtype=np.float32)
    for i in range(size - 1):
        transition_matrix[i, i + 1] = 1  # Direct path to the next node
    return transition_matrix


def test_linear_graph():
    transition_matrix = generate_linear_graph(4)
    hash2idx = {0: 0, 1: 1, 2: 2, 3: 3}
    target_node = 3
    expected_policy = {0: 1, 1: 2, 2: 3, 3: 3}
    planner = make_amdp_planner(transition_matrix, hash2idx, target_node)
    computed_policy = planner.get_policy()
    assert policy_equals(computed_policy, expected_policy, (target_node,)), \
        f'Expected {expected_policy} but got {computed_policy}. Value: {planner.get_values()}'
    print('[+] linear_graph test passed.')


def test_branching_graph():
    transition_matrix = np.array([
        [0, 0.5, 0.5, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    hash2idx = {0: 0, 1: 1, 2: 2, 3: 3}
    expected_policy1 = {0: 1, 1: 3, 2: 3, 3: 3}
    expected_policy2 = {0: 2, 1: 3, 2: 3, 3: 3}
    target_node = 3
    planner = make_amdp_planner(transition_matrix, hash2idx, target_node)
    computed_policy = planner.get_policy()
    assert policy_equals(computed_policy, expected_policy1, (target_node,)) or \
        policy_equals(computed_policy, expected_policy2, (target_node,)), \
        f'Expected {expected_policy1} or {expected_policy2} but got {computed_policy}.'
    print('[+] branching_graph test passed.')


def test_cyclic_graph():
    transition_matrix = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0.75, 0, 0, 0.5],
        [0, 0, 0, 0]
    ])
    target_node = 3
    hash2idx = {0: 0, 1: 1, 2: 2, 3: 3}
    expected_policy = {0: 1, 1: 2, 2: 3, 3: 3}
    planner = make_amdp_planner(transition_matrix, hash2idx, target_node)
    computed_policy = planner.get_policy()
    assert policy_equals(computed_policy, expected_policy, (target_node,)), \
        f'Expected {expected_policy} but got {computed_policy}.'
    print('[+] cyclic_graph test passed.')


def test_probabilistic_simple_graph1():
    transition_matrix = np.array([
        [0, 0.7, 0.3, 0],
        [0, 0, 0.8, 0.2],
        [0.4, 0.6, 0, 0],
        [0, 0, 0, 0]
    ])
    target_node = 3
    hash2idx = {0: 0, 1: 1, 2: 2, 3: 3}
    expected_policy = {0: 1, 1: 3, 2: 1, 3: 3}
    planner = make_amdp_planner(transition_matrix, hash2idx, target_node)
    computed_policy = planner.get_policy()
    assert policy_equals(computed_policy, expected_policy, (target_node,)), \
        f'Expected {expected_policy} but got {computed_policy}.'
    print('[+] probabilistic_simple_graph test passed.')


def test_probabilistic_simple_graph2():
    transition_matrix = np.array([
        [0, 0.7, 0.3, 0],
        [0, 0, 0.8, 0.2],
        [0.4, 0.27, 0, 0],
        [0, 0, 0, 0]
    ])
    target_node = 3
    hash2idx = {0: 0, 1: 1, 2: 2, 3: 3}
    expected_policy = {0: 1, 1: 3, 2: 0, 3: 3}
    planner = make_amdp_planner(transition_matrix, hash2idx, target_node)
    computed_policy = planner.get_policy()
    assert policy_equals(computed_policy, expected_policy, (target_node,)), \
        f'Expected {expected_policy} but got {computed_policy}.'
    print('[+] probabilistic_simple_graph2 test passed.')


def test_graph_with_dead_ends():
    transition_matrix = np.array([
        [0, 0.6, 0.4, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],  # Dead-end
        [0, 0, 0.5, 0, 0.5],
        [0, 0, 0, 0, 0]
    ])
    hash2idx = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    target_node = 4
    expected_policy = {0: 1, 1: 3, 2: 2, 3: 4, 4: 4}
    planner = make_amdp_planner(transition_matrix, hash2idx, target_node)
    computed_policy = planner.get_policy()
    assert policy_equals(computed_policy, expected_policy, (target_node, 2)), \
        f'Expected {expected_policy} but got {computed_policy}.'
    print('[+] graph_with_dead_ends test passed.')


def negative_value_probability():
    print('Starting negative_value_probability test.')
    transition_matrix = np.array([
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1]
    ])
    hash2idx = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    target_node = 4
    # reward_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: -10}
    reward_dict = None
    planner = make_amdp_planner(
        transition_matrix,
        hash2idx,
        target_node,
        amdp_reward_factor=-1.)
    computed_policy = planner.get_policy()
    values = planner.get_values()
    value_vector = np.array([values[i] for i in range(len(values))])
    reward_vector = planner._reward_vector
    discount_vector = planner._discount_vector
    q_matrix = transition_matrix @ (reward_vector[:, None] + discount_vector * value_vector)
    print(f'reward_vec: {planner._reward_vector}')
    print(f'discount_vec: {planner._discount_vector}')
    print(f'Q-Matrix: {q_matrix}')
    print(f'[+] negative_value_probability policy: {computed_policy}.')
    print(f'[+] negative_value_probability values: {values}.')


def get_plans_at_different_horizons(max_horizon, max_vi_iterations=10):
    results = []

    for horizon in range(2, max_horizon + 1):
        transition_matrix = generate_linear_graph(horizon)
        hash2idx = {i: i for i in range(horizon)}  # Mapping each node to itself

        # Initialize the Planner
        planner = make_amdp_planner(
            transition_matrix, hash2idx, horizon - 1,
            verbose=False, max_vi_iterations=max_vi_iterations)

        # Measure the time taken to compute the plan
        start_time = time.time()
        policy = planner.get_policy()
        end_time = time.time()

        # Expected policy for a linear graph
        expected_policy = {i: i + 1 for i in range(horizon - 1)}

        # Check if the policy is correct and record the time taken
        is_correct = policy_equals(policy, expected_policy, (horizon - 1,))
        time_taken = end_time - start_time
        results.append((horizon, is_correct, time_taken))

    return results


def test_planning_at_different_horizons():
    max_horizon = 20
    results = get_plans_at_different_horizons(max_horizon, max_vi_iterations=10)
    horizons = [result[0] for result in results]
    correctness = [result[1] for result in results]
    timing = [result[2] for result in results]

    results2 = get_plans_at_different_horizons(max_horizon, max_vi_iterations=20)
    horizons2 = [result[0] for result in results2]
    correctness2 = [result[1] for result in results2]
    timing2 = [result[2] for result in results2]

    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot for correctness
    axs[0].plot(horizons, correctness, marker='o', linestyle='-', label='VI Iterations = 10')
    axs[0].plot(horizons2, correctness2, marker='o', linestyle='-', label='VI Iterations = 20')
    axs[0].set_title('Planner Correctness Across Different Horizons')
    axs[0].set_xlabel('Horizon')
    axs[0].set_ylabel('Correctness (1 = Correct, 0 = Incorrect)')
    axs[0].set_ylim(-0.1, 1.1)  # To clearly show binary values
    axs[0].legend()

    # Plot for timing
    axs[1].plot(horizons, timing, marker='o', linestyle='-', label='VI Iterations = 10')
    axs[1].plot(horizons2, timing2, marker='o', linestyle='-', label='VI Iterations = 20')
    axs[1].set_title('Time Taken by Planner Across Different Horizons')
    axs[1].set_xlabel('Horizon')
    axs[1].set_ylabel('Time (seconds)')
    axs[1].legend()

    # Display the plot
    plt.tight_layout()
    plt.savefig('planning_at_different_horizons_rmax200.png')
    plt.close()


def test_low_probability_graph(size=4, transition_probability=1e-10):
    """
    Generates a simple linear graph of a given size with low transition probabilities.

    :param size: Number of states in the graph.
    :param transition_probability: The low probability for transition between states.
    :return: Transition matrix representing the graph.
    """
    transition_matrix = np.zeros((size, size))
    for i in range(size - 1):
        transition_matrix[i, i + 1] = transition_probability
    
    hash2idx = {i: i for i in range(size)}
    target_node = size - 1
    planner = make_amdp_planner(transition_matrix, hash2idx, target_node,
                                amdp_reward_factor=1e10, max_vi_iterations=100)
    computed_policy = planner.get_policy()
    print('[Underflow test] value function: ', planner.get_values())
    print('[Underflow test] computed_policy: ', computed_policy)


def test_large_sparse_graph(n_nodes=4000):
    transition_matrix = np.zeros((n_nodes, n_nodes))
    # Sparse transitions
    for i in range(n_nodes - 1):
        transition_matrix[i, i + 1] = 1

    hash2idx = {i: i for i in range(n_nodes)}
    target_node = n_nodes - 1
    planner = make_amdp_planner(
        transition_matrix, hash2idx, target_node,
        max_vi_iterations=n_nodes, use_sparse_matrix=True)
    start_time = time.time()
    computed_policy = planner.get_policy()
    elapsed_time = time.time() - start_time

    expected_policy = {i: i + 1 for i in range(n_nodes - 1)}
    assert policy_equals(computed_policy, expected_policy, (target_node,)), \
        f'Expected {expected_policy} but got {computed_policy}.'

    print(f'[+] Large Sparse Graph Test: Elapsed time = {elapsed_time} seconds')


def performance_comparison(size):

    transition_matrix = np.zeros((size, size))
    for i in range(size - 1):
        transition_matrix[i, i + 1] = 1
    
    hash2idx = {i: i for i in range(size)}
    target_node = size - 1
    
    # Dense
    start_time_dense = time.time()
    make_amdp_planner(transition_matrix, hash2idx, target_node,
        max_vi_iterations=size, use_sparse_matrix=False).get_policy()
    elapsed_time_dense = time.time() - start_time_dense
    
    # Sparse
    start_time_sparse = time.time()
    make_amdp_planner(transition_matrix, hash2idx, target_node,
        max_vi_iterations=size, use_sparse_matrix=True).get_policy()
    elapsed_time_sparse = time.time() - start_time_sparse
    
    print(f'Dense matrix elapsed time: {elapsed_time_dense} seconds')
    print(f'Sparse matrix elapsed time: {elapsed_time_sparse} seconds')


if __name__ == '__main__':
    test_linear_graph()
    test_branching_graph()
    test_cyclic_graph()
    test_probabilistic_simple_graph1()
    test_probabilistic_simple_graph2()
    test_graph_with_dead_ends()
    test_planning_at_different_horizons()
    negative_value_probability()
    test_low_probability_graph(size=40, transition_probability=0.1)
    test_large_sparse_graph(n_nodes=4000)
    performance_comparison(size=4000)