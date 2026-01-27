import gym
import ipdb
import pickle
import functools
import numpy as np
from dm_env import specs
from gym.core import Wrapper

from acme import wrappers
from acme.wrappers import montezuma_wrapper
from acme.wrappers.gymnasium_wrapper import GymnasiumWrapper
from acme.wrappers.oar_goal import ObservationActionRewardGoalWrapper


class MontezumaInfoWrapper(Wrapper):
    def __init__(self, env, reset_to_laser_room: bool = False):
        super().__init__(env)
        self._timestep = 0
        self.num_lives = None
        self.imaginary_ladder_locations = set()
        self.reset_to_laser_room = reset_to_laser_room
        self.reset()
    
    def reset(self):
        s0 = self.env.reset()
        if self.reset_to_laser_room:
            s0, info0 = self._reset_player_to_laser_room()
        self.num_lives = self.get_num_lives(self.get_current_ram())
        info = self.get_current_info(info={})
        return s0, info

    def step(self, action):
        self._timestep += 1
        obs, reward, done, info = self.env.step(action[0].item())
        info = self.get_current_info(info=info)
        self.num_lives = info["lives"]
        return obs, reward, done, info

    def get_current_info(self, info, update_lives=False):
        ram = self.get_current_ram()
    
        info["lives"] = self.get_num_lives(ram)
        info["player_x"] = self.get_player_x(ram)
        info["player_y"] = self.get_player_y(ram)
        info["room_number"] = self.get_room_number(ram)
        info["jumping"] = self.get_is_jumping(ram)
        info["dead"] = self.get_is_player_dead(ram)
        info["falling"] = self.get_is_falling(ram)
        info["uncontrollable"] = self.get_is_in_non_controllable_state(ram)
        # info["buggy_state"] = self.get_is_climbing_imaginary_ladder(ram)
        info["left_door_open"] = self.get_is_left_door_unlocked(ram)
        info["right_door_open"] = self.get_is_right_door_unlocked(ram)
        info["inventory"] = self.get_player_inventory(ram)
        info["TimeLimit.truncated"] = info.get('TimeLimit.truncated', False)
        info['truncated'] = info['TimeLimit.truncated']
        info['terminated'] = self.get_current_ale().game_over()

        if update_lives:
            self.num_lives = info["lives"]

        return info

    def get_current_position(self):
        ram = self.get_current_ram()
        return self.get_player_x(ram), self.get_player_y(ram)

    def get_player_x(self, ram):
        return int(self.getByte(ram, 'aa'))

    def get_player_y(self, ram):
        return int(self.getByte(ram, 'ab'))

    def get_num_lives(self, ram):
        return int(self.getByte(ram, 'ba'))

    def get_player_inventory(self, ram):
        # 'torch', 'sword', 'sword', 'key', 'key', 'key', 'key', 'hammer'
        return format(self.getByte(ram, 'c1'), '08b')
    
    def get_is_falling(self, ram):
        return int(int(self.getByte(ram, 'd8')) != 0)

    def get_is_jumping(self, ram):
        return int(self.getByte(ram, 'd6') != 0xFF)

    def get_room_number(self, ram):
        return int(self.getByte(ram, '83'))

    def get_current_ale(self):
        return self.unwrapped.ale

    def get_current_ram(self):
        return self.get_current_ale().getRAM()

    @staticmethod
    def _getIndex(address):
        assert type(address) == str and len(address) == 2 
        row, col = tuple(address)
        row = int(row, 16) - 8
        col = int(col, 16)
        return row*16+col

    @staticmethod
    def getByte(ram, address):
        # Return the byte at the specified emulator RAM location
        idx = MontezumaInfoWrapper._getIndex(address)
        return ram[idx]

    def get_player_status(self, ram):
        status = self.getByte(ram, '9e')
        status_codes = {
            0x00: 'standing',
            0x2A: 'running',
            0x3E: 'on-ladder',
            0x52: 'climbing-ladder',
            0x7B: 'on-rope',
            0x90: 'climbing-rope',
            0xA5: 'mid-air',
            0xBA: 'dead',  # dive 1
            0xC9: 'dead',  # dive 2
            0xC8: 'dead',  # dissolve 1
            0xDD: 'dead',  # dissolve 2
            0xFD: 'dead',  # smoke 1
            0xE7: 'dead',  # smoke 2
        }
        return status_codes[status]

    def get_is_player_dead(self, ram):
        player_status = self.get_player_status(ram)
        dead = player_status == "dead"
        time_to_spawn = self.getByte(ram, "b7")
        respawning = time_to_spawn > 0
        return dead or respawning

    def get_is_in_non_controllable_state(self, ram):
        player_status = self.get_player_status(ram)
        return self.get_is_jumping(ram) or \
            player_status in ("mid-air") or\
            self.get_is_falling(ram) or \
            self.get_is_player_dead(ram)

    def get_is_left_door_unlocked(self, ram):
        objects = format(self.getByte(ram, 'c2'), '08b')[-4:]
        left_door = objects[0]
        locked = int(left_door) == 1 and self.get_room_number(ram) in [1, 5, 17]
        return not locked

    def get_is_right_door_unlocked(self, ram):
        objects = format(self.getByte(ram, 'c2'), '08b')[-4:]
        right_door = objects[1]
        locked = int(right_door) == 1 and self.get_room_number(ram) in [1, 5, 17]
        return not locked

    def _reset_player_to_laser_room(self):
        ram_location = "ale_state_in_room_0.pkl"
        with open(ram_location, 'rb') as f:
            state = pickle.load(f)
        self.env.restore_state(state)
        new_obs, reward, done, new_info = self.env.step(0)
        return new_obs, new_info

    def binary2info(self, binary_vector, sparse_info: bool = False):
        """
        Convert a binary vector back into an info dictionary for Montezuma's Revenge.

        Args:
            binary_vector (np.ndarray): Binary vector representation of the info dict.
            sparse_info (bool): If True, omits fields not present in the binary vector.

        Returns:
            dict: Reconstructed info dictionary.
        """
        # Define the inventory mapping
        inventory_items = ['torch', 'sword', 'sword', 'key', 'key', 'key', 'key', 'hammer']

        # Initialize the info dictionary
        info = {}

        # Decode player_x (first 206 indices)
        player_x_index = np.where(binary_vector[:206] == 1)[0]
        if len(player_x_index) > 0 or not sparse_info:
            info["player_x"] = player_x_index[0] if len(player_x_index) > 0 else -1

        # Decode player_y (indices 206 to 341, normalized to 120-255)
        player_y_index = np.where(binary_vector[206:342] == 1)[0]
        if len(player_y_index) > 0 or not sparse_info:
            info["player_y"] = player_y_index[0] + 120 if len(player_y_index) > 0 else -1

        # Decode room number (indices 342 to 373)
        room_number_index = np.where(binary_vector[342:374] == 1)[0]
        if len(room_number_index) > 0 or not sparse_info:
            info["room_number"] = room_number_index[0] if len(room_number_index) > 0 else -1

        # Decode player state flags
        if not sparse_info or binary_vector[374]:
            info["jumping"] = bool(binary_vector[374])
        if not sparse_info or binary_vector[375]:
            info["dead"] = bool(binary_vector[375])
        if not sparse_info or binary_vector[376]:
            info["falling"] = bool(binary_vector[376])
        if not sparse_info or binary_vector[377]:
            info["uncontrollable"] = bool(binary_vector[377])

        # Decode door states
        if not sparse_info or binary_vector[378]:
            info["left_door_open"] = bool(binary_vector[378])
        if not sparse_info or binary_vector[379]:
            info["right_door_open"] = bool(binary_vector[379])

        # Decode inventory (binary string starting from index 380)
        inventory_binary = binary_vector[380:380 + len(inventory_items)]
        decoded_inventory = [
            item for bit, item in zip(inventory_binary, inventory_items) if bit
        ]
        if len(decoded_inventory) > 0 or not sparse_info:
            info["inventory"] = decoded_inventory

        # Decode task goal flag (last bit)
        if not sparse_info or binary_vector[-1]:
            info["task_goal"] = bool(binary_vector[-1])

        return info


class TransposeObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        shape = self.observation_space.shape
        new_shape = (shape[2], shape[0], shape[1])
        self._observation_space = gym.spaces.Box(
            low=0, high=255, shape=new_shape, dtype=np.uint8)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)

def info2goals(info):
    """Serialize the info dict into a np vector of integers."""
    # TODO(ab): don't put jumping, dead and falling in goal-space but access it using the info.
    goals = [
        info["player_x"],
        info["player_y"],
        info["room_number"],
        info["jumping"],
        info["dead"],
        info["falling"],
        info["uncontrollable"],
        info["left_door_open"],
        info["right_door_open"],
    ]
    for char in info["inventory"]:
        goals.append(int(char))
    goals.append(False)  # last bit denotes task reward function
    return np.asarray(goals, dtype=np.int16)


def goals2info(goals: np.ndarray):
    """Deserialize the info dict from a np vector of integers."""
    return {
        "player_x": goals[0],
        "player_y": goals[1],
        "room_number": goals[2],
        "jumping": bool(goals[3]),
        "dead": bool(goals[4]),
        "falling": bool(goals[5]),
        "uncontrollable": bool(goals[6]),
        "left_door_open": bool(goals[7]),
        "right_door_open": bool(goals[8]),
        "inventory": ''.join(str(int(x)) for x in goals[9:-1]),
        "task_goal": goals[-1]
    }


def determine_n_goal_dims(env):
    _, info = env.reset()
    return len(info2binary(info))


def determine_task_goal_features(env):
    env.reset()
    info = env.get_current_info(info={})
    goal_vector = info2binary(info)
    task_goal = np.zeros_like(goal_vector)
    task_goal[-1] = 1
    return task_goal


def info2binary(info):
    """
    Convert the info dict from Montezuma's Revenge into a binary vector.

    Args:
        info (dict): Dictionary containing environment info.

    Returns:
        np.ndarray: Binary vector representation of the info dict.
    """
    # Determine vector size based on necessary fields
    binary_vector_size = 128 + (256 - 120) + 206  # Adjust as necessary
    binary_vector = np.zeros(binary_vector_size, dtype=bool)

    # Encode player_x (0 to 205)
    binary_vector[info["player_x"]] = 1  # Direct mapping for x-coordinates

    # Encode player_y (120 to 255, normalized to 0-135 for vector index)
    binary_vector[206 + (info["player_y"] - 120)] = 1

    # Encode room number (assuming a maximum of 32 rooms for simplicity)
    binary_vector[342 + info["room_number"]] = 1

    # Encode player state flags
    binary_vector[374] = info["jumping"]
    binary_vector[375] = info["dead"]
    binary_vector[376] = info["falling"]
    binary_vector[377] = info["uncontrollable"]

    # Encode door states
    binary_vector[378] = info["left_door_open"]
    binary_vector[379] = info["right_door_open"]

    # Encode inventory as a binary string
    for i, item in enumerate(info["inventory"]):
        binary_vector[380 + i] = int(item)

    # Encode task goal flag
    binary_vector[-1] = info.get("task_goal", False)

    return binary_vector


class UVFAObsSpecWrapper(montezuma_wrapper.AtariWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_goal_features = determine_task_goal_features(self._environment)

    def _init_observation_spec(self):
        pixel_spec = super()._init_observation_spec()
        old_shape = pixel_spec.shape
        if self._grayscaling:
            new_shape = (old_shape[0], old_shape[1], 2)
        else:
            new_shape = (old_shape[0], old_shape[1], old_shape[2] + 1)
        print(f'Creating goal-conditioned wrapper with shape {new_shape}')
        pixel_spec = specs.Array(
            shape=new_shape, dtype=pixel_spec.dtype, name=pixel_spec.name)
        pixel_spec = self._frame_stacker.update_spec(pixel_spec)
        return pixel_spec

    def step(self, action):
        ts = super().step(action)
        if self._grayscaling:
            ts = ts._replace(observation=ts.observation[..., np.newaxis])
        return ts

    def reset(self):
        ts = super().reset()
        if self._grayscaling:
            ts = ts._replace(observation=ts.observation[..., np.newaxis])
        return ts
    

def environment_builder(
    seed=None,
    max_episode_steps=108_000,  # 4500
    sticky_actions=False,
    goal_conditioned=False,
    num_stacked_frames=1,
    flatten_frame_stack=True,
    grayscaling=True,
    scale_dims=(84, 84),
    to_float=True,
    oarg_wrapper=True,
    action_repeat=4,
    reset_to_laser_room=False,
    use_learned_goal_classifiers=True,
    classifier_trigger_dir=None,
):
    version = 'v0' if sticky_actions else 'v4'
    level_name = f'MontezumaRevengeNoFrameskip-{version}'
    env = gym.make(level_name, full_action_space=True)
    env = MontezumaInfoWrapper(env, reset_to_laser_room=reset_to_laser_room)
    n_goal_dims = determine_n_goal_dims(env)
    env = GymnasiumWrapper(env)
    
    if goal_conditioned:
        env = UVFAObsSpecWrapper(
            env,
            scale_dims=scale_dims,
            to_float=to_float,
            max_episode_len=max_episode_steps,
            num_stacked_frames=num_stacked_frames,
            flatten_frame_stack=flatten_frame_stack,
            grayscaling=True,
            max_abs_reward=1.0,
            action_repeats=action_repeat,
            pooled_frames=1 if action_repeat == 1 else 2,
        )  # TODO(ab): reward clipping
    else:
        env = montezuma_wrapper.AtariWrapper(
            env,
            scale_dims=scale_dims,
            to_float=to_float,
            max_episode_len=max_episode_steps,
            num_stacked_frames=num_stacked_frames,
            flatten_frame_stack=flatten_frame_stack,
            grayscaling=True,
            max_abs_reward=1.0,
            action_repeats=action_repeat,
            pooled_frames=1 if action_repeat == 1 else 2,
        ) # TODO(ab): reward clipping
    
    if oarg_wrapper:
        env = ObservationActionRewardGoalWrapper(
            env,
            info2goals=info2binary,
            n_goal_dims=n_goal_dims,
            use_learned_goal_classifiers=use_learned_goal_classifiers,
            classifier_trigger_dir=classifier_trigger_dir,
        )
    env = wrappers.SinglePrecisionWrapper(env)
    return env