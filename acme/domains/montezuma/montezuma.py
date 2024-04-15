import gym
import ipdb
import functools
import numpy as np
from dm_env import specs
from gym.core import Wrapper

from acme import wrappers
from acme.wrappers import montezuma_wrapper
from acme.wrappers.gymnasium_wrapper import GymnasiumWrapper
from acme.wrappers.oar_goal import ObservationActionRewardGoalWrapper


class MontezumaInfoWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._timestep = 0
        self.num_lives = None
        self.imaginary_ladder_locations = set()
        self.reset()
    
    def reset(self):
        s0 = self.env.reset()
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
    return len(info2goals(info))


def determine_task_goal_features(env):
    env.reset()
    info = env.get_current_info(info={})
    goal_vector = info2goals(info)
    task_goal = np.zeros_like(goal_vector)
    task_goal[-1] = 1
    return task_goal


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
            new_shape = (old_shape[0], old_shape[1], old_shape[2] * 2)
        print(f'Creating goal-conditioned wrapper with shape {new_shape}')
        pixel_spec = specs.Array(
            shape=new_shape, dtype=pixel_spec.dtype, name=pixel_spec.name)
        pixel_spec = self._frame_stacker.update_spec(pixel_spec)
        return pixel_spec
    

def environment_builder(
    seed=None,
    max_episode_steps=108_000,  # 4500
    sticky_actions=False,
    goal_conditioned=False,
    num_stacked_frames=1,
    flatten_frame_stack=True,
    grayscaling=False,
    scale_dims=(84, 84),
    to_float=True,
    oarg_wrapper=True,
):
    version = 'v0' if sticky_actions else 'v4'
    level_name = f'MontezumaRevengeNoFrameskip-{version}'
    env = gym.make(level_name, full_action_space=True)
    env = MontezumaInfoWrapper(env)
    n_goal_dims = determine_n_goal_dims(env)
    env = GymnasiumWrapper(env)
    atari_wrapper = UVFAObsSpecWrapper if goal_conditioned else montezuma_wrapper.AtariWrapper
    env = atari_wrapper(
            env,
            scale_dims=scale_dims,
            to_float=to_float,
            max_episode_len=max_episode_steps,
            num_stacked_frames=num_stacked_frames,
            flatten_frame_stack=flatten_frame_stack,
            grayscaling=grayscaling,
            max_abs_reward=1.0)  # TODO(ab): reward clipping
    
    if oarg_wrapper:
        env = ObservationActionRewardGoalWrapper(env, info2goals=info2goals, n_goal_dims=n_goal_dims)
    env = wrappers.SinglePrecisionWrapper(env)
    return env
