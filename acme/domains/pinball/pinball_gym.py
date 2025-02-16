import cv2
import gym
import dm_env
import imageio
from gym import spaces
from gym.error import DependencyNotInstalled

import numpy as np

from acme import wrappers
from acme.wrappers.gymnasium_wrapper import GymnasiumWrapper
from acme.domains.pinball.pinball import PinballModel, PinballView, BallModel
from acme.wrappers.minigrid_wrapper import MiniGridWrapper as PinballAcmeWrapper
from acme.wrappers.oar_goal import ObservationActionRewardGoalWrapper

from collections import UserDict, deque
from matplotlib.path import Path

import pygame
import random


def environment_builder(
    config_filename,
    seed=42,
    obs_type="pixel",
    goal_conditioned=True,
    scale_dims=(84, 84),
    sparse_reward=True,
    episode_length=1000,
):
    del seed
    assert obs_type in ["state", "pixel"], f"Invalid observation type {obs_type}"
    env = PinballEnv(config_filename, render_mode="rgb_array")

    if obs_type == "pixel":
        env = PinballPixelWrapper(env, n_frames=1)
    
    env = PinballInfoWrapper(env, sparse_reward=sparse_reward, seed=42)
    n_goal_dims = determine_n_goal_dims(env)
    task_goal_feats = determine_task_goal_features(env)
    env = GymnasiumWrapper(env)
    env = wrappers.StepLimitWrapper(env, episode_length)
    env = PinballAcmeWrapper(
        env,
        scale_dims=scale_dims,
        num_stacked_frames=1,
        action_repeats=1,
        flatten_frame_stack=True,
        grayscaling=False,
        pooled_frames=1,
        to_float=False,
        goal_conditioned=goal_conditioned,
        task_goal_features=task_goal_feats,
    )
    env = ObservationActionRewardGoalWrapper(env, info2goals, n_goal_dims)
    env = wrappers.SinglePrecisionWrapper(env)
    return env


class PinballInfoWrapper(gym.Wrapper):
    """
    A wrapper for the Pinball gym environment that augments the info dict.
    
    For each call to step() and reset(), extra information is added:
      - 'ball_position': The first two entries of the state (assumed [x, y]).
      - 'ball_velocity': The final two entries of the state (assumed [xdot, ydot]).
      - 'target_position': Taken from the underlying pinball model if set.
      - 'reached': A flag (here, simply set equal to done) to indicate if the goal was reached.
      - 'timestep': A counter of the number of steps taken.
    """
    def __init__(self, env, sparse_reward: bool = True, seed: int = 42):
        super(PinballInfoWrapper, self).__init__(env)
        self._seed = seed
        self._set_seed(seed)
        self._timestep = 0
        self._sparse_reward = sparse_reward

    def _set_seed(self, seed):
        self.env.seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action.item())
        info = self._modify_info_dict(info, terminated, truncated)
        self._timestep += 1
        reward = float(reward > 0) if self._sparse_reward else reward
        return state, reward, terminated or truncated, info

    def _modify_info_dict(self, info, terminated=False, truncated=False):
        ball_position = info['next_state'][:2].tolist()
        ball_velocity = info['next_state'][2:].tolist()
        
        info['player_pos'] = ball_position
        info['player_x'], info['player_y'] = ball_position
        info['player_velocity'] = ball_velocity
        
        info['target_position'] = self.env.env.pinball.target_pos

        info['truncated'] = truncated
        info['terminated'] = terminated
        info['timestep'] = self._timestep
        info['needs_reset'] = truncated  # pfrl needs this flag
        info['TimeLimit.truncated'] = truncated  # acme needs this flag
        info['reached'] = terminated

        return info

    def reset(self, **kwargs):
        self._timestep = 0
        state, info = self.env.reset(**kwargs)
        info = self._modify_info_dict(info)
        return state, info


def info2goals(info):
    pos = info['player_pos']
    discretized = np.array(pos, dtype=np.float32).round(decimals=2)
    integer_version = (discretized * 100).astype(np.int16)
    return integer_version

def determine_n_goal_dims(env: PinballInfoWrapper):
    """Determines the dimensionality of the goal vector."""
    state, info = env.reset()
    return len(info2goals(info))

def determine_task_goal_features(env: PinballInfoWrapper):
    """
    Returns a vector of “task goal” features.
    
    For Pinball, you might simply use the target position.
    """
    goal_pos = np.asarray([0.5, 0.06], dtype=np.float32).round(decimals=2)
    return (goal_pos * 100).astype(np.int16)


class GoalPinballModel(PinballModel):
    def set_target_pos(self, target_pos):
        self.target_pos = target_pos

    def set_initial_pos(self, start_pos):
        self._ball_rad = self.ball.radius
        self.ball = BallModel(start_position=start_pos, radius=self._ball_rad)
    
    def set_initial_state(self, state):
        self.ball.position[0], self.ball.position[1] = state[:2]
        self.ball.xdot, self.ball.ydot = state[2:]

class PinballEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    
    def __init__(self, config, start_pos=None, target_pos=None, width=500, height=500, render_mode=None):
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=np.zeros(4), high=np.ones(4))
        self.gamma = 1
        self.configuration = config
        self.width = width
        self.height = height
        self.render_mode = render_mode
        self.screen = None  
        self.clock = None

        self.pinball = GoalPinballModel(self.configuration)
        self.state = self.pinball.get_state()
        self._obstacles = [Path(obstacle.points) for obstacle in self.pinball.obstacles]
        if start_pos:
            self.pinball.set_initial_pos(start_pos)
        if target_pos:
            self.pinball.set_target_pos(target_pos)


    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid, action_space {self.action_space}"

        reward = self.pinball.take_action(action)
        next_state = self.pinball.get_state()
        done = self.pinball.episode_ended()

        if self.render_mode == "human":
            self.render()

        self.state = next_state

        return np.array(next_state), reward, done, False, {}

    def get_obstacles(self):
        """
            return list of Pinball Obstacles
        """
        return self.pinball.obstacles

    
    def sample_initial_positions(self, N):
        """
            Samples initial positions uniformly with velocity 0.
        """
        all_points = []
        total_points = 0
        vels = np.zeros((N, 2))
        while total_points < N:
            _p = np.random.uniform(low=0.02, high=0.98, size=(N, 2))
            points = self._get_points_outside_obstacles(_p)
            all_points.append(points)
            total_points += points.shape[0]
        pos = np.vstack(all_points)[:N]
        return np.hstack([pos, vels])

    def sample_init_states(self, N):
        """
            Samples states uniformly 
        """
        all_points = []
        total_points = 0
        vels = np.random.uniform(size=(N, 2))
        while total_points < N:
            
            points = self._get_points_outside_obstacles(_p)
            all_points.append(points)
            total_points += points.shape[0]
        return np.hstack([np.vstack(all_points)[:N], vels])

    def _get_points_outside_obstacles(self, points):
        points_mask = np.ones(points.shape[0], dtype=np.bool8)
        for obstacle in self._obstacles:
            in_obstacle = obstacle.contains_points(points, radius=0.)
            points_mask = np.logical_and(np.logical_not(in_obstacle), points_mask)
        return points[points_mask]
    
    def is_valid_state(self, state):
        for obstacle in self._obstacles:
            if obstacle.contains_points([state[:2]], radius=0.):
                return False
        return True

    def reset(self, state=None):
        self.pinball = GoalPinballModel(self.configuration) 
        self.state = self.pinball.get_state()
        self.screen = None
        return np.array(self.state)

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )
        
        if self.screen is None:
            if self.render_mode == "human":
                pygame.init()
                pygame.display.set_caption('Pinball Domain')
                self.screen = pygame.display.set_mode([self.width, self.height])
                self.environment_view = PinballView(self.screen, self.pinball)      
            else:
                self.screen = pygame.Surface((self.width, self.height))
                self.environment_view = PinballView(self.screen, self.pinball)  

        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.render_mode == "human":
            self.environment_view.blit()
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            self.environment_view.blit()
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen))/255, axes=(1, 0, 2)
            )


class PinballPixelWrapper(gym.Env):
    def __init__(self, environment, n_frames=1):
        self.frames = deque(maxlen=n_frames)
        self.env = environment
        self.n_frames = n_frames

    def step(self, *args, **kwargs):
        ret = self.env.step(*args, **kwargs)
        frame = self.env.render()
        frame = (frame * 255).astype(np.uint8)  # Convert float images to uint8
        if len(self.frames) < self.n_frames:
            self._init_queue(frame)
        self.frames.append(frame)

        return self._queue2array(self.frames), *ret[1:-1], {"next_state": ret[0]}

    def _init_queue(self, frame):
        for i in range(self.n_frames):
            self.frames.append(np.zeros_like(frame))

    def _queue2array(self, frames):
        arr = np.array(frames, dtype=np.uint8)
        assert arr.shape[0] == self.n_frames, arr.shape
        return arr.squeeze(0) if arr.shape[0] == 1 else arr

    def reset(self, *args, **kwargs):
        self.frames.clear()
        state = self.env.reset(*args, **kwargs)
        frame = self.env.render()
        frame = (frame * 255).astype(np.uint8)  # Convert float images to uint8
        if len(self.frames) < self.n_frames:
            self._init_queue(frame)
        self.frames.append(frame)
        return self._queue2array(self.frames), {"next_state": state}

    def sample_initial_positions(self, N):
        return self.env.sample_initial_positions(N)
    
    @property
    def action_space(self):
        return self.env.action_space

    def get_obstacles(self):
        return self.env.get_obstacles()
    
    @property
    def observation_space(self):
        shape = (self.n_frames, 500, 500, 3) if self.n_frames > 1 else (500, 500, 3)
        return spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)


def rollout_to_gif(env: dm_env.Environment, steps: int = 100, gif_filename: str = "rollout.gif"):
    """
    Runs a rollout in the given Gym environment for `steps` timesteps,
    overlays the frame number on each rendered frame, 
    and saves them to a GIF (using imageio).
    """
    ts = env.reset()
    frames = []

    for i in range(steps):
        # frame = env.render()
        frame = ts.observation.observation

        if not np.issubdtype(frame.dtype, np.uint8):
            frame = np.array(frame * 255, dtype=np.uint8)

        # Convert from RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        pos = (ts.observation.goals[0], ts.observation.goals[1])
        
        cv2.putText(
            img=frame_bgr,
            text=f"{pos}",
            org=(5, 20),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.33,
            color=(255, 255, 255),  # White text in BGR
            thickness=1
        )

        # Convert back from BGR to RGB
        frame_annotated = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Collect frame in list
        frames.append(frame_annotated)

        # Sample a random action from the dm_env action spec
        action = random.randint(0, env.action_spec().num_values - 1)
        ts = env.step(action)

        if ts.last():
            print("Episode ended.")
            break

    imageio.mimsave(gif_filename, frames, )
    print(f"GIF saved to {gif_filename}")


if __name__ == "__main__":
    environment = environment_builder(
        'acme/domains/pinball/configs/pinball_hard_single.cfg', seed=42, obs_type="pixel")
    # environment.reset()
