import atexit
import os
import copy
import random
import sys
import threading
import traceback

from pyrep.const import TextureMappingMode
from pyrep.const import RenderMode

import cloudpickle
from functools import partial
import gym
import numpy as np
from rlbench.utils import name_to_task_class

from rlbench import RandomizeEvery
from rlbench import VisualRandomizationConfig

import time

try:
    from pyrep.errors import ConfigurationPathError, IKError
    from rlbench.backend.exceptions import InvalidActionError
except:
    pass


class RLBench:
    def __init__(
        self,
        langs,
        name,
        camera_keys,
        size=(64, 64),
        actions_min_max=None,
        shaped_rewards=False,
        use_lang_embeddings=False,
        boundary_reward_penalty=False,
        randomize=False,
        finetune_lang_encoding=None,
    ):
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import (
            EndEffectorPoseViaPlanning,
        )
        from rlbench.action_modes.gripper_action_modes import (
            Discrete,
        )
        from rlbench.environment import Environment
        from rlbench.observation_config import ObservationConfig
        from rlbench.tasks import (
            PhoneOnBase,
            PickAndLift,
            PickUpCup,
            PutRubbishInBin,
            TakeLidOffSaucepan,
            TakeUmbrellaOutOfUmbrellaStand,
            MultiTaskMicrofridgesauce,
            # MultiTaskBusplanesauce,
            PickShapenetObjects
        )

        # we only support reach_target in this codebase
        obs_config = ObservationConfig()

        ## Camera setups
        obs_config.front_camera.set_all(False)
        obs_config.wrist_camera.set_all(False)
        obs_config.left_shoulder_camera.set_all(False)
        obs_config.right_shoulder_camera.set_all(False)
        obs_config.overhead_camera.set_all(False)

        if "image_front" in camera_keys:
            obs_config.front_camera.rgb = True
            obs_config.front_camera.image_size = size
            obs_config.front_camera.render_mode = RenderMode.OPENGL

        if "image_wrist" in camera_keys:
            obs_config.wrist_camera.rgb = True
            obs_config.wrist_camera.image_size = size
            obs_config.wrist_camera.render_mode = RenderMode.OPENGL

        if "image_overhead" in camera_keys:
            obs_config.overhead_camera.rgb = True
            obs_config.overhead_camera.image_size = size
            obs_config.overhead_camera.render_mode = RenderMode.OPENGL

        obs_config.joint_forces = False
        obs_config.joint_positions = True
        obs_config.joint_velocities = True
        obs_config.task_low_dim_state = True
        obs_config.gripper_touch_forces = False
        obs_config.gripper_pose = True
        obs_config.gripper_open = True
        obs_config.gripper_matrix = False
        obs_config.gripper_joint_positions = True

        if randomize:
            rand_config = [
                VisualRandomizationConfig(image_directory='common/assets/textures/table', whitelist = ['diningTable_visible']),
                VisualRandomizationConfig(image_directory='common/assets/textures/wall', whitelist = ['Wall1', 'Wall2', 'Wall3', 'Wall4']),
                VisualRandomizationConfig(image_directory='common/assets/textures/floor', whitelist = ['Floor'])
            ]
            tex_kwargs = [
                {'mapping_mode': TextureMappingMode.PLANE, 'repeat_along_u': False, 'repeat_along_v': False, 'uv_scaling': [1.6, 1.1]},
                {'mapping_mode': TextureMappingMode.PLANE, 'repeat_along_u': False, 'repeat_along_v': False, 'uv_scaling': [5.0, 3.0]},
                {'mapping_mode': TextureMappingMode.PLANE, 'repeat_along_u': False, 'repeat_along_v': False, 'uv_scaling': [5.0, 5.0]}
            ]
            randomized_every = RandomizeEvery.EPISODE
        else:
            rand_config = None
            randomized_every = None
            tex_kwargs = None

        env = Environment(
            action_mode=MoveArmThenGripper(
                arm_action_mode=EndEffectorPoseViaPlanning(False),
                gripper_action_mode=Discrete(),
            ),
            obs_config=obs_config,
            headless=True,
            shaped_rewards=shaped_rewards,
            randomize_every=randomized_every, 
            visual_randomization_config=rand_config,
            tex_kwargs=tex_kwargs
        )
        env.launch()

        if name == "phone_on_base":
            task = PhoneOnBase
        elif name == "pick_and_lift":
            task = PickAndLift
        elif name == "pick_up_cup":
            task = PickUpCup
        elif name == "put_rubbish_in_bin":
            task = PutRubbishInBin
        elif name == "take_lid_off_saucepan":
            task = TakeLidOffSaucepan
        elif name == "take_umbrella_out_of_umbrella_stand":
            task = TakeUmbrellaOutOfUmbrellaStand
        elif name == "multi_task_microfridgesauce":
            task = MultiTaskMicrofridgesauce
        elif name == "pick_shapenet_objects":
            task = PickShapenetObjects
        elif name in ["reach_for_bus", "reach_for_plane"]:
            task = MultiTaskBusplanesauce
        else:
            task = name_to_task_class(name)
        self._env = env
        self._task = env.get_task(task)
        self.task_name = name

        if "pick_shapenet_objects" in name:
            try:
                n_obj = int(name.split("_")[-1])
                self._task._task.set_num_objects(n_obj)
            except:
                self._task._task.set_num_objects(1)

        _, obs = self._task.reset()

        task_low_dim = obs.task_low_dim_state.shape[0]
        self._state_dim = obs.get_low_dim_data().shape[0] - 14 - task_low_dim
        self._prev_obs, self._prev_reward = None, None
        self._ep_success = None

        self._size = size
        self._shaped_rewards = shaped_rewards
        self._camera_keys = camera_keys
        self._use_lang_embeddings = use_lang_embeddings
        self.finetune_lang_encoding = finetune_lang_encoding
        self._boundary_reward_penalty = boundary_reward_penalty
        self.langs = langs

        if actions_min_max:
            self.register_min_max(actions_min_max)
        else:
            self.low = np.array([-0.03, -0.03, -0.03])
            self.high = np.array([0.03, 0.03, 0.03])

        
        self._name = name

    @property
    def container_pos(self):
        if "pick_shapenet_objects" in self._name:
            return self._task._task.large_container.get_position()
        return None

    @property
    def obs_space(self):
        spaces = {
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            "success": gym.spaces.Box(0, 1, (), dtype=bool),
            "state": gym.spaces.Box(
                -np.inf, np.inf, (self._state_dim,), dtype=np.float32
            ),
            "image": gym.spaces.Box(
                0,
                255,
                (self._size[0], self._size[1] * len(self._camera_keys), 3),
                dtype=np.uint8,
            ),
            "init_state": gym.spaces.Box(
                -np.inf, np.inf, (self._state_dim,), dtype=np.float32
            ),
            "init_image": gym.spaces.Box(
                0,
                255,
                (self._size[0], self._size[1] * len(self._camera_keys), 3),
                dtype=np.uint8,
            ),
            "lang_num": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.uint8),
        }
        if self._use_lang_embeddings:
            spaces["lang_embedding"] = gym.spaces.Box(
                -np.inf, np.inf, (768,), dtype=np.float32)
        return spaces

    def register_min_max(self, actions_min_max):
        self.low, self.high = actions_min_max

    @property
    def act_space(self):
        assert self.low is not None
        if self.low.shape[0] == 3:
            self.low = np.hstack([self.low, [0.0]])
            self.high = np.hstack([self.high, [1.0]])
        action = gym.spaces.Box(
            low=self.low, high=self.high, shape=(self.low.shape[0],), dtype=np.float32
        )
        return {"action": action}

    def unnormalize(self, a):
        # Un-normalize gripper pose normalized to [-1, 1]
        assert self.low is not None
        pose = a[:3]
        pose = (pose + 1) / 2 * (self.high[:3] - self.low[:3]) + self.low[:3]

        # Manual handling of overflow in z axis
        curr_pose = self._task._task.robot.arm.get_tip().get_pose()[:3]
        curr_z = curr_pose[2]
        init_z = self._init_pose[2]
        delta_z = pose[2]

        if curr_z + delta_z >= init_z:
            pose[2] = 0.0

        # Un-normalize gripper action normalized to [-1, 1]
        gripper = a[3:4]
        gripper = (gripper + 1) / 2 * (self.high[3:4] - self.low[3:4]) + self.low[3:4]

        target_pose = pose

        # Identity quaternion
        quat = np.array([0.0, 0.0, 0.0, 1.0])

        action = np.hstack([target_pose, quat, gripper])
        assert action.shape[0] == 8
        return action

    def step(self, action):
        assert np.isfinite(action["action"]).all(), action["action"]
        try:
            original_action = self.unnormalize(action["action"])
            _obs, _reward, _ = self._task.step(original_action)
            terminal = False
            success, _ = self._task._task.success()
            if success:
                self._ep_success = True
            self._prev_obs, self._prev_reward = _obs, _reward
            if not self._shaped_rewards:
                reward = float(self._ep_success)
            else:
                reward = _reward
        except ConfigurationPathError:
            print("ConfigurationPathError")
            _obs = self._prev_obs
            terminal = False
            success = False
            if not self._shaped_rewards:
                reward = float(self._ep_success)
            else:
                reward = self._prev_reward
        except (IKError, InvalidActionError) as e:
            # print(e)
            _obs = self._prev_obs
            success = False
            if self._boundary_reward_penalty:
                terminal = True
                reward = -0.05
            else:
                terminal = False
                if not self._shaped_rewards:
                    reward = float(self._ep_success)
                else:
                    reward = self._prev_reward

        _obs.joint_velocities = None
        _obs.joint_positions = None
        _obs.task_low_dim_state = None

        obs = {
            "reward": reward,
            "is_first": False,
            "is_last": terminal,
            "is_terminal": terminal,
            "success": success,
            "state": _obs.get_low_dim_data(), 
            'lang_num': 0
        }
        images = []
        for key in self._camera_keys:
            if key == "image_front":
                images.append(_obs.front_rgb)
            if key == "image_wrist":
                images.append(_obs.wrist_rgb)
            if key == "image_overhead":
                images.append(_obs.overhead_rgb)
        obs["image"] = np.concatenate(images, axis=-2)
        if self._use_lang_embeddings and self.finetune_lang_encoding is not None:
            obs['lang_embedding'] = self.finetune_lang_encoding
        self._time_step += 1
        return obs

    def reset(self):
        self.lang = random.choice(self.langs)
        self._task._task.change_reward(self.lang)
        _, _obs = self._task.reset()
        print(f"Reset in env {self.task_name}.")
        self._prev_obs = _obs
        self._init_pose = copy.deepcopy(
            self._task._task.robot.arm.get_tip().get_pose()[:3]
        )
        self._time_step = 0
        self._ep_success = False

        _obs.joint_velocities = None
        _obs.joint_positions = None
        _obs.task_low_dim_state = None

        obs = {
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "success": False,
            "state": _obs.get_low_dim_data(),
            'lang_num': 0
        }
        images = []
        for key in self._camera_keys:
            if key == "image_front":
                images.append(_obs.front_rgb)
            if key == "image_wrist":
                images.append(_obs.wrist_rgb)
            if key == "image_overhead":
                images.append(_obs.overhead_rgb)
        obs["image"] = np.concatenate(images, axis=-2)
        if self._use_lang_embeddings and self.finetune_lang_encoding is not None:
            obs['lang_embedding'] = self.finetune_lang_encoding
        return obs
    
    def change_lang_encoding(self, lang_encoding):
        self.finetune_lang_encoding = lang_encoding

class VidLangRLBench(RLBench):
    def __init__(self, name, langs, camera_keys, size=(64, 64), actions_min_max=None, 
                    shaped_rewards=False, lang_to_num=None, lang_to_encoding=None, 
                    use_lang_embeddings=False, boundary_reward_penalty=False, 
                    curriculum=None, lang_curriculum=None, synonym_dict=None,
                    randomize=False):
        self.lang_to_num = lang_to_num
        self.lang_to_encoding = lang_to_encoding
        self.langs = langs
        self.init_langs = copy.deepcopy(langs)
        self.episode_count = 0 # maybe not the most accurate, not sure
        if curriculum is not None:
            self.use_curriculum = curriculum.use 
            self.curriculum = curriculum
            n_episodes = [int(x) for x in curriculum.num_episodes.split("|")]
            # prefix sum
            self.curriculum_benchmarks = [0 for _ in range(len(n_episodes))]
            for i in range(len(n_episodes)):
                for j in range(i):
                    # technically the last_n_episodes doesn't matter
                    # env will keep running until config.steps
                    self.curriculum_benchmarks[i] += n_episodes[j]
            self.lang_prompts = lang_curriculum
            self.synonym_dict = synonym_dict
            
            self.objects = curriculum.objects.split("|")
            for i in range(len(self.objects)):
                self.objects[i] = self.objects[i].split(",")
            self.num_objects = curriculum.num_objects.split("|")
            self.num_objects = [int(x) for x in self.num_objects]
            self.num_unique_per_class = curriculum.num_unique_per_class.split("|")
            self.num_unique_per_class = [int(x) for x in self.num_unique_per_class]

        super(VidLangRLBench, self).__init__(langs, name, camera_keys, size, actions_min_max, shaped_rewards, use_lang_embeddings, boundary_reward_penalty, randomize)

    def reset_curriculum(self):
        # todo: this only works for the shapenet env for now.
        # find index of self.episode_count in the list self.curriculum_benchmarks, if it exists
        if self.episode_count in self.curriculum_benchmarks:
            index = self.curriculum_benchmarks.index(self.episode_count)
            self.init_langs, self.langs = self.lang_prompts[index].copy(), self.lang_prompts[index].copy()
            self._task._task.reset_samplers(self.objects[index], self.objects[index], self.num_unique_per_class[index], self.num_unique_per_class[index])
            self._task._task.set_num_objects(self.num_objects[index])
            print(f"[curriculum] lang prompts: {self.langs}")
            print(f"[curriculum] objects: {self.num_objects[index]} of {self.objects[index]}")

    def reset(self):
        if self.use_curriculum:
            self.reset_curriculum()
        if self.init_langs:
            self.lang = random.choice(self.init_langs)
            self.init_langs.remove(self.lang)
        else:
            self.lang = random.choice(self.langs)

        time_step = super(VidLangRLBench, self).reset()
        if "[NOUN]" in self.lang:
            curr_obj = self._task._task.bin_objects_meta[0]
            if isinstance(self.synonym_dict[curr_obj], list):
                synonym = random.choice(self.synonym_dict[curr_obj])
            else:
                synonym = self.synonym_dict[curr_obj]
            self.lang = self.lang.replace("[NOUN]", synonym)
        self._task._task.change_reward(self.lang)

        lang_num = self.lang_to_num[self.lang]
        print(f"Collecting for {self.lang} language instruction.")
        print(f"Reward is for {self._task._task.reward_lang}.")
        vidlang_time_step = time_step.copy()
        vidlang_time_step['init_image'] = vidlang_time_step['image']
        vidlang_time_step['init_state'] = vidlang_time_step['state']
        vidlang_time_step['lang_num'] = lang_num
        if self._use_lang_embeddings:
            vidlang_time_step['lang_embedding'] = self.lang_to_encoding[self.lang]
        self._init_vidlang_time_step = vidlang_time_step
        self.episode_count += 1
        return vidlang_time_step

    def step(self, action):
        time_step = super(VidLangRLBench, self).step(action)
        lang_num = self.lang_to_num[self.lang]
        vidlang_time_step = time_step.copy()
        vidlang_time_step['init_image'] = self._init_vidlang_time_step['init_image']
        vidlang_time_step['init_state'] = self._init_vidlang_time_step['init_state']
        vidlang_time_step['lang_num'] = lang_num
        if self._use_lang_embeddings:
            vidlang_time_step['lang_embedding'] = self.lang_to_encoding[self.lang]
        return vidlang_time_step

class MultiTaskRLBench(RLBench):
    def __init__(self, task_names, camera_keys, size=(64, 64), actions_min_max=None, shaped_rewards=False, lang_to_num=None, lang_to_encoding=None, use_lang_embeddings=False, boundary_reward_penalty=False):
        self.task_names = task_names
        self.init_tasks = self.task_names.copy()
        self.task_name = random.choice(self.init_langs)

        super(MultiTaskRLBench, self).__init__(self.task_name, camera_keys, size, actions_min_max, shaped_rewards, use_lang_embeddings, boundary_reward_penalty)
        self.name_to_class = {}
        for t in self.task_names:
            self.name_to_class[t] = name_to_task_class(t)
            
    def reset(self):
        if self.init_tasks:
            self.task_name = random.choice(self.init_tasks)
            self.init_langs.remove(self.task_name)
        else:
            self.task_name = random.choice(self.task_names)
        self._task = self._env.get_task(self.name_to_class[self.task_name])
        return super(MultiTaskRLBench, self).reset()

class MultiTaskVidLangRLBench(MultiTaskRLBench):
    def __init__(self, task_names, camera_keys, size=(64, 64), actions_min_max=None, shaped_rewards=False, lang_to_num=None, lang_to_encoding=None, use_lang_embeddings=False, boundary_reward_penalty=False):
        self.lang_to_num = lang_to_num
        self.lang_to_encoding = lang_to_encoding
        super(MultiTaskVidLangRLBench, self).__init__(task_names, camera_keys, size, actions_min_max, shaped_rewards, use_lang_embeddings, boundary_reward_penalty)

    def reset(self):
        time_step = super(MultiTaskVidLangRLBench, self).reset()
        lang_num = self.lang_to_num[self.task_name]
        vidlang_time_step = time_step.copy()
        vidlang_time_step['init_image'] = vidlang_time_step['image']
        vidlang_time_step['init_state'] = vidlang_time_step['state']
        vidlang_time_step['lang_num'] = lang_num
        self._init_vidlang_time_step = vidlang_time_step
        if self._use_lang_embeddings:
            vidlang_time_step['lang_embedding'] = self.lang_to_encoding[self.lang]
        return vidlang_time_step

    def step(self, action):
        time_step = super(MultiTaskVidLangRLBench, self).step(action)
        lang_num = self.lang_to_num[self.task_name]
        vidlang_time_step = time_step.copy()
        vidlang_time_step['init_image'] = self._init_vidlang_time_step['init_image']
        vidlang_time_step['init_state'] = self._init_vidlang_time_step['init_state']
        vidlang_time_step['lang_num'] = lang_num
        if self._use_lang_embeddings:
            vidlang_time_step['lang_embedding'] = self.lang_to_encoding[self.lang]
        return vidlang_time_step

class TimeLimit:
    def __init__(self, env, duration):
        self._env = env
        self._duration = duration
        self._step = None

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    def step(self, action):
        assert self._step is not None, "Must reset environment."
        obs = self._env.step(action)
        self._step += 1
        if self._duration and self._step >= self._duration:
            obs["is_last"] = True
            self._step = None
        return obs

    def reset(self):
        self._step = 0
        return self._env.reset()


class ResizeImage:
    def __init__(self, env, size=(64, 64)):
        self._env = env
        self._size = size
        self._keys = [
            k
            for k, v in env.obs_space.items()
            if len(v.shape) > 1 and v.shape[:2] != size
        ]
        print(f'Resizing keys {",".join(self._keys)} to {self._size}.')
        if self._keys:
            from PIL import Image

            self._Image = Image

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def obs_space(self):
        spaces = self._env.obs_space
        for key in self._keys:
            shape = self._size + spaces[key].shape[2:]
            spaces[key] = gym.spaces.Box(0, 255, shape, np.uint8)
        return spaces

    def step(self, action):
        obs = self._env.step(action)
        for key in self._keys:
            obs[key] = self._resize(obs[key])
        return obs

    def reset(self):
        obs = self._env.reset()
        for key in self._keys:
            obs[key] = self._resize(obs[key])
        return obs

    def _resize(self, image):
        image = self._Image.fromarray(image)
        image = image.resize(self._size, self._Image.NEAREST)
        image = np.array(image)
        return image


class RenderImage:
    def __init__(self, env, key="image"):
        self._env = env
        self._key = key
        self._shape = self._env.render().shape

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def obs_space(self):
        spaces = self._env.obs_space
        spaces[self._key] = gym.spaces.Box(0, 255, self._shape, np.uint8)
        return spaces

    def step(self, action):
        obs = self._env.step(action)
        obs[self._key] = self._env.render("rgb_array")
        return obs

    def reset(self):
        obs = self._env.reset()
        obs[self._key] = self._env.render("rgb_array")
        return obs


class Async:

    # Message types for communication via the pipe.
    _ACCESS = 1
    _CALL = 2
    _RESULT = 3
    _CLOSE = 4
    _EXCEPTION = 5

    def __init__(self, constructor, strategy="thread"):
        self._pickled_ctor = cloudpickle.dumps(constructor)
        if strategy == "process":
            import multiprocessing as mp

            context = mp.get_context("spawn")
        elif strategy == "thread":
            import multiprocessing.dummy as context
        else:
            raise NotImplementedError(strategy)
        self._strategy = strategy
        self._conn, conn = context.Pipe()
        self._process = context.Process(target=self._worker, args=(conn,))
        atexit.register(self.close)
        self._process.start()
        self._receive()  # Ready.
        self._obs_space = None
        self._act_space = None

    def access(self, name):
        self._conn.send((self._ACCESS, name))
        return self._receive

    def call(self, name, *args, **kwargs):
        payload = name, args, kwargs
        self._conn.send((self._CALL, payload))
        return self._receive

    def close(self):
        try:
            self._conn.send((self._CLOSE, None))
            self._conn.close()
        except IOError:
            pass  # The connection was already closed.
        self._process.join(5)

    @property
    def obs_space(self):
        if not self._obs_space:
            self._obs_space = self.access("obs_space")()
        return self._obs_space

    @property
    def act_space(self):
        if not self._act_space:
            self._act_space = self.access("act_space")()
        return self._act_space

    def step(self, action, blocking=False):
        promise = self.call("step", action)
        if blocking:
            return promise()
        else:
            return promise

    def reset(self, blocking=False):
        promise = self.call("reset")
        if blocking:
            return promise()
        else:
            return promise

    def _receive(self):
        try:
            message, payload = self._conn.recv()
        except (OSError, EOFError):
            raise RuntimeError("Lost connection to environment worker.")
        # Re-raise exceptions in the main process.
        if message == self._EXCEPTION:
            stacktrace = payload
            raise Exception(stacktrace)
        if message == self._RESULT:
            return payload
        raise KeyError("Received message of unexpected type {}".format(message))

    def _worker(self, conn):
        try:
            ctor = cloudpickle.loads(self._pickled_ctor)
            env = ctor()
            conn.send((self._RESULT, None))  # Ready.
            while True:
                try:
                    # Only block for short times to have keyboard exceptions be raised.
                    if not conn.poll(0.1):
                        continue
                    message, payload = conn.recv()
                except (EOFError, KeyboardInterrupt):
                    break
                if message == self._ACCESS:
                    name = payload
                    result = getattr(env, name)
                    conn.send((self._RESULT, result))
                    continue
                if message == self._CALL:
                    name, args, kwargs = payload
                    result = getattr(env, name)(*args, **kwargs)
                    conn.send((self._RESULT, result))
                    continue
                if message == self._CLOSE:
                    break
                raise KeyError("Received message of unknown type {}".format(message))
        except Exception:
            stacktrace = "".join(traceback.format_exception(*sys.exc_info()))
            print("Error in environment process: {}".format(stacktrace))
            conn.send((self._EXCEPTION, stacktrace))
        finally:
            try:
                conn.close()
            except IOError:
                pass  # The connection was already closed.
