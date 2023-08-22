import random
import re
import copy
import numpy as np


def collect_demo(
    env,
    replay,
    num_demos,
    camera_keys,
    shaped_rewards=False,
    task_name_to_num=None,
):
    transitions = []
    for _ in range(num_demos):
        success = False
        while not success:
            try:
                demo = env._task.get_demos(1, live_demos=True)[0]
                success = True
            except:
                pass
        transitions.extend(extract_from_demo(demo, shaped_rewards, camera_keys, env.task_name, task_name_to_num))

    # Restrict translation space by min_max
    actions = []

    for obs in transitions:
        if obs["is_first"]:
            continue
        action = obs["action"]
        actions.append(action)

    low, high = np.min(actions, 0)[:3], np.max(actions, 0)[:3]
    low -= 0.2 * np.fabs(low)
    high += 0.2 * np.fabs(high)

    for obs in transitions:
        if obs["is_first"]:
            # for first action, let's just label with zero action
            obs["action"] = np.zeros(3 + 1)
        else:
            action = obs["action"]
            updated_action = []

            pose = action[:3]
            norm_pose = 2 * ((pose - low) / (high - low)) - 1
            updated_action.append(norm_pose)

            gripper = action[3:4]
            norm_gripper = gripper * 2 - 1.0
            updated_action.append(norm_gripper)

            obs["action"] = np.hstack(updated_action)

        replay.add_step(obs)

    print(f"Position min/max: {low}/{high}")
    actions_min_max = low, high

    return actions_min_max


def get_action(prev_obs, obs):
    prev_pose = prev_obs.gripper_pose[:3]
    cur_pose = obs.gripper_pose[:3]
    pose = cur_pose - prev_pose
    gripper_action = float(obs.gripper_open)
    prev_action = np.hstack([pose, gripper_action])
    return prev_action


def extract_from_demo(demo, shaped_rewards, camera_keys, task_name=None, task_name_to_num=None):
    transitions = []
    if task_name_to_num is not None:
        init_image, init_state = None, None
    for k, obs in enumerate(demo):
        if k == 0:
            prev_action = None
        else:
            prev_obs = demo[k - 1]
            prev_action = get_action(prev_obs, obs)

        terminal = k == len(demo) - 1
        first = k == 0
        success = terminal

        if shaped_rewards:
            reward = obs.task_low_dim_state[0]
        else:
            reward = float(success)

        # Not to override obs
        _obs = copy.deepcopy(obs)
        _obs.joint_velocities = None
        _obs.joint_positions = None
        _obs.task_low_dim_state = None

        transition = {
            "reward": reward,
            "is_first": first,
            "is_last": False,
            "is_terminal": False,
            "success": success,
            "action": prev_action,
            "state": _obs.get_low_dim_data(),
        }

        keys = get_camera_keys(camera_keys)
        images = []
        for key in keys:
            if key == "image_front":
                images.append(_obs.front_rgb)
            if key == "image_wrist":
                images.append(_obs.wrist_rgb)
        transition["image"] = np.concatenate(images, axis=-2)
        if task_name_to_num is not None:
            if k == 0:
                init_image = transition["image"]
                init_state = transition["state"]
            transition["init_image"] = init_image
            transition["init_state"] = init_state
            transition['task_num'] = task_name_to_num[task_name]
        transitions.append(transition)

    if len(transitions) % 50 == 0:
        time_limit = len(transitions)
    else:
        time_limit = 50 * (1 + (len(transitions) // 50))
    while len(transitions) < time_limit:
        transitions.append(copy.deepcopy(transition))
    transitions[-1]["is_last"] = True
    return transitions


def get_camera_keys(keys):
    camera_keys = keys.split("|")
    return camera_keys
