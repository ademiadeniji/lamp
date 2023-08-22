import collections
import datetime
import io
import pathlib
import uuid

import numpy as np
import tensorflow as tf

import albumentations as A


class Replay:
    def __init__(
        self,
        directory,
        load_directory=None,
        capacity=0,
        minlen=1,
        maxlen=0,
        prioritize_ends=False,
    ):
        self._directory = pathlib.Path(directory).expanduser()
        self._directory.mkdir(parents=True, exist_ok=True)
        self._capacity = capacity
        self._minlen = minlen
        self._maxlen = maxlen
        self._prioritize_ends = prioritize_ends
        self._random = np.random.RandomState()

        self.load_directory = load_directory
        if load_directory is None:
            load_directory = self._directory
        else:
            load_directory = pathlib.Path(load_directory).expanduser()
        # filename -> key -> value_sequence
        self._complete_eps = load_episodes(load_directory, capacity, minlen)
        if len(self._complete_eps) != 0:
            self._eps_keys = list(self._complete_eps.keys())
            idxs = np.sort(
                np.argsort(np.random.uniform(size=len(self._eps_keys)))[
                    : int(np.ceil(len(self._eps_keys) / 2))
                ]
            )
        else:
            self._eps_keys = []
        self._eps_masks = dict()
        # worker -> key -> value_sequence
        self._ongoing_eps = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )
        self._total_episodes, self._total_steps = count_episodes(directory)
        self._loaded_episodes = len(self._complete_eps)
        self._loaded_steps = sum(eplen(x) for x in self._complete_eps.values())
        self.reward_func = None

    @property
    def stats(self):
        return {
            "total_steps": self._total_steps,
            "total_episodes": self._total_episodes,
            "loaded_steps": self._loaded_steps,
            "loaded_episodes": self._loaded_episodes,
        }

    def reward_relabel(self, reward_func):
        self.reward_func = reward_func

    def add_step(self, transition, worker=0):
        episode = self._ongoing_eps[worker]
        for key, value in transition.items():
            episode[key].append(value)
        if transition["is_last"]:
            self.add_episode(episode)
            episode.clear()

    def add_episode(self, episode):
        if 'lang_num' not in episode.keys() or len(episode['lang_num']) != len(episode['reward']):
            # running into errors when running pure RL, no multi-task learning
            # there's likely a better way to fix, but couldn't find it
            episode['lang_num'] = [0] * len(episode['reward'])
        length = eplen(episode)
        if length < self._minlen:
            print(f"Skipping short episode of length {length}.")
            return
        self._total_steps += length
        self._loaded_steps += length
        self._total_episodes += 1
        self._loaded_episodes += 1
        episode = {key: convert(value) for key, value in episode.items()}
        if self.reward_func is not None:
            vidlang_reward = self.reward_func(episode)
            episode["reward"] = vidlang_reward
        filename = save_episode(self._directory, episode)
        self._complete_eps[str(filename)] = episode

        self._eps_keys.append(str(filename))
        self._enforce_limit()

    def add_demo_episode(self, episode):
        success = np.sum(episode["success"]) >= 1.0
        if success:
            self.add_episode(episode)

    def dataset(self, batch, length):
        example = next(iter(self._generate_chunks(length)))
        dataset = tf.data.Dataset.from_generator(
            lambda: self._generate_chunks(length),
            {k: v.dtype for k, v in example.items()},
            {k: v.shape for k, v in example.items()},
        )
        dataset = dataset.batch(batch, drop_remainder=True)
        dataset = dataset.prefetch(5)
        return dataset

    def _generate_chunks(self, length):
        sequence = self._sample_sequence()
        while True:
            chunk = collections.defaultdict(list)
            added = 0
            while added < length:
                needed = length - added
                adding = {k: v[:needed] for k, v in sequence.items()}
                sequence = {k: v[needed:] for k, v in sequence.items()}
                for key, value in adding.items():
                    chunk[key].append(value)
                added += len(adding["reward"])
                if len(sequence["reward"]) < 1:
                    sequence = self._sample_sequence()
            chunk = {k: np.concatenate(v) for k, v in chunk.items()}
            yield chunk

    def _sample_sequence(self):
        eps_keys = self._eps_keys
        L = len(eps_keys)
        i = np.random.randint(0, L)
        episode_key = eps_keys[i]
        episode = self._complete_eps[episode_key]

        total = len(episode["reward"])
        length = total
        if self._maxlen:
            length = min(length, self._maxlen)
        # Randomize length to avoid all chunks ending at the same time in case the
        # episodes are all of the same length.
        length -= np.random.randint(self._minlen)
        length = max(self._minlen, length)
        upper = total - length + 1
        if self._prioritize_ends:
            upper += self._minlen
        index = min(self._random.randint(upper), total - length)
        sequence = {
            k: convert(v[index : index + length])
            for k, v in episode.items()
            if not k.startswith("log_")
        }
        sequence["is_first"] = np.zeros(len(sequence["reward"]), bool)
        sequence["is_first"][0] = True
        if self._maxlen:
            assert self._minlen <= len(sequence["reward"]) <= self._maxlen
        return sequence

    def _enforce_limit(self):
        if not self._capacity:
            return
        while self._loaded_episodes > 1 and self._loaded_steps > self._capacity:
            # Relying on Python preserving the insertion order of dicts.
            oldest, episode = next(iter(self._complete_eps.items()))
            self._loaded_steps -= eplen(episode)
            self._loaded_episodes -= 1
            del self._complete_eps[oldest]


def count_episodes(directory):
    filenames = list(directory.glob("*.npz"))
    num_episodes = len(filenames)
    num_steps = sum(int(str(n).split("-")[-1][:-4]) - 1 for n in filenames)
    return num_episodes, num_steps


def save_episode(directory, episode):
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    identifier = str(uuid.uuid4().hex)
    length = eplen(episode)
    filename = directory / f"{timestamp}-{identifier}-{length}.npz"
    with io.BytesIO() as f1:
        np.savez_compressed(f1, **episode)
        f1.seek(0)
        with filename.open("wb") as f2:
            f2.write(f1.read())
    return filename


def load_episodes(directory, capacity=None, minlen=1):
    # The returned directory from filenames to episodes is guaranteed to be in
    # temporally sorted order.
    filenames = sorted(directory.glob("*.npz"))
    if capacity:
        num_steps = 0
        num_episodes = 0
        for filename in reversed(filenames):
            length = int(str(filename).split("-")[-1][:-4])
            num_steps += length
            num_episodes += 1
            if num_steps >= capacity:
                break
        filenames = filenames[-num_episodes:]
    episodes = {}
    for filename in filenames:
        try:
            with filename.open("rb") as f:
                episode = np.load(f)
                episode = {k: episode[k] for k in episode.keys()}
        except Exception as e:
            print(f"Could not load episode {str(filename)}: {e}")
            continue
        episodes[str(filename)] = episode
    return episodes


def convert(value):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
        return value.astype(np.float32)
    elif np.issubdtype(value.dtype, np.signedinteger):
        return value.astype(np.int32)
    elif np.issubdtype(value.dtype, np.uint8):
        return value.astype(np.uint8)
    return value


def eplen(episode):
    return len(episode["reward"]) - 1
