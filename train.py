import collections
import functools
import logging
import os
import pathlib
import re
import sys
import warnings
import torch
import itertools
from omegaconf import OmegaConf

try:
    import rich.traceback

    rich.traceback.install()
except ImportError:
    pass

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger().setLevel("ERROR")
warnings.filterwarnings("ignore", ".*box bound precision lowered.*")

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
from keras import backend as K
import ruamel.yaml as yaml

import common

def process_curriculum_str(lang_curriculum_str):
    cfg_lang_prompts = lang_curriculum_str.split("|")
    lang_curriculum = []
    for i in range(len(cfg_lang_prompts)):
        lang_curriculum.append(cfg_lang_prompts[i].split(","))

    for i in range(len(lang_curriculum)):
        for j in range(len(lang_curriculum[i])):
            if os.path.exists(lang_curriculum[i][j]):
                with open(lang_curriculum[i][j], 'r') as f:
                    lang_curriculum[i].pop(j)
                    lang_curriculum[i].extend(f.read().splitlines())

    lang_instructions = list(itertools.chain(*lang_curriculum))
    lang_instructions = list(set(lang_instructions))
    return lang_curriculum, lang_instructions

def get_lang_info(multi_task_vidlang, task, lang_prompt, synonym_folder, objects):
    """
    Returns
    final_lang_instructions: list of all possible processed language instructions
    lang_to_num: dict mapping language instruction (from above) to number
    lang_curriculum: list of non-processed instructions for each stage of curriculum
    synonym_dict: dict mapping object to object synonyms
    """
    if multi_task_vidlang:
        lang_instructions = task.split(',')
        return lang_instructions, None, None
    else:
        lang_curriculum, lang_instructions = process_curriculum_str(lang_prompt)

        _, all_objs = process_curriculum_str(objects)
        noun_variations = []
        synonym_dict = {}
        if synonym_folder is None:
            for obj in all_objs:
                synonym_dict[obj] = obj
            noun_variations = all_objs
        else:
            for obj in all_objs:
                with open(os.path.join(synonym_folder, f"synonym_{obj}.txt"), 'r') as f:
                    synonyms = f.read().splitlines()
                    synonym_dict[obj] = synonyms
                    noun_variations.extend(synonyms)
        noun_variations = list(set(noun_variations))

        final_lang_instructions = []
        for lang_instr in lang_instructions:
            if "[NOUN]" in lang_instr:
                for noun in noun_variations:
                    final_lang_instructions.append(lang_instr.replace("[NOUN]", noun))
            else:
                final_lang_instructions.append(lang_instr)

        lang_nums = range(len(final_lang_instructions))
        lang_to_num = dict(zip(final_lang_instructions, lang_nums))
        return final_lang_instructions, lang_to_num, lang_curriculum, synonym_dict

def main():

    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )
    parsed, remaining = common.Flags(configs=["defaults"]).parse(known_only=True)
    config = common.Config(configs["defaults"])
    for name in parsed.configs:
        config = config.update(configs[name])
    config = common.Flags(config).parse(remaining)

    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    config.save(logdir / "config.yaml")

    print(config, "\n")
    print("Logdir", logdir)

    loaddir = pathlib.Path(config.loaddir).expanduser()
    print("Loaddir", loaddir)

    import tensorflow as tf
    tf.config.experimental_run_functions_eagerly(not config.jit)
    message = "No GPU found. To actually train on CPU remove this assert."
    assert tf.config.experimental.list_physical_devices("GPU"), message
    print(tf.config.experimental.list_physical_devices("GPU"))
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    assert config.precision in (16, 32), config.precision
    if config.precision == 16:
        from tensorflow.keras.mixed_precision import experimental as prec

        prec.set_policy(prec.Policy("mixed_float16"))

    train_replay = common.Replay(logdir / "train_episodes", **config.replay)
    eval_replay = common.Replay(
        logdir / "eval_episodes",
        **dict(
            capacity=config.replay.capacity // 10,
            minlen=config.replay.minlen,
            maxlen=config.replay.maxlen,
        ),
    )
    step = common.Counter(train_replay.stats["total_steps"])
    outputs =   [
        common.TerminalOutput(),
        common.JSONLOutput(logdir),
        common.TensorBoardOutput(logdir),
    ]
    logger = common.Logger(step, outputs, multiplier=config.action_repeat)
    metrics = collections.defaultdict(list)

    should_train = common.Every(config.train_every)
    should_train_mae = common.Every(config.train_mae_every)
    should_log = common.Every(config.log_every)

    lang_instructions, lang_to_num, lang_curriculum, synonym_dict = get_lang_info(config.multi_task_vidlang, config.task, config.curriculum.lang_prompt, config.curriculum.synonym_folder, config.curriculum.objects)

    if (loaddir / f"variables_{config.ts}.pkl").exists(): 
        with open(loaddir / 'config.yaml', 'r') as f:
            pt_cfg = OmegaConf.create(yaml.safe_load(f))
        pt_lang_instructions, _, _, _ = get_lang_info(pt_cfg.multi_task_vidlang, pt_cfg.task, pt_cfg.curriculum.lang_prompt, pt_cfg.curriculum.synonym_folder, pt_cfg.curriculum.objects)
        if (pt_cfg.plan2explore or pt_cfg.rnd):
            if (pt_cfg.use_r3m_reward or pt_cfg.use_internvideo_reward or pt_cfg.use_clip_reward):
                config = config.update({'mae.state_dim': config.mae.state_dim + 768})
                config = config.update({'pretrain_mode': 'lang_emb'})
            else:
                config = config.update({'pretrain_mode': 'no_lang'})
        elif pt_cfg.use_lang_embeddings:
            config = config.update({'mae.state_dim': config.mae.state_dim + 768})
            config = config.update({'pretrain_mode': 'lang_emb'})
        else:
            config = config.update({'mae.state_dim': config.mae.state_dim + len(pt_lang_instructions)})
            config = config.update({'pretrain_mode': 'one_hot'})
        config = config.update({'num_langs': len(pt_lang_instructions)})
        config = config.update({'train_mode': 'finetune'})
    elif (config.plan2explore or config.rnd) and not (config.use_r3m_reward or config.use_internvideo_reward or config.use_clip_reward):
        config = config.update({'train_mode': 'pretrain'})
    else:
        config = config.update({'pretrain_mode': None})
        if config.use_lang_embeddings:
            config = config.update({'mae.state_dim': config.mae.state_dim + 768})
        else:
            config = config.update({'mae.state_dim': config.mae.state_dim + len(lang_instructions)})
        config = config.update({'num_langs': len(lang_instructions)})
        config = config.update({'train_mode': 'pretrain'})
    config.save(logdir / "config_updated.yaml")

    finetune_lang_encoding = None
    if config.train_mode == "finetune" and config.pretrain_mode == "lang_emb":
        import r3mreward as r3mreward
        from r3m import load_r3m
        finetune_instruction = config.task.replace("_", " ")
        model = load_r3m('resnet50').module.eval().to(config.vidlang_model_device)
        model = r3mreward.R3MReward(model, [finetune_instruction], config.standardize_rewards, 
            config.queue_size, config.update_stats_steps, config.num_top_images, config.use_lang_embeddings)
        finetune_lang_encoding = model.get_lang_encoding([finetune_instruction]).cpu().numpy()[0]
        if config.tune_instruction:
            if os.path.exists(config.instructions_file):
                print("Loading instructions to tune with...")
                with open(config.instructions_file, 'r') as f:
                    candidate_instructions = f.read().splitlines()
                candidate_instructions = candidate_instructions[:config.num_tune_instructions-10]
                print(f"Loaded {config.num_tune_instructions} instructions from {config.instructions_file}")
            else:
                print("Generating instructions to tune with...")
                pt_lang_instructions, _, lang_curriculum, _ = get_lang_info(pt_cfg.multi_task_vidlang, pt_cfg.task, pt_cfg.curriculum.lang_prompt, pt_cfg.curriculum.synonym_folder, pt_cfg.curriculum.objects)
                candidate_instructions = np.random.choice(pt_lang_instructions, config.num_tune_instructions-10, replace=False)
                with open(f"prompts/tune_instructions_{config.num_tune_instructions}.txt", 'w') as f:
                    for instr in candidate_instructions:
                        f.write(instr + '\n')
                print(f"Generated {config.num_tune_instructions} instructions and saved to prompts/tune_instructions_{config.num_tune_instructions}.txt")
            with open(f"prompts/{config.task}.txt", 'r') as f:
                task_candidate_instructions = f.read().splitlines()
                candidate_instructions.extend(task_candidate_instructions)
            candidate_instructions_encodings = []
            for instr in candidate_instructions:
                candidate_instructions_encodings.append(model.get_lang_encoding([instr]).cpu().numpy()[0])
            candidate_instructions_scores = [0] * len(candidate_instructions)
            global score_idx 
            score_idx = 0
        del model
        import gc; gc.collect()
        lang_instructions = [finetune_instruction]

    if not config.use_r3m_reward and config.train_mode == "pretrain" and config.use_lang_embeddings:
        import r3mreward as r3mreward
        from r3m import load_r3m
        sentences = [t.replace("_", " ") for t in lang_instructions]
        model = load_r3m('resnet50').module.eval().to(config.vidlang_model_device)
        model = r3mreward.R3MReward(model, sentences, config.standardize_rewards, 
            config.queue_size, config.update_stats_steps, config.num_top_images, config.use_lang_embeddings)
        lang_encodings = model.get_lang_encoding(sentences).cpu().numpy()
        lang_to_encoding = dict(zip(lang_instructions, lang_encodings))
        del model
        import gc; gc.collect()
            

    if config.use_r3m_reward: 
        import r3mreward as r3mreward
        from r3m import load_r3m
        sentences = [t.replace("_", " ") for t in lang_instructions]

        model = load_r3m('resnet50').module.eval().to(config.vidlang_model_device)
        model = r3mreward.R3MReward(model, sentences, config.standardize_rewards, 
            config.queue_size, config.update_stats_steps, config.num_top_images, config.use_lang_embeddings)
        if config.use_lang_embeddings:
            lang_encodings = model.get_lang_encoding(sentences).cpu().numpy()
            lang_to_encoding = dict(zip(lang_instructions, lang_encodings))
    
        def get_r3m_reward(data, step=0):
            with torch.no_grad():
                if config.camera_keys == "image_front|image_wrist" or config.camera_keys == "image_overhead|image_wrist":
                    init_image = np.split(data['init_image'], 2, axis=-2)[0]
                    image = np.split(data['image'], 2, axis=-2)[0]
                else:
                    init_image = data['init_image']
                    image = data['image']
                init_image = torch.from_numpy(init_image).to(config.vidlang_model_device)
                image = torch.from_numpy(image).to(config.vidlang_model_device)
                lang_num = torch.from_numpy(data['lang_num']).to(config.vidlang_model_device).unsqueeze(-1)
                if config.use_lang_embeddings:
                    lang_embedding = torch.from_numpy(data['lang_embedding']).to(config.vidlang_model_device)
                else:
                    lang_embedding = None
                init_image = init_image.permute((0, 3, 1, 2))
                image = image.permute((0, 3, 1, 2))
                reward, _, _ = model.get_reward(init_image, image, lang_num, lang_embedding, step)
                return reward.squeeze(-1).cpu().numpy()

        train_replay.reward_relabel(get_r3m_reward)
    elif config.use_internvideo_reward:
        sys.path.append('InternVideo/Pretrain/Multi-Modalities-Pretraining')
        import InternVideo
        from InternVideo import video_transform
        from torchvision import transforms
        print('Loading InternVideo model from path: {}...'.format(config.internvideo_load_dir))
        model = InternVideo.load_model(config.internvideo_load_dir).cuda().to(config.vidlang_model_device)
        upsample = torch.nn.Upsample(size=(224,224), mode='bilinear', align_corners=False)
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        trans = transforms.Compose([
            video_transform.ClipToTensor(channel_nb=3),
            video_transform.Normalize(mean=input_mean, std=input_std)
        ])
        def get_internvideo_reward(data, step=0):
            with torch.no_grad():
                if config.camera_keys == "image_front|image_wrist" or config.camera_keys == "image_overhead|image_wrist":
                    image = np.split(data['image'], 2, axis=-2)[0]
                else:
                    image = data['image']
                videos = trans(image).to(config.vidlang_model_device)
                videos = upsample(videos)
                text_cand = [lang_instructions[lang_num] for lang_num in data["lang_num"]]
                text = InternVideo.tokenize(text_cand).cuda().to(config.vidlang_model_device)
                text_features = model.encode_text(text)
                
                reward = []
                for i in range(videos.shape[1]):
                    if i < 8:
                        video = videos[:, :i+1, :, :]
                        video = torch.cat([video, video[:, -1:, :, :].repeat(1, 8-(i+1), 1, 1)], dim=1).unsqueeze(0)
                    else:  
                        indices = np.ceil(np.linspace(0, i, 8)).astype(int)
                        video = videos[:, indices, :, :].unsqueeze(0)
                    video_features = model.encode_video(video)
                    video_features = torch.nn.functional.normalize(video_features, dim=1)
                    text_features = torch.nn.functional.normalize(text_features, dim=1)
                    t = model.logit_scale.exp()
                    reward.append((video_features @ text_features[i]).cpu().numpy())
                
                return np.array(reward).squeeze(-1)

        train_replay.reward_relabel(get_internvideo_reward)
    elif config.use_clip_reward:
        import clip
        sys.path.append('InternVideo/Pretrain/Multi-Modalities-Pretraining')
        import InternVideo
        from InternVideo import video_transform
        from torchvision import transforms
        model, _ = clip.load('ViT-B/32', config.vidlang_model_device)
        upsample = torch.nn.Upsample(size=(224,224), mode='bilinear', align_corners=False)
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        trans = transforms.Compose([
            video_transform.ClipToTensor(channel_nb=3),
            video_transform.Normalize(mean=input_mean, std=input_std)
        ])
        def get_clip_reward(data, step=0):
            with torch.no_grad():
                if config.camera_keys == "image_front|image_wrist" or config.camera_keys == "image_overhead|image_wrist":
                    image = np.split(data['image'], 2, axis=-2)[0]
                    init_image = np.split(data['init_image'], 2, axis=-2)[0]
                else:
                    image = data['image']
                    init_image = data['init_image']
                init_videos = trans(init_image).to(config.vidlang_model_device)
                init_videos = upsample(init_videos)
                videos = trans(image).to(config.vidlang_model_device)
                videos = upsample(videos)
                text_inputs = torch.cat([clip.tokenize(lang_instructions[lang_num]) for lang_num in data["lang_num"]]).to(config.vidlang_model_device)
                init_image_inputs = init_videos.permute((1, 0, 2, 3))
                image_inputs = videos.permute((1, 0, 2, 3))
                init_image_features = model.encode_image(init_image_inputs)
                image_features = model.encode_image(image_inputs)
                init_text_features = model.encode_text(text_inputs)
                text_features = model.encode_text(text_inputs)
                init_image_features /= init_image_features.norm(dim=-1, keepdim=True)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                init_text_features /= init_text_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                delta_image_features = image_features - init_image_features
                delta_text_features = text_features - init_text_features
                reward = (delta_image_features * delta_text_features).sum(dim=-1).cpu().numpy()
                return reward
        train_replay.reward_relabel(get_clip_reward)
    else:
        print("Training from task reward...")

    if not config.use_lang_embeddings:
        lang_to_encoding = None
    def make_env(mode, actions_min_max=None):
        camera_keys = common.get_camera_keys(config.camera_keys)
        task = config.task.split(",")

        if config.franka_kitchen:
            env = common.KitchenEnv(task)
        elif config.multi_task_vidlang:
            env = common.MultiTaskVidLangRLBench(task,
                camera_keys,
                config.render_size,
                shaped_rewards=config.shaped_rewards,
                lang_to_num=lang_to_num,
                lang_to_encoding=lang_to_encoding,
                use_lang_embeddings=config.use_lang_embeddings,
                boundary_reward_penalty=config.boundary_reward_penalty,
                randomize=config.randomize,
            )
        elif config.use_r3m_reward or config.use_internvideo_reward or config.use_clip_reward:
            env = common.VidLangRLBench(task[0],
                lang_instructions,
                camera_keys,
                config.render_size,
                shaped_rewards=config.shaped_rewards,
                lang_to_num=lang_to_num,
                lang_to_encoding=lang_to_encoding,
                use_lang_embeddings=config.use_lang_embeddings,
                boundary_reward_penalty=config.boundary_reward_penalty,
                curriculum=config.curriculum,
                lang_curriculum=lang_curriculum,
                synonym_dict=synonym_dict,
                randomize=config.randomize
            )
        else:
            env = common.RLBench(
                lang_instructions,
                task[0],
                camera_keys,
                config.render_size,
                shaped_rewards=config.shaped_rewards,
                use_lang_embeddings=config.use_lang_embeddings,
                randomize=config.randomize,
                finetune_lang_encoding=finetune_lang_encoding
            )
        if actions_min_max:
            env.register_min_max(actions_min_max)

        env = common.TimeLimit(env, config.time_limit)
        return env

    def per_episode(ep, mode, lang_instructions=None):
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        success = float(np.sum(ep["success"]) >= 1.0)
        print(
            f"{mode.title()} episode has {float(success)} success, {length} steps and return {score:.1f}."
        )
        logger.scalar(f"{mode}_success", float(success))
        logger.scalar(f"{mode}_return", score)
        logger.scalar(f"{mode}_length", length)
        for key, value in ep.items():
            if re.match(config.log_keys_sum, key):
                logger.scalar(f"sum_{mode}_{key}", ep[key].sum())
            if re.match(config.log_keys_mean, key):
                logger.scalar(f"mean_{mode}_{key}", ep[key].mean())
            if re.match(config.log_keys_max, key):
                logger.scalar(f"max_{mode}_{key}", ep[key].max(0).mean())
        replay = dict(train=train_replay, eval=eval_replay)[mode]
        logger.add(replay.stats, prefix=mode)
        logger.write()

    print("Create envs.")
    num_eval_envs = min(config.envs, config.eval_eps)
    train_envs = [make_env("train") for _ in range(config.envs)]

    
    actions_min_max = None

    act_space = train_envs[0].act_space
    obs_space = train_envs[0].obs_space

    import agent as agent

    agnt = agent.Agent(config, obs_space, act_space, step)
    eval_policy = lambda *args: agnt.policy(*args, mode="eval")

    if config.tune_instruction:
        print("Tuning instruction...")
        def tune(ep, candidate_instructions_encodings=candidate_instructions_encodings, candidate_instructions_scores=candidate_instructions_scores):
            global score_idx 
            candidate_instructions_scores[score_idx] = ep["reward"].sum()
            score_idx += 1

        tune_driver = common.Driver(train_envs)
        tune_driver.on_episode(lambda ep: tune(ep))
        train_envs[0].change_lang_encoding(candidate_instructions_encodings[0])
        tune_driver(eval_policy, episodes=config.num_tune_instructions)
        max_score_idx = np.argmax(np.array(candidate_instructions_scores))
        print("Best instruction: {} with reward {}".format(candidate_instructions[max_score_idx], candidate_instructions_scores[max_score_idx]))
        train_envs[0].change_lang_encoding(candidate_instructions_encodings[max_score_idx])
        finetune_lang_encoding = candidate_instructions_encodings[max_score_idx]

    make_async_env = lambda mode: common.Async(
        functools.partial(make_env, mode, actions_min_max), config.envs_parallel
    )
    eval_envs = [make_async_env("eval") for _ in range(num_eval_envs)]

    print("Creating train and eval drivers.")
    train_driver = common.Driver(train_envs)
    train_driver.on_episode(lambda ep: per_episode(ep, mode="train"))
    train_driver.on_step(lambda tran, worker: step.increment())
    train_driver.on_episode(train_replay.add_episode)
    eval_driver = common.Driver(eval_envs)
    eval_driver.on_episode(lambda ep: per_episode(ep, mode="eval", lang_instructions=lang_instructions))
    eval_driver.on_episode(eval_replay.add_episode)

    prefill = max(0, config.prefill - train_replay.stats["total_steps"])
    if prefill:
        print(f"Prefill dataset ({prefill} steps).")
        random_agent = common.RandomAgent(act_space)
        train_driver(random_agent, steps=prefill, episodes=1)
        eval_driver(random_agent, episodes=1)
        train_driver.reset()
        eval_driver.reset()

    print("Create agent.")
    train_dataset = iter(train_replay.dataset(**config.dataset))
    mae_train_dataset = iter(train_replay.dataset(**config.mae_dataset))
    report_dataset = iter(train_replay.dataset(**config.dataset))

    if not config.use_imagenet_mae:
        train_mae = agnt.train_mae
    train_agent = common.CarryOverState(agnt.train)
    
    if not config.use_imagenet_mae:
        train_mae(next(mae_train_dataset))
    train_agent(next(train_dataset))

    if (loaddir / f"variables_{config.ts}.pkl").exists():
        print("Loading agent.") 
        try:
            agnt.load(loaddir / f"variables_{config.ts}.pkl")
        except Exception as e:
            raise Exception(f"Error loading agent: {e}")
    else:
        assert config.loaddir == ''
    
    print("Pretrain agent.")
    for _ in range(config.mae_pretrain):
        data = next(mae_train_dataset)
        if config.use_zero_rewards:
            data['reward'] = data['reward'] * tf.cast(data['is_terminal'], tf.float32)
        train_mae(data)
    for _ in range(config.pretrain):
        data = next(train_dataset)
        if config.use_zero_rewards:
            data['reward'] = data['reward'] * tf.cast(data['is_terminal'], tf.float32)
        train_agent(data)

    train_policy = lambda *args: agnt.policy(*args, mode="train")

    def train_step(tran, worker):
        if not config.use_imagenet_mae:
            if should_train_mae(step):
                for _ in range(config.train_mae_steps):
                    data = next(mae_train_dataset)
                    if config.use_zero_rewards:
                        data['reward'] = data['reward'] * tf.cast(data['is_terminal'], tf.float32)
                    mets = train_mae(data)
                    [metrics[key].append(value) for key, value in mets.items()]
        if should_train(step):
            for _ in range(config.train_steps):
                data = next(train_dataset)
                if config.use_zero_rewards:
                    data['reward'] = data['reward'] * tf.cast(data['is_terminal'], tf.float32)
                mets = train_agent(data)
                [metrics[key].append(value) for key, value in mets.items()]
        if should_log(step):
            for name, values in metrics.items():
                logger.scalar(name, np.array(values, np.float64).mean())
                metrics[name].clear()
            logger.add(
                agnt.report(next(report_dataset)),
                prefix="train",
            )
            logger.write(fps=True)

    train_driver.on_step(train_step)

    config.save(logdir / "config_updated.yaml")
    while step < config.steps:
        logger.write()
        print("Start evaluation.")
        eval_driver(eval_policy, episodes=config.eval_eps)
        print("Start training.")
        train_driver(train_policy, steps=config.eval_every)
        agnt.save(logdir / f"variables_{step.value}.pkl")
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass
    agnt.save(logdir / f"variables_final.pkl")


if __name__ == "__main__":
    main()
