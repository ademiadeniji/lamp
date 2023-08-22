from collections import deque
import torch
import numpy as np

class R3MReward(object):
    def __init__(self, embed_model, sentences, standardize_rewards, queue_size, update_stats_steps, num_top_images, use_lang_embeddings):
        self.embed_model = embed_model
        self.reward_model = self.embed_model.get_reward
        self.sentences = sentences
        self.standardize_rewards = standardize_rewards
        self.use_sharding_r3m = False
        self.log_top_images = True
        self.r3m_reward_bonus = 10.0
        self.update_stats_steps = update_stats_steps
        self.use_lang_embeddings = use_lang_embeddings
        if self.standardize_rewards:
            self.stats = {}
            for t in self.sentences:
                self.stats[t] = deque(maxlen=queue_size)
        if self.log_top_images:
            self.top_images = {}
            for t in self.sentences:
                self.top_images[t] = {"images": [], "rewards": []}
            self.num_top_images = num_top_images

    def get_lang_encoding(self, lang):
        return self.embed_model.lang_enc(lang)

    def get_reward(self, init, curr, lang, lang_emb, step=None):
        if isinstance(lang, int):
            lang_strings = [self.sentences[lang]]
            init_image = torch.unsqueeze(init, 0)
            curr_image = torch.unsqueeze(curr, 0)
        else:
            lang_strings = [self.sentences[i] for i in lang[:, 0]]
            init_image = init
            curr_image = curr
        
        
        init = self.embed_model(init_image[:, -3:, :, :])
        curr = self.embed_model(curr_image[:, -3:, :, :])
        if self.use_lang_embeddings:
            reward = self.embed_model.get_reward_le(init, curr, lang_emb)[0].unsqueeze(-1) 
        else:
            reward = self.reward_model(init, curr, lang_strings)[0].unsqueeze(-1) 
        
        return reward, None, None