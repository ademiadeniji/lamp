import numpy as np
import pickle
import tfimm

from tfimm.architectures.vit import ViTBlock
from tfimm.layers import PatchEmbeddings
from tfimm.layers.factory import norm_layer_factory

import common

import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow.keras import layers as tfkl
from tensorflow.keras import mixed_precision as prec


class MaskedViTEncoder(common.Module):
    def __init__(
        self,
        img_h_size,
        img_w_size,
        ncams,
        patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        norm_layer="layer_norm_eps_1e-6",
        early_conv=False,
        state_pred=False,
        view_masking=False,
        control_input="front_wrist",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.img_h_size, self.img_w_size = img_h_size, img_w_size
        self.ncams = ncams
        self.num_patches = (img_h_size // patch_size) * (img_w_size // patch_size)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self.early_conv = early_conv
        self.state_pred = state_pred
        self.view_masking = view_masking
        self.control_input = control_input
        self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

        self.construct_pos_embed()

    def construct_pos_embed(self):
        each_view_w = self.img_w_size // self.ncams
        each_view_h = self.img_h_size

        self.cls_pos_embed = tf.constant(
            np.zeros([1, self.embed_dim]), name="cls_pos_embed", dtype=tf.float32
        )
        self.pos_embed = tf.constant(
            common.mae_utils.get_2d_sincos_pos_embed(
                self.embed_dim,
                int(each_view_h // self.patch_size),
                int(each_view_w // self.patch_size),
            )[None],
            name="front_pos_embed",
            dtype=tf.float32,
        )

    def random_view_masking(self, x, mask_ratio, T, ncams, view_masking=False):
        N, L, D = x.shape

        if view_masking == False or ncams == 1:
            noise = tf.random.uniform([N, L], 0.0, 1.0)
            if mask_ratio == 0.0:
                noise = tf.sort(noise)
            len_keep = int(L * (1 - mask_ratio))
        elif ncams == 2:
            assert ncams == 2
            len_keep = int(L * (1 - mask_ratio))

            if self.state_pred:
                state_noise = tf.random.uniform([N, T], 0.0, 1.0)
                img_x = x[:, :-T]
            else:
                img_x = x
            img_xs = tf.split(img_x, T, axis=1)
            noises = []
            for t in range(T):
                _img = img_xs[t]
                _img_size = _img.shape[1] // 2
                uniform_noise = tf.random.uniform([N, _img_size], 0.0, 1.0)
                view_noise = tf.ones([N, _img_size], dtype=uniform_noise.dtype)

                # randomly select one view for each timestep
                wrist_mask = tf.concat([uniform_noise, view_noise], 1)
                front_mask = tf.concat([view_noise, uniform_noise], 1)

                view_cond = tf.random.uniform([N, 1]) > 0.5
                view_not_cond = ~view_cond

                noise = (
                    tf.cast(view_cond, wrist_mask.dtype) * wrist_mask
                    + tf.cast(view_not_cond, front_mask.dtype) * front_mask
                )
                noises.append(noise)
            noise = tf.concat(noises, axis=1)
            if self.state_pred:
                noise = tf.concat([noise, state_noise], 1)
            if mask_ratio == 0.0:
                noise = tf.random.uniform([N, L], 0.0, 1.0)
                noise = tf.sort(noise)

        # sort noise for each sample
        # keep small, remove large
        ids_shuffle = tf.argsort(noise, axis=1)
        ids_restore = tf.argsort(ids_shuffle, axis=1)

        # trick for tensorflow-gather
        row_ids = tf.ones_like(ids_shuffle) * tf.expand_dims(tf.range(N), 1)
        _ids_shuffle = tf.stack([row_ids, ids_shuffle], -1)  # [N, L, 2]
        _ids_restore = tf.stack([row_ids, ids_restore], -1)  # [N, L, 2]

        # keep the first subset
        ids_keep = _ids_shuffle[:, :len_keep]
        x_masked = tf.gather_nd(x, ids_keep)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = tf.concat([tf.zeros([N, len_keep]), tf.ones([N, L - len_keep])], axis=1)
        # unshuffle to get ther binary mask
        mask = tf.gather_nd(mask, _ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, T, state=None, dynamics=False):
        # embed patches
        x = self._cast(x)
        batch_size = tf.shape(x)[0]

        w = x.shape[-2]
        _ncams = 1 if w == (self.img_w_size // self.ncams) else self.ncams

        if self.early_conv:
            x = self.forward_early_conv(x, _ncams)
        else:
            x = self.forward_patch_embedding(x, _ncams)

        # Reshape to sequential shape
        x = tf.reshape(x, [batch_size // T, T * x.shape[1], x.shape[2]])

        if self.state_pred:
            state = tf.reshape(self._cast(state), [batch_size // T, T, state.shape[-1]])
            state = self.get("state_embed", tfkl.Dense, self.embed_dim)(state)
            x = tf.concat([x, state], 1)

        pos_embed, ncams = self.get_pos_embed(x, dynamics, T)

        # add pos embed w/o cls token
        x = x + self._cast(pos_embed)

        # masking: length -> length * mask_ratio
        masking_fn = self.random_view_masking
        x, mask, ids_restore = masking_fn(x, mask_ratio, T, ncams, self.view_masking)

        # append class token
        cls_token = self.get(
            "cls_token", common.mae_utils.Token, "cls", self.embed_dim
        )(x)
        cls_token = cls_token + self.cls_pos_embed
        cls_tokens = tf.repeat(cls_token, repeats=x.shape[0], axis=0)
        x = tf.concat([self._cast(cls_tokens), x], axis=1)

        # apply Transformer blocks
        for j in range(self.depth):
            x = self.get(
                f"vit_encoder_block_{j}",
                ViTBlock,
                embed_dim=self.embed_dim,
                nb_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.0,
                norm_layer=self.norm_layer,
                act_layer="gelu",
            )(x)
        x = self.get("vit_encoder_norm", norm_layer_factory(self.norm_layer))(x)

        return x, mask, ids_restore

    def forward_early_conv(self, x, ncams):
        # x : [B, H, W, C]
        B, H, W, C = x.shape

        if ncams != 1:
            # split across W dimension and concat to B
            x = tf.split(x, ncams, axis=2)
            x = tf.concat(x, axis=0)

        nconvs = int(np.log2(self.patch_size))
        for i in range(nconvs):
            depth = self.embed_dim // (2 ** (nconvs - i))
            x = self.get(
                f"early_conv_{i}",
                tfkl.Conv2D,
                depth,
                4,
                2,
                padding="SAME",
            )(x)
            x = tf.nn.relu(x)
        x = self.get("early_conv_proj", tfkl.Conv2D, self.embed_dim, 1, 1)(x)
        x = tf.reshape(x, [x.shape[0], -1, self.embed_dim])

        if ncams != 1:
            # split across B dimension and concat to W
            x = tf.split(x, ncams, axis=0)
            x = tf.concat(x, axis=1)
        return x

    def forward_patch_embedding(self, x, ncams):
        if ncams != 1:
            # split across W dimension and concat to B
            x = tf.split(x, ncams, axis=2)
            x = tf.concat(x, axis=0)

        x = self.get(
            "encoder_patch_embed",
            PatchEmbeddings,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            norm_layer="",
        )(x)

        if ncams != 1:
            # split across B dimension and concat to W
            x = tf.split(x, ncams, axis=0)
            x = tf.concat(x, axis=1)
        return x

    def get_pos_embed(self, x, dynamics, T):
        if self.control_input in ["front_wrist"] or (self.ncams != 1 and not dynamics):
            if self.ncams == 2:
                cams = ["front", "wrist"]
        elif self.control_input in ["overhead_wrist"] or (self.ncams != 1 and not dynamics):
            if self.ncams == 2:
                cams = ["overhead", "wrist"]            
        elif self.control_input == "front":
            cams = ["front"]
        elif self.control_input == "overhead":
            cams = ["overhead"]
        elif self.control_input == "wrist":
            cams = ["wrist"]
        else:
            raise ValueError(self.control_input)

        _pos_embed = []
        for t in range(T):
            for cam in cams:
                cam_pos_embed = self.pos_embed
                cam_token = self.get(
                    f"{cam}_token", common.mae_utils.Token, cam, self.embed_dim
                )(x)
                time_token = self.get(
                    f"time_{t}_token",
                    common.mae_utils.Token,
                    f"time_{t}",
                    self.embed_dim,
                )(x)
                cam_time_token = tf.tile(
                    cam_token + time_token, [1, cam_pos_embed.shape[1], 1]
                )
                _pos_embed.append(cam_pos_embed + cam_time_token)

        img_pos_embed = tf.concat(_pos_embed, axis=1)
        pos_embed = img_pos_embed

        if self.state_pred:
            _state_pos_embed = []
            for t in range(T):
                state_token = self.get(
                    "state_token", common.mae_utils.Token, "state", self.embed_dim
                )(x)
                time_token = self.get(
                    f"time_{t}_token",
                    common.mae_utils.Token,
                    f"time_{t}",
                    self.embed_dim,
                )(x)
                state_time_token = state_token + time_token
                _state_pos_embed.append(state_time_token)
            state_pos_embed = tf.concat(_state_pos_embed, axis=1)
            pos_embed = tf.concat([pos_embed, state_pos_embed], axis=1)

        return pos_embed, len(cams)


class MaskedViTDecoder(common.Module):
    def __init__(
        self,
        img_h_size,
        img_w_size,
        ncams,
        patch_size,
        in_chans=3,
        embed_dim=512,
        depth=8,
        num_heads=16,
        mlp_ratio=4.0,
        norm_layer="layer_norm_eps_1e-6",
        masked_decoder_loss=False,
        reward_pred=True,
        state_pred=False,
        state_dim=4,
        control_input="front_wrist",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.img_h_size, self.img_w_size = img_h_size, img_w_size
        self.ncams = ncams
        self.in_chans = in_chans
        self.num_patches = (img_h_size // patch_size) * (img_w_size // patch_size)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self.masked_decoder_loss = masked_decoder_loss
        self.reward_pred = reward_pred
        self.state_pred = state_pred
        self.state_dim = state_dim
        self.control_input = control_input
        self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

        self.construct_pos_embed()

    def construct_pos_embed(self):
        each_view_w = self.img_w_size // self.ncams
        each_view_h = self.img_h_size

        self.cls_pos_embed = tf.constant(
            np.zeros([1, self.embed_dim]),
            name="cls_pos_embed",
            dtype=tf.float32,
        )
        self.pos_embed = tf.constant(
            common.mae_utils.get_2d_sincos_pos_embed(
                self.embed_dim,
                int(each_view_h // self.patch_size),
                int(each_view_w // self.patch_size),
            )[None],
            name="pos_embed",
            dtype=tf.float32,
        )

    def patchify(self, imgs):
        """
        imgs: [N, H, W, 3]
        x: [N, L, patch_size**2 * 3]
        """
        p = self.patch_size
        c = imgs.shape[-1]
        assert imgs.shape[1] % p == 0 and imgs.shape[2] % p == 0

        x = tf.image.extract_patches(
            imgs, [1, p, p, 1], [1, p, p, 1], [1, 1, 1, 1], "VALID"
        )
        x = tf.reshape(x, [imgs.shape[0], -1, p ** 2 * c])

        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 * 3)
        imgs: (N, H, W, 3)
        """
        p = self.patch_size
        c = x.shape[-1] // (p ** 2)
        h = self.img_h_size // p
        w = self.img_w_size // p

        if h * w == x.shape[1] * self.ncams:
            # front / wrist inference case
            w = w // self.ncams
            _ncams = 1
        else:
            _ncams = self.ncams

        assert h * w == x.shape[1]

        x_split = tf.split(x, _ncams, axis=1)
        imgs = []
        for _x in x_split:
            _h, _w = h, int(w // _ncams)
            _x = tf.reshape(_x, [_x.shape[0], _h, _w, p, p, c])
            _x = tf.einsum("nhwpqc->nhpwqc", _x)
            _img = tf.reshape(_x, [_x.shape[0], _h * p, _w * p, c])
            imgs.append(_img)
        imgs = tf.concat(imgs, axis=-2)
        return imgs

    def forward_decoder(self, x, ids_restore, T, dynamics=False):
        # embed tokens
        x = self._cast(x)
        x = self.get(
            "decoder_embed",
            tfkl.Dense,
            self.embed_dim,
        )(x)

        # trick for tensorflow-gather
        N = ids_restore.shape[0]
        row_ids = tf.ones_like(ids_restore) * tf.expand_dims(tf.range(N), 1)
        ids_restore = tf.stack([row_ids, ids_restore], -1)  # [N, L, 2]

        mask_token = self.get(
            "mask_token", common.mae_utils.Token, "mask", self.embed_dim
        )(x)
        mask_tokens = self._cast(
            tf.tile(
                mask_token,
                [x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1],
            )
        )
        x_ = tf.concat([x[:, 1:, :], mask_tokens], axis=1)  # no cls token
        x_ = tf.gather_nd(x_, ids_restore)  # unshuffle

        camera_state_size = x_.shape[1]
        camera_size = camera_state_size - T if self.state_pred else camera_state_size

        # append mask token for reward prediction
        # we use same mask token for rew prediction. Maybe try different token?
        if self.reward_pred:
            rew_mask_token = self._cast(
                tf.tile(
                    mask_token,
                    [x.shape[0], T, 1],
                )
            )
            x_ = tf.concat([x_, rew_mask_token], axis=1)

        x = tf.concat([x[:, :1, :], x_], axis=1)  # append cls token

        # add pos embed
        pos_embed = self.get_pos_embed(x, dynamics, T)
        dec_pos_embed = self._cast(pos_embed)
        x = x + tf.repeat(dec_pos_embed, repeats=x.shape[0], axis=0)

        # apply Transformer blocks
        for j in range(self.depth):
            x = self.get(
                f"vit_decoder_block_{j}",
                ViTBlock,
                embed_dim=self.embed_dim,
                nb_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.0,
                norm_layer=self.norm_layer,
                act_layer="gelu",
            )(x)
        x = self.get("vit_decoder_norm", norm_layer_factory(self.norm_layer))(x)

        dec = self.get(
            "vit_decoder_pred", tfkl.Dense, self.patch_size ** 2 * self.in_chans
        )(x[:, 1 : 1 + camera_size])
        # Revert to batch-wise shape
        dec = tf.reshape(dec, [dec.shape[0] * T, dec.shape[1] // T, dec.shape[2]])

        if self.state_pred and self.reward_pred:
            state = self.get("vit_state_pred", tfkl.Dense, self.state_dim)(
                x[:, -2 * T : -T, :]
            )
            rew = self.get("vit_reward_pred", tfkl.Dense, 1)(x[:, -T:, :])
            state = tf.reshape(state, [state.shape[0] * T, 1, state.shape[2]])
            rew = tf.reshape(rew, [rew.shape[0] * T, 1, rew.shape[2]])
        elif self.reward_pred:
            state = None
            rew = self.get("vit_reward_pred", tfkl.Dense, 1)(x[:, -T:, :])
            rew = tf.reshape(rew, [rew.shape[0] * T, 1, rew.shape[2]])
        elif self.state_pred:
            state = self.get("vit_state_pred", tfkl.Dense, self.state_dim)(x[:, -T:, :])
            state = tf.reshape(state, [state.shape[0] * T, 1, state.shape[2]])
            rew = None
        else:
            state, rew = None, None
        return dec, state, rew

    def get_pos_embed(self, x, dynamics, T):
        cls_token = self.get(
            "cls_token", common.mae_utils.Token, "cls", self.embed_dim
        )(x)
        cls_pos_embed = cls_token

        if self.control_input in ["front_wrist"] or (self.ncams != 1 and not dynamics):
            if self.ncams == 2:
                cams = ["front", "wrist"]
        elif self.control_input in ["overhead_wrist"] or (self.ncams != 1 and not dynamics):
            if self.ncams == 2:
                cams = ["overhead", "wrist"]
        elif self.control_input == "front":
            cams = ["front"]
        elif self.control_input == "wrist":
            cams = ["wrist"]
        else:
            raise ValueError(self.control_input)

        _pos_embed = []
        for t in range(T):
            for cam in cams:
                cam_pos_embed = self.pos_embed
                cam_token = self.get(
                    f"{cam}_token", common.mae_utils.Token, cam, self.embed_dim
                )(x)
                time_token = self.get(
                    f"time_{t}_token",
                    common.mae_utils.Token,
                    f"time_{t}",
                    self.embed_dim,
                )(x)
                cam_time_token = tf.tile(
                    cam_token + time_token, [1, cam_pos_embed.shape[1], 1]
                )
                _pos_embed.append(cam_pos_embed + cam_time_token)

        img_pos_embed = tf.concat(_pos_embed, axis=1)
        pos_embed = tf.concat([cls_pos_embed, img_pos_embed], axis=1)

        if self.state_pred:
            _state_pos_embed = []
            for t in range(T):
                state_token = self.get(
                    "state_token",
                    common.mae_utils.Token,
                    "state",
                    self.embed_dim,
                )(x)
                time_token = self.get(
                    f"time_{t}_token",
                    common.mae_utils.Token,
                    f"time_{t}",
                    self.embed_dim,
                )(x)
                state_time_token = state_token + time_token
                _state_pos_embed.append(state_time_token)
            state_pos_embed = tf.concat(_state_pos_embed, axis=1)
            pos_embed = tf.concat([pos_embed, state_pos_embed], axis=1)

        if self.reward_pred:
            _reward_pos_embed = []
            for t in range(T):
                reward_token = self.get(
                    "reward_token",
                    common.mae_utils.Token,
                    "reward",
                    self.embed_dim,
                )(x)
                time_token = self.get(
                    f"time_{t}_token",
                    common.mae_utils.Token,
                    f"time_{t}",
                    self.embed_dim,
                )(x)
                reward_time_token = reward_token + time_token
                _reward_pos_embed.append(reward_time_token)
            reward_pos_embed = tf.concat(_reward_pos_embed, axis=1)
            pos_embed = tf.concat([pos_embed, reward_pos_embed], axis=1)

        return pos_embed

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, H, W, 3]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove
        """
        imgs = tf.cast(imgs, tf.float32)
        pred = tf.cast(pred, tf.float32)
        mask = tf.cast(mask, tf.float32)

        imgs_split = tf.split(imgs, self.ncams, axis=-2)
        target_split = [self.patchify(split) for split in imgs_split]
        target = tf.concat(target_split, axis=1)

        loss = (pred - target) ** 2
        loss = tf.reduce_mean(loss, -1)  # [N, L], mean loss per patch

        if self.masked_decoder_loss:
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()
        return loss

    def forward_reward_loss(self, rews, preds):
        rews = tf.cast(rews, tf.float32)
        preds = tf.cast(preds, tf.float32)
        dist = common.SymlogDist(preds, 1, "mean")
        loss = -dist.log_prob(rews)
        return loss.mean()

    def forward_state_loss(self, states, preds):
        states = tf.cast(states, tf.float32)
        preds = tf.cast(preds, tf.float32)
        dist = common.MSEDist(preds, 1, "mean")
        loss = -dist.log_prob(states)
        return loss.mean()


class ViTEncoder(common.Module):
    def __init__(
        self,
        img_h_size,
        img_w_size,
        patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        norm_layer="layer_norm_eps_1e-6",
        state_pred=False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.img_h_size, self.img_w_size = img_h_size, img_w_size
        self.num_patches = (img_h_size // patch_size) * (img_w_size // patch_size)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self.state_pred = state_pred
        self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

        self.construct_pos_embed()

    def construct_pos_embed(self):
        ncams = int(self.img_w_size // self.img_h_size)
        each_view_w = self.img_w_size // ncams
        each_view_h = self.img_h_size
        self.ncams = ncams

        self.pos_embed = tf.constant(
            common.mae_utils.get_2d_sincos_pos_embed(
                self.embed_dim,
                int(each_view_h // self.patch_size),
                int(each_view_w // self.patch_size),
            )[None],
            name="pos_embed",
            dtype=tf.float32,
        )

    def forward_encoder(self, x):
        # embed patches
        x = self._cast(x)
        x = self.get("encoder_embed", tfkl.Dense, self.embed_dim)(x)
        pos_embed = self.get_pos_embed(x)
        x = x + self._cast(pos_embed)

        # apply Transformer blocks
        for j in range(self.depth):
            x = self.get(
                f"vit_encoder_block_{j}",
                ViTBlock,
                embed_dim=self.embed_dim,
                nb_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.0,
                norm_layer=self.norm_layer,
                act_layer="gelu",
            )(x)
        x = self.get("vit_encoder_norm", norm_layer_factory(self.norm_layer))(x)

        return x

    def get_pos_embed(self, x):
        img_pos_embed = tf.concat([self.pos_embed] * self.ncams, axis=1)

        img_tokens = []
        for i in range(self.ncams):
            token = self.get(
                f"token_{i}",
                common.mae_utils.Token,
                f"{i}",
                self.embed_dim,
            )(x)
            token = tf.tile(
                token,
                [1, self.pos_embed.shape[1], 1],
            )
            img_tokens.append(token)
        img_token = tf.concat(img_tokens, axis=1)
        img_pos_embed = img_pos_embed + img_token

        pos_embed = img_pos_embed

        if self.state_pred:
            state_token = self.get(
                "state_token", common.mae_utils.Token, "state", self.embed_dim
            )(x)
            state_pos_embed = state_token
            pos_embed = tf.concat([pos_embed, state_pos_embed], axis=1)

        mae_cls_token = self.get(
            "mae_cls_token", common.mae_utils.Token, "mae_cls", self.embed_dim
        )(x)
        mae_cls_pos_embed = mae_cls_token
        pos_embed = tf.concat([pos_embed, mae_cls_pos_embed], axis=1)

        return pos_embed


class ViTDecoder(common.Module):
    def __init__(
        self,
        img_h_size,
        img_w_size,
        patch_size,
        in_chans=3,
        embed_dim=512,
        depth=8,
        num_heads=16,
        mlp_ratio=4.0,
        norm_layer="layer_norm_eps_1e-6",
        state_pred=False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.img_h_size, self.img_w_size = img_h_size, img_w_size
        self.in_chans = in_chans
        self.num_patches = (
            (img_h_size // patch_size) * (img_w_size // patch_size)
            + 1
            + int(state_pred)
        )
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self.state_pred = state_pred
        self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

        self.construct_pos_embed()

    def construct_pos_embed(self):
        ncams = int(self.img_w_size // self.img_h_size)
        each_view_w = self.img_w_size // ncams
        each_view_h = self.img_h_size
        self.ncams = ncams

        self.pos_embed = tf.constant(
            common.mae_utils.get_2d_sincos_pos_embed(
                self.embed_dim,
                int(each_view_h // self.patch_size),
                int(each_view_w // self.patch_size),
            )[None],
            name="pos_embed",
            dtype=tf.float32,
        )

    def forward_decoder(self, x):
        # embed tokens
        x = self._cast(x)
        x = self.get(
            "decoder_embed",
            tfkl.Dense,
            self.embed_dim,
        )(x)

        mask_token = self.get(
            "mask_token", common.mae_utils.Token, "mask", self.embed_dim
        )(x)
        mask_tokens = self._cast(
            tf.tile(
                mask_token,
                [x.shape[0], self.num_patches, 1],
            )
        )
        x = tf.concat([x[:, :1, :], mask_tokens], axis=1)  # append cls token

        # add pos embed
        decoder_pos_embed = self.get_pos_embed(x)
        x = x + tf.repeat(self._cast(decoder_pos_embed), repeats=x.shape[0], axis=0)

        # apply Transformer blocks
        for j in range(self.depth):
            x = self.get(
                f"vit_decoder_block_{j}",
                ViTBlock,
                embed_dim=self.embed_dim,
                nb_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.0,
                norm_layer=self.norm_layer,
                act_layer="gelu",
            )(x)
        x = self.get("vit_decoder_norm", norm_layer_factory(self.norm_layer))(x)

        # predictor projection
        x = self.get(
            "vit_decoder_pred", tfkl.Dense, self.patch_size ** 2 * self.in_chans
        )(x)

        # remove cls token
        x = x[:, 1:, :]
        return x

    def get_pos_embed(self, x):
        cls_token = self.get(
            "cls_token", common.mae_utils.Token, "cls", self.embed_dim
        )(x)
        cls_pos_embed = cls_token

        img_pos_embed = tf.concat([self.pos_embed] * self.ncams, axis=1)

        img_tokens = []
        for i in range(self.ncams):
            token = self.get(
                f"token_{i}",
                common.mae_utils.Token,
                f"{i}",
                self.embed_dim,
            )(x)
            token = tf.tile(
                token,
                [1, self.pos_embed.shape[1], 1],
            )
            img_tokens.append(token)
        img_token = tf.concat(img_tokens, axis=1)
        img_pos_embed = img_pos_embed + img_token

        pos_embed = tf.concat([cls_pos_embed, img_pos_embed], axis=1)

        if self.state_pred:
            state_token = self.get(
                "state_token", common.mae_utils.Token, "state", self.embed_dim
            )(x)
            state_pos_embed = state_token
            pos_embed = tf.concat([pos_embed, state_pos_embed], axis=1)

        mae_cls_token = self.get(
            "mae_cls_token", common.mae_utils.Token, "mae_cls", self.embed_dim
        )(x)
        mae_cls_pos_embed = mae_cls_token
        pos_embed = tf.concat([pos_embed, mae_cls_pos_embed], axis=1)

        return pos_embed


def mae_factory(
    img_h_size,
    img_w_size,
    ncams,
    patch_size,
    embed_dim,
    depth,
    num_heads,
    decoder_embed_dim,
    decoder_depth,
    decoder_num_heads,
    reward_pred=True,
    in_chans=3,
    early_conv=False,
    state_pred=False,
    state_dim=4,
    view_masking=False,
    control_input="front_wrist",
):
    encoder = MaskedViTEncoder(
        img_h_size=img_h_size,
        img_w_size=img_w_size,
        ncams=ncams,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        norm_layer="layer_norm_eps_1e-6",
        early_conv=early_conv,
        state_pred=state_pred,
        view_masking=view_masking,
        control_input=control_input,
    )

    decoder = MaskedViTDecoder(
        img_h_size=img_h_size,
        img_w_size=img_w_size,
        ncams=ncams,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=decoder_embed_dim,
        depth=decoder_depth,
        num_heads=decoder_num_heads,
        mlp_ratio=4.0,
        norm_layer="layer_norm_eps_1e-6",
        reward_pred=reward_pred,
        state_pred=state_pred,
        state_dim=state_dim,
        control_input=control_input,
    )
    return encoder, decoder


def flat_vit_factory(
    img_h_size,
    img_w_size,
    patch_size,
    embed_dim,
    depth,
    num_heads,
    decoder_embed_dim,
    decoder_depth,
    decoder_num_heads,
    in_chans=3,
    state_pred=False,
):
    encoder = ViTEncoder(
        img_h_size=img_h_size,
        img_w_size=img_w_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        norm_layer="layer_norm_eps_1e-6",
        state_pred=state_pred,
    )
    decoder = ViTDecoder(
        img_h_size=img_h_size,
        img_w_size=img_w_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=decoder_embed_dim,
        depth=decoder_depth,
        num_heads=decoder_num_heads,
        mlp_ratio=4.0,
        norm_layer="layer_norm_eps_1e-6",
        state_pred=state_pred,
    )
    return encoder, decoder
