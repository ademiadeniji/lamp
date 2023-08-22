import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.keras import mixed_precision as prec
import numpy as np


class Token(tf.keras.layers.Layer):
    def __init__(
        self,
        name,
        embed_dim,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._name = name
        self.embed_dim = embed_dim
        self.mask_token = None

    def build(self, input_shape):
        self.mask_token = self.add_weight(
            name=f"{self._name}_token",
            shape=(1, 1, self.embed_dim),
            initializer=tf.random_normal_initializer(stddev=0.02),
            trainable=True,
        )

    def call(self, x):
        return self.mask_token


def get_2d_sincos_pos_embed(embed_dim, grid_h_size, grid_w_size):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    H, W = grid_h_size, grid_w_size

    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_h_size, grid_w_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, grid_size):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    assert embed_dim % 2 == 0
    grid_t = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_t)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid[0]

    omega = np.arange(embed_dim / 2, dtype=np.float32)
    omega /= embed_dim / 2
    omega = 1.0 / 10000 ** omega
    pos = grid
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    pos_emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return pos_emb
