import math
from functools import partial

import torch
import torch.nn.functional as F
from icecream import ic
from torch import nn
from transformers import SegformerForSemanticSegmentation, SegformerModel
from transformers.modeling_outputs import BaseModelOutput

from reference.rope_vit.vit_rope import (
  compute_axial_cis,
  compute_mixed_cis,
  init_random_2d_freqs,
  init_t_xy,
)
from reference.transformers.segformer.modeling_segformer import (
  SegformerDropPath,
  SegformerMixFFN,
  SegformerOverlapPatchEmbeddings,
  SegformerSelfOutput,
)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
  # (B, num_heads, N, C_per_head // 2)
  # freqs_cis (N, C_per_head) or (num_heads, N, C_per_head//2)

  ndim = x.ndim

  if ndim <= 1:
    raise ValueError("ndim must be greater than 1")

  if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
  elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
    shape = [d if i >= ndim - 3 else 1 for i, d in enumerate(x.shape)]
  else:
    msg = f"Invalid shape for `freqs_cis {freqs_cis.shape}` and `x {x.shape}`"
    raise ValueError(msg)

  return freqs_cis.view(*shape)  # (1, 1 or num_heads, N, C_per_head//2)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, sr_ratio: int, freqs_cis: torch.Tensor, height, width):
  # freqs_cis (N, C_per_head) or (num_heads, N, C_per_head//2)

  xq_ = torch.view_as_complex(
    xq.float().reshape(*xq.shape[:-1], -1, 2)
  )  # (B, num_heads, N, C_per_head) -> (B, num_heads, N, C_per_head//2)
  xk_ = torch.view_as_complex(
    xk.float().reshape(*xk.shape[:-1], -1, 2)
  )  # (B, num_heads, H*W/sr^2, C_per_head) -> (B, num_heads, H*W/sr^2, C_per_head//2)

  xq_freqs_cis = reshape_for_broadcast(freqs_cis, xq_)  # (1, 1 or num_heads, N, C_per_head//2)
  xk_freqs_cis = xq_freqs_cis  # (1, 1 or num_heads, N, C_per_head//2)

  if sr_ratio > 1:
    f_b, f_head, _, f_c = xq_freqs_cis.shape

    xk_freqs_cis = xk_freqs_cis.view(f_b, f_head, height, width, f_c).permute(
      0, 1, 4, 2, 3
    )  # (1, 1 or num_heads, C_per_head//2, H, W)

    xk_freqs_cis = xk_freqs_cis.view(f_b * f_head, f_c, height, width)
    # (1, 1 or num_heads, C_per_head//2, H, W)

    # === pooling ===

    real_part = xk_freqs_cis.real
    imag_part = xk_freqs_cis.imag

    pooled_real = F.avg_pool2d(real_part, kernel_size=sr_ratio, stride=sr_ratio)
    pooled_imag = F.avg_pool2d(imag_part, kernel_size=sr_ratio, stride=sr_ratio)

    xk_freqs_cis = torch.complex(pooled_real, pooled_imag)

    # === end of pooling ===

    xk_freqs_cis = xk_freqs_cis.view(f_b, f_head, f_c, height // sr_ratio, width // sr_ratio)

    xk_freqs_cis = xk_freqs_cis.permute(0, 1, 3, 4, 2).reshape(
      f_b, f_head, -1, f_c
    )  # (1, 1 or num_heads, H*W/sr^2, C_per_head//2)

  xq_out = torch.view_as_real(xq_ * xq_freqs_cis).flatten(
    3
  )  # (B, num_heads, N, C_per_head//2) -> (B, num_heads, N, C_per_head//2, 2) -> (B, num_heads, N, C_per_head)
  xk_out = torch.view_as_real(
    xk_ * xk_freqs_cis
  ).flatten(
    3
  )  # (B, num_heads, H*W/sr^2, C_per_head//2) -> (B, num_heads, N/sr^2, C_per_head//2, 2) -> (B, num_heads, N/sr^2, C_per_head)
  return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(
    xk.device
  )  # (B, num_heads, N, C_per_head), (B, num_heads, N/sr^2, C_per_head)


class SegformerWithRoPEEfficientSelfAttention(nn.Module):
  """Efficient Self Attention with RoPE module."""

  def __init__(self, config, hidden_size, num_attention_heads, sequence_reduction_ratio):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_attention_heads = num_attention_heads

    if self.hidden_size % self.num_attention_heads != 0:
      msg = (
        f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
        f"heads ({self.num_attention_heads})"
      )
      raise ValueError(msg)

    self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size  # == hidden_size

    self.query = nn.Linear(self.hidden_size, self.all_head_size)  # (B, H*W, C) -> (B, H*W, C)
    self.key = nn.Linear(self.hidden_size, self.all_head_size)  # (B, H*W, C) -> (B, H*W, C)
    self.value = nn.Linear(self.hidden_size, self.all_head_size)  # (B, H*W, C) -> (B, H*W, C)

    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    self.sr_ratio = sequence_reduction_ratio
    if sequence_reduction_ratio > 1:
      self.sr = nn.Conv2d(
        hidden_size, hidden_size, kernel_size=sequence_reduction_ratio, stride=sequence_reduction_ratio
      )
      self.layer_norm = nn.LayerNorm(hidden_size)

  def transpose_for_scores(self, hidden_states):
    new_shape = hidden_states.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
    hidden_states = hidden_states.view(new_shape)
    return hidden_states.permute(0, 2, 1, 3)

  def forward(
    self,
    hidden_states,
    height,
    width,
    freqs_cis,  # added (N, C_per_head//2) or (num_heads, N, C_per_head//2)
    output_attentions=False,
  ):
    query_layer = self.transpose_for_scores(
      self.query(hidden_states)
    )  # (B, H*W, C) -> (B, H*W, C) -> (B, num_attention_heads, H*W, C_per_head)

    if self.sr_ratio > 1:
      batch_size, seq_len, num_channels = hidden_states.shape  # (B, H*W, C)
      # Reshape to (batch_size, num_channels, height, width)
      hidden_states = hidden_states.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)  # (B, C, H, W)
      # Apply sequence reduction
      hidden_states = self.sr(hidden_states)  # (B, C, H, W) -> (B, C, H/sr, W/sr)
      # Reshape back to (batch_size, seq_len, num_channels)
      hidden_states = hidden_states.reshape(batch_size, num_channels, -1).permute(
        0, 2, 1
      )  # (B, C, H/sr*W/sr) -> (B, H/sr*W/sr, C) = (B, H*W/sr^2, C)
      hidden_states = self.layer_norm(hidden_states)

    key_layer = self.transpose_for_scores(
      self.key(hidden_states)
    )  # (B, H*W, C) -> (B, H*W, C) -> (B, num_attention_heads, H*W, C_per_head)
    value_layer = self.transpose_for_scores(
      self.value(hidden_states)
    )  # (B, H*W, C) -> (B, H*W, C) -> (B, num_attention_heads, H*W, C_per_head)

    # === Apply RoPE ===

    # input: xq, xk, sr_ratio, freqs_cis
    #   query_layer: (B, num_attention_heads, H*W, C_per_head)
    #   key_layer: (B, num_attention_heads, H*W/sr^2, C_per_head)
    # output: (B, num_heads, N, C_per_head), (B, num_heads, N/sr^2, C_per_head)
    query_layer, key_layer = apply_rotary_emb(
      query_layer,
      key_layer,
      self.sr_ratio,
      freqs_cis,
      height,
      width,
    )

    # === End of RoPE ===

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

    attention_scores = attention_scores / math.sqrt(self.attention_head_size)

    # Normalize the attention scores to probabilities.
    attention_probs = nn.functional.softmax(attention_scores, dim=-1)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = self.dropout(attention_probs)

    context_layer = torch.matmul(attention_probs, value_layer)

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(new_context_layer_shape)

    outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

    return outputs


class SegformerWithRoPEAttention(nn.Module):
  def __init__(self, config, hidden_size, num_attention_heads, sequence_reduction_ratio):
    super().__init__()
    self.self = SegformerWithRoPEEfficientSelfAttention(
      config=config,
      hidden_size=hidden_size,
      num_attention_heads=num_attention_heads,
      sequence_reduction_ratio=sequence_reduction_ratio,
    )
    self.output = SegformerSelfOutput(config, hidden_size=hidden_size)
    self.pruned_heads = set()

  # def prune_heads(self, heads):
  #     if len(heads) == 0:
  #         return
  #     heads, index = find_pruneable_heads_and_indices(
  #         heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
  #     )

  #     # Prune linear layers
  #     self.self.query = prune_linear_layer(self.self.query, index)
  #     self.self.key = prune_linear_layer(self.self.key, index)
  #     self.self.value = prune_linear_layer(self.self.value, index)
  #     self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

  #     # Update hyper params and store pruned heads
  #     self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
  #     self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
  #     self.pruned_heads = self.pruned_heads.union(heads)

  def forward(
    self,
    hidden_states,
    height,
    width,
    freqs_cis,  # added (N, C_per_head//2) or (num_heads, N, C_per_head//2)
    output_attentions=False,
  ):
    self_outputs = self.self(hidden_states, height, width, freqs_cis, output_attentions)

    attention_output = self.output(self_outputs[0], hidden_states)
    outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
    return outputs


class SegformerWithRoPELayer(nn.Module):
  """This corresponds to the Block class in the original implementation."""

  def __init__(self, config, hidden_size, num_attention_heads, drop_path, sequence_reduction_ratio, mlp_ratio):
    super().__init__()
    self.layer_norm_1 = nn.LayerNorm(hidden_size)
    self.attention = SegformerWithRoPEAttention(
      config,
      hidden_size=hidden_size,
      num_attention_heads=num_attention_heads,
      sequence_reduction_ratio=sequence_reduction_ratio,
    )
    self.drop_path = SegformerDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    self.layer_norm_2 = nn.LayerNorm(hidden_size)
    mlp_hidden_size = int(hidden_size * mlp_ratio)
    self.mlp = SegformerMixFFN(config, in_features=hidden_size, hidden_features=mlp_hidden_size)

  def forward(
    self,
    hidden_states,
    height,
    width,
    freqs_cis,  # added (N, C_per_head//2) or (num_heads, N, C_per_head//2)
    output_attentions=False,
  ):
    self_attention_outputs = self.attention(
      self.layer_norm_1(hidden_states),  # in Segformer, layernorm is applied before self-attention
      height,
      width,
      freqs_cis,
      output_attentions=output_attentions,
    )

    attention_output = self_attention_outputs[0]
    outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

    # first residual connection (with stochastic depth)
    attention_output = self.drop_path(attention_output)
    hidden_states = attention_output + hidden_states

    mlp_output = self.mlp(self.layer_norm_2(hidden_states), height, width)

    # second residual connection (with stochastic depth)
    mlp_output = self.drop_path(mlp_output)
    layer_output = mlp_output + hidden_states

    outputs = (layer_output,) + outputs

    return outputs


class SegformerWithRoPEEncoder(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    # stochastic depth decay rule
    drop_path_decays = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]

    # patch embeddings
    embeddings = []
    for i in range(config.num_encoder_blocks):  # 4 layers
      embeddings.append(
        SegformerOverlapPatchEmbeddings(
          patch_size=config.patch_sizes[i],
          stride=config.strides[i],
          num_channels=config.num_channels if i == 0 else config.hidden_sizes[i - 1],
          hidden_size=config.hidden_sizes[i],
        )
      )
    self.patch_embeddings = nn.ModuleList(embeddings)

    # Transformer blocks
    blocks = []

    cur = 0
    for i in range(config.num_encoder_blocks):  # 4 layers
      # each block consists of layers
      layers = []
      if i != 0:
        cur += config.depths[i - 1]  # b0 [2, 2, 2, 2]
      for j in range(config.depths[i]):
        layers.append(
          SegformerWithRoPELayer(
            config,
            hidden_size=config.hidden_sizes[i],
            num_attention_heads=config.num_attention_heads[i],
            drop_path=drop_path_decays[cur + j],
            sequence_reduction_ratio=config.sr_ratios[i],
            mlp_ratio=config.mlp_ratios[i],
          )
        )
      blocks.append(nn.ModuleList(layers))

    # axial: (encoder_blocks, H*W, dim_per_head//2 )
    # mixed: (encoder_blocks, transformer_blocks, 2, num_blocks, num_heads, dim//2)
    freqs = []
    embedding_size = config.image_size
    self.compute_cis = []

    # === compute cis ===
    for i in range(config.num_encoder_blocks):  # 4
      embedding_size = (
        (embedding_size + 2 * (config.patch_sizes[i] // 2) - config.patch_sizes[i]) // config.strides[i]
      ) + 1

      if self.config.rope_mixed:
        compute_cis = partial(compute_mixed_cis, num_heads=config.num_attention_heads[i])
        self.compute_cis.append(compute_cis)

        f = []  # (blocks, 2, num_heads, C_per_heads//2)

        for _ in range(config.depths[i]):
          f.append(
            init_random_2d_freqs(
              dim=config.hidden_sizes[i] // config.num_attention_heads[i],
              num_heads=config.num_attention_heads[i],
              theta=config.rope_theta,
            )  # (2, num_heads, C_per_heads//2)
          )

        # (2, num_heads, C_per_heads//2)[num_blocks] -> (2, num_blocks, num_heads, C_per_heads // 2) -> (2, config.depths[i], num_blocks * num_heads * C_per_heads // (2 * config.depths[i]))
        # -> (2, config.depths[i], C//2)
        f = torch.stack(f, dim=1).view(2, config.depths[i], -1)

        freqs.append(nn.Parameter(f.clone()))

        _t_x, _t_y = init_t_xy(end_x=embedding_size, end_y=embedding_size)

        self.register_buffer(f"t_x_{i}", _t_x)
        self.register_buffer(f"t_y_{i}", _t_y)

      else:
        compute_cis = partial(
          compute_axial_cis,
          theta=self.config.rope_theta,
          dim=config.hidden_sizes[i] // config.num_attention_heads[i],
        )
        self.compute_cis.append(compute_cis)

        freqs_cis = compute_cis(end_x=embedding_size, end_y=embedding_size)  # (N, C_per_head//2)

        freqs.append(freqs_cis)

    if self.config.rope_mixed:
      self.freqs = nn.ParameterList(freqs)
    else:
      self.freqs = freqs

    # === end of compute cis ===

    self.block = nn.ModuleList(blocks)

    # Layer norms
    self.layer_norm = nn.ModuleList([nn.LayerNorm(config.hidden_sizes[i]) for i in range(config.num_encoder_blocks)])

  def forward(
    self,
    pixel_values: torch.FloatTensor,
    output_attentions: bool | None = False,
    output_hidden_states: bool | None = False,
    return_dict: bool | None = True,
  ) -> tuple | BaseModelOutput:
    all_hidden_states = () if output_hidden_states else None
    all_self_attentions = () if output_attentions else None

    batch_size = pixel_values.shape[0]

    hidden_states = pixel_values
    for idx, x in enumerate(zip(self.patch_embeddings, self.block, self.layer_norm, strict=False)):
      embedding_layer, block_layer, norm_layer = x

      # first, obtain patch embeddings
      hidden_states, height, width = embedding_layer(hidden_states)  # (B, H*W, C)

      # second, send embeddings through blocks
      if self.config.rope_mixed:
        t_x = getattr(self, f"t_x_{idx}")
        t_y = getattr(self, f"t_y_{idx}")

        compute_cis = self.compute_cis[idx]

        # freqs_cis: (2, config.depths[i], C//2) -> # (depth, num_heads, H*W, C//2)
        freqs_cis = compute_cis(freqs=self.freqs[idx], t_x=t_x, t_y=t_y)
      else:
        freqs_cis = self.freqs[idx].to(
          pixel_values.device
        )  # (encoder_blocks, H*W, dim_per_head//2 )[idx] -> (N, C_per_head//2)

      for i, blk in enumerate(block_layer):
        layer_outputs = blk(
          hidden_states, height, width, freqs_cis[i] if self.config.rope_mixed else freqs_cis, output_attentions
        )

        hidden_states = layer_outputs[0]

        if output_attentions:
          all_self_attentions = (*all_self_attentions, layer_outputs[1])

      # third, apply layer norm
      hidden_states = norm_layer(hidden_states)

      # fourth, optionally reshape back to (batch_size, num_channels, height, width)
      if idx != len(self.patch_embeddings) - 1 or (
        idx == len(self.patch_embeddings) - 1 and self.config.reshape_last_stage
      ):
        hidden_states = hidden_states.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
      if output_hidden_states:
        all_hidden_states = (*all_hidden_states, hidden_states)

    if not return_dict:
      return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
    return BaseModelOutput(
      last_hidden_state=hidden_states,
      hidden_states=all_hidden_states,
      attentions=all_self_attentions,
    )


class SegformerWithRoPEModel(SegformerModel):
  def __init__(self, config):
    super().__init__(config)
    self.encoder = SegformerWithRoPEEncoder(config)
    self.post_init()


class SegformerWithRoPEForSemanticSegmentation(SegformerForSemanticSegmentation):
  def __init__(self, config):
    super().__init__(config)
    self.segformer = SegformerWithRoPEModel(config)
    self.post_init()
