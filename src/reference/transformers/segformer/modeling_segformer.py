import torch
from torch import nn
from transformers.activations import ACT2FN


def init_random_2d_freqs(dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
  freqs_x = []  # (num_heads, dim//4)
  freqs_y = []  # (num_heads, dim//4)
  mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))  # (dim//4)
  for _ in range(num_heads):
    angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)  # (1)
    fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi / 2 + angles)], dim=-1)  # (dim//2)
    fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi / 2 + angles)], dim=-1)
    freqs_x.append(fx)
    freqs_y.append(fy)
  freqs_x = torch.stack(freqs_x, dim=0)
  freqs_y = torch.stack(freqs_y, dim=0)
  freqs = torch.stack([freqs_x, freqs_y], dim=0)
  return freqs  # (2, num_heads, dim // 2)


def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
  """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

  Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
  however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
  See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
  layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
  argument.
  """
  if drop_prob == 0.0 or not training:
    return input
  keep_prob = 1 - drop_prob
  shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
  random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
  random_tensor.floor_()  # binarize
  output = input.div(keep_prob) * random_tensor
  return output


def compute_mixed_cis(freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor, num_heads: int):
  N = t_x.shape[0]
  depth = freqs.shape[1]  # (2, len(self.blocks), C//2)[1]
  # No float 16 for this range
  with torch.amp.autocast(device_type="cuda", enabled=False):
    freqs_x = (
      (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2))
      # (N, depth, C//2)
      .view(depth, N, num_heads, -1)  # (depth, N, num_heads, C_per_head//2)
      .permute(0, 2, 1, 3)
    )
    freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2)).view(depth, N, num_heads, -1).permute(0, 2, 1, 3)
    freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)

  return freqs_cis  # (depth, num_heads, N, dim//2)


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 100.0):
  freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
  freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

  t_x, t_y = init_t_xy(end_x, end_y)
  freqs_x = torch.outer(t_x, freqs_x)
  freqs_y = torch.outer(t_y, freqs_y)
  freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
  freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
  return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def init_t_xy(end_x: int, end_y: int):
  t = torch.arange(end_x * end_y, dtype=torch.float32)
  t_x = (t % end_x).float()
  t_y = torch.div(t, end_x, rounding_mode="floor").float()
  return t_x, t_y


class SegformerSelfOutput(nn.Module):
  def __init__(self, config, hidden_size):
    super().__init__()
    self.dense = nn.Linear(hidden_size, hidden_size)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)

  def forward(self, hidden_states, input_tensor):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.dropout(hidden_states)
    return hidden_states


class SegformerDropPath(nn.Module):
  """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

  def __init__(self, drop_prob: float | None = None) -> None:
    super().__init__()
    self.drop_prob = drop_prob

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    return drop_path(hidden_states, self.drop_prob, self.training)

  def extra_repr(self) -> str:
    return "p={}".format(self.drop_prob)


class SegformerOverlapPatchEmbeddings(nn.Module):
  """Construct the overlapping patch embeddings."""

  def __init__(self, patch_size, stride, num_channels, hidden_size):
    super().__init__()
    self.proj = nn.Conv2d(
      num_channels,
      hidden_size,
      kernel_size=patch_size,
      stride=stride,
      padding=patch_size // 2,
    )

    self.layer_norm = nn.LayerNorm(hidden_size)

  def forward(self, pixel_values):
    embeddings = self.proj(pixel_values)
    _, _, height, width = embeddings.shape
    # (batch_size, num_channels, height, width) -> (batch_size, num_channels, height*width) -> (batch_size, height*width, num_channels)
    # this can be fed to a Transformer layer
    embeddings = embeddings.flatten(2).transpose(1, 2)
    embeddings = self.layer_norm(embeddings)
    return embeddings, height, width


class SegformerDWConv(nn.Module):
  def __init__(self, dim=768):
    super().__init__()
    self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

  def forward(self, hidden_states, height, width):
    batch_size, seq_len, num_channels = hidden_states.shape
    hidden_states = hidden_states.transpose(1, 2).view(batch_size, num_channels, height, width)
    hidden_states = self.dwconv(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2)

    return hidden_states


class SegformerMixFFN(nn.Module):
  def __init__(self, config, in_features, hidden_features=None, out_features=None):
    super().__init__()
    out_features = out_features or in_features
    self.dense1 = nn.Linear(in_features, hidden_features)
    self.dwconv = SegformerDWConv(hidden_features)
    if isinstance(config.hidden_act, str):
      self.intermediate_act_fn = ACT2FN[config.hidden_act]
    else:
      self.intermediate_act_fn = config.hidden_act
    self.dense2 = nn.Linear(hidden_features, out_features)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)

  def forward(self, hidden_states, height, width):
    hidden_states = self.dense1(hidden_states)
    hidden_states = self.dwconv(hidden_states, height, width)
    hidden_states = self.intermediate_act_fn(hidden_states)
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.dense2(hidden_states)
    hidden_states = self.dropout(hidden_states)
    return hidden_states
