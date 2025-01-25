import torch


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
