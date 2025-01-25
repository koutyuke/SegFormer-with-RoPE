from transformers import SegformerConfig


class SegformerWithRoPEConfig(SegformerConfig):
  """The configuration class of a Segformer model with a RoPE module.

  Args:
      SegformerConfig (_type_): _description_

  """

  model_type = "segformer-with-rope"

  def __init__(self, rope_theta: float = 100.0, image_size: int = 512, *, rope_mixed: bool = False, **kwargs):
    super().__init__(**kwargs)
    self.rope_theta = rope_theta
    self.image_size = image_size
    self.rope_mixed = rope_mixed
