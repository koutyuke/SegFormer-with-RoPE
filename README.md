<div align="center">

# SegFormer with Rotary Position Embedding

**[@koutyuke](https://github.com/koutyuke)**

![Static Badge](https://img.shields.io/badge/license-Apache--2.0-blue)
![Static Badge](https://img.shields.io/badge/Google--Colab-available-brightgreen)

### New proposal for _"RoPE Efficient Self-Attention"_

</div>

## What's this project?

This project is an adaptation of RoPE modified for vision to Segformer.

Specifically, RoPE is added to Efficient-Self-attention used inside SegFormer.

This enables more robust position coding.

```txt
Details of Transformer Block

      +-----------------+
      |     input       |
      +-----------------+
              |
              |     +--------+    ↑
              + ←-- |  RoPE  |    |
              |     +--------+    |
              ↓                   | RoPE Efficient Self-Attention
      +-----------------+         |
      |  Efficient MSA  |         |
      +-----------------+         ↓
              ↓
      +-----------------+
      |     Mix-FFN     |
      +-----------------+
              ↓
      +-----------------+
      |     output      |
      +-----------------+

```

## What's inside?

```txt
/src
├── colab
│   └── ...
├── reference
│   ├── rope_vit
│   │   └── ...
│   └── transformers
│       └── segformer
│           └── ...
└── segformer_with_rope
    └── ...
```

- `src/...` : Main implementation
- `src/colab/...` : The Jupyter Notebook files used to tran the model
- `src/reference/repo_vit/...` : The file of code implemented in [rope-vit](https://github.com/naver-ai/rope-vit)
- `src/reference/transformers/...` : The file of code implemented in [rope-vit](https://github.com/huggingface/transformers)
- `src/segformer_with_rope/...` : SegFormer with RoPE code

## Getting Started

We use `uv` to manage packages and python version. so, install uv your environments.

Install python and packages.

```sh
uv sync
```

Open `/src/colab/Segformer-with-RoPE-b0-{ axial | mixed }.ipynb` at editor

Connect `.venv/ (python)` kernel.

🎉 Run all 🎉

## Experiments

We used [ADE20K(SceneParse150)](https://huggingface.co/datasets/zhoubolei/scene_parse_150) to train our model.

### Detail

| Setting Name    | Value           |
| --------------- | --------------- |
| Train Data      | 20210 images    |
| Validation Data | 1000 images     |
| Epoch size      | 1024            |
| Batch size      | 16              |
| Learning Late   | 1e-3            |
| Warmup Ratio    | 0.1(≒100 epoch) |

### Result

| Model Name           | Size | mIoU      |
| -------------------- | ---- | --------- |
| SegFormer            | b0   | 21.83     |
| SegFOrmer+RoPE-Axial | b0   | 23.82     |
| SegFOrmer+RoPE-Mixed | b0   | **24.11** |

## License

This project is distributed under [Apache-2.0](http://www.apache.org/licenses/LICENSE-2.0).

- `src/reference/rope_vit/vit_rope.py`
- `src/reference/transformers/segformer/modeling_segformer.py`
- `src/segformer_with_rope/config.py`
- `src/segformer_with_rope/model.py`
- `src/colab/Segformer-with-RoPE-b0-axial.ipynb`
- `src/colab/Segformer-with-RoPE-b0-mixed.ipynb`

The code is the following files are derived from the code of projects [transformers](https://github.com/huggingface/transformers) and [rope-vit](https://github.com/naver-ai/rope-vit).

Both projects are distributed under the Apache-2.0 license. Therefore, using these files requires compliance with the license.

## References Project

https://github.com/naver-ai/rope-vit

https://github.com/huggingface/transformers
