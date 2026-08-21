# FADLinear and LWPT-Focus

[![Paper](https://img.shields.io/badge/Ocean%20Engineering-10.1016%2Fj.oceaneng.2025.122693-blue)](https://doi.org/10.1016/j.oceaneng.2025.122693)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Research code accompanying the paper **“Adaptive processing of non-stationary ship lubrication signals under sporadic impulsive noise”**, published in *Ocean Engineering* (2025).

## Release scope / 发布范围

> **This repository provides only the FADLinear model and the LWPT-Focus denoising training strategy used in the paper. It is not a complete standalone forecasting framework.**

> **本仓库仅提供论文中的 FADLinear 模型与 LWPT-Focus 降噪训练策略，并非完整、可独立运行的预测框架。**

For the remaining infrastructure, please use the upstream projects:

- Forecasting framework, data loaders, experiment runner, and baseline implementation: [DLinear / LTSF-Linear](https://github.com/cure-lab/LTSF-Linear)
- Learnable wavelet transform implementation required by `LWPT_main.py`: [Learnable Wavelet Transform](https://github.com/FrusqueGaetan/Learnable-Wavelet-Transform)

The scripts retain the paths and settings used in the original research environment. Update local paths and integrate the released modules into the upstream repositories before running experiments.

## Method overview

The paper addresses non-stationary ship lubrication signals affected by sporadic impulsive noise through a three-stage workflow:

1. **Adaptive signal extension** preserves boundary continuity before wavelet processing.
2. **LWPT-Focus** learns a wavelet-packet denoiser while computing reconstruction loss only on the original, non-padded signal region. A sparsity penalty is applied to the learned wavelet coefficients.
3. **FADLinear** combines frequency-guided adaptive decomposition with channel-wise spectral attention and lightweight linear forecasting heads.

This release focuses on stages 2 and 3. The full methodological description, equations, ablations, and experimental settings are provided in the paper.

## Repository contents

| Path | Description |
| --- | --- |
| `FADLinear.py` | FADLinear model for integration with the DLinear/LTSF-Linear forecasting framework. |
| `LWPT_main.py` | LWPT-Focus training and ablation script. Requires the upstream `Code/` modules from Learnable Wavelet Transform. |
| `singnalpadding.py` | Legacy signal-extension experiment script retained as-is for traceability; it is outside the supported release scope. |
| `voyage_Galveston_to_South_Korea_2022-09-06_2022-11-04.csv` | Voyage data used in the study. |
| `proposed_method_v4_always_AR_C0C1_enforced_padded_data_local_metrics.csv` | Padded signal artifact used by the denoising stage. |
| `final_metrics_summary_v4_always_AR_local_metrics.csv` | Signal-extension evaluation summary. |
| `LW3_96_336_FADLinear_*` | Retained experiment output/checkpoint artifact. |

Existing data and result files are retained for research traceability; they do not replace the upstream training framework.

## Installation and integration

Create an environment with Python 3.10 or a compatible version, then install the local dependencies:

```bash
pip install -r requirements.txt
```

### FADLinear

1. Clone [LTSF-Linear](https://github.com/cure-lab/LTSF-Linear) and follow its environment and dataset instructions.
2. Add `FADLinear.py` to the upstream model directory.
3. Register `FADLinear` in the upstream experiment runner in the same way as the other model modules.
4. Configure `seq_len`, `pred_len`, `enc_in`, and `individual` through the upstream configuration object.

The model expects an input tensor shaped `[batch, input_length, channels]` and returns `[batch, prediction_length, channels]`.

### LWPT-Focus

1. Clone [Learnable Wavelet Transform](https://github.com/FrusqueGaetan/Learnable-Wavelet-Transform).
2. Place `LWPT_main.py` at the project root, where the upstream `Code/` directory is available.
3. Update `file_path` near the bottom of `LWPT_main.py` to point to the padded CSV file.
4. Run `python LWPT_main.py`.

The default research configuration evaluates Focus/NoFocus and SmoothL1/L1 combinations. Training settings should be adjusted for the available hardware and data.

## Citation

If this code or dataset supports your work, please cite:

```bibtex
@article{zhang2025adaptive,
  title   = {Adaptive processing of non-stationary ship lubrication signals under sporadic impulsive noise},
  author  = {Zhang, Meng and Liu, Jilong and Han, Bing and Dong, Shengli and Cui, Tong and Ren, Yan},
  journal = {Ocean Engineering},
  volume  = {342},
  pages   = {122693},
  year    = {2025},
  doi     = {10.1016/j.oceaneng.2025.122693}
}
```

GitHub also exposes the same metadata through [`CITATION.cff`](CITATION.cff).

## Contribution and maintenance

The published CRediT statement identifies **Jilong Liu** with conceptualization, methodology, software, formal analysis, validation, visualization, original-draft preparation, and review/editing contributions.

Jilong Liu maintains this repository and is currently pursuing a Ph.D. at Northeastern University, China.

Contact: [liujilong@mails.neu.edu.cn](mailto:liujilong@mails.neu.edu.cn)

## License

This repository is distributed under the terms in [LICENSE](LICENSE). Please cite the paper when reusing the research code or data.

