## Interpretable Models for Granger Causality Using Self-explaining Neural Networks

### Introduction
Exploratory analysis of time series data can yield a better understanding of complex dynamical systems. Granger causality is a practical framework for analysing interactions in sequential data, applied in a wide range of domains. We propose a novel framework for inferring multivariate Granger causality under nonlinear dynamics based on an extension of self-explaining neural networks. This framework is more interpretable than other neural-network-based techniques for inferring Granger causality, since in addition to relational inference, it also allows detecting signs of Granger-causal effects and inspecting their variability over time. In comprehensive experiments on simulated data, we show that our framework performs on par with several powerful baseline methods at inferring Granger causality and that it achieves better performance at inferring interaction signs. The results suggest that our framework is a viable and more interpretable alternative to sparse-input neural networks for inferring Granger causality.

*Relational inference in time series*:
<p align="center">
  <img align="middle" src="https://github.com/i6092467/GVAR/blob/master/images/scheme_panel_1.png" alt="relational inference" width="500"/>
</p>

*In addition to structure, our approach allows inferring Granger-causal effect signs*:
<p align="center">
  <img align="middle" src="https://github.com/i6092467/GVAR/blob/master/images/scheme_panel_2.png" alt="interpretable relational inference" width="500"/>
</p>

*The overall summary of the proposed framework*:
<p align="center">
  <img align="middle" src="https://github.com/i6092467/GVAR/blob/master/images/GC_ICLR_Thumbnail.png" alt="inference framework summary" width="500"/>
</p>

This project iplements an autoregressive model for inferring Granger causality based on self-explaining neural networks – **generalised vector autoregression (GVAR)**. The description of the model, inference framework, experiments, comparison to baselines, and ablations can be found in the [ICLR 2021 paper](https://openreview.net/forum?id=DEa4JdMWRHp). A short explanation of the method is provided in [this talk](https://slideslive.com/38941433/interpretable-models-for-granger-causality-using-selfexplaining-neural-networks). The poster is available [here](https://github.com/i6092467/GVAR/blob/master/images/GC_ICLR_Poster.png).

### Requirements
All the libraries required are in the conda environment `environment.yml`. To install it, follow the instructions below:
```
conda env create -f environment.yml   # install dependencies
conda activate SENGC                  # activate environment
```

Note, that the current implementation of GVAR requires a GPU supported by CUDA 10.1.0.

### Experiments
`/bin` folder contains shell scripts for the three simulation experiments described in the paper (all arguments are given in the scripts):
- **Lorenz 96**: `run_grid_search_lorenz96`
- **fMRI**: `run_grid_search_fMRI`
- **Lotka–Volterra**: `run_grid_search_lotka_volterra`

The data used to generate results in the [paper](https://openreview.net/forum?id=DEa4JdMWRHp) are stored in the folder `datasets/experiment_data`.

Further details are documented within the code.

### Acknowledgements

- Simulated fMRI time series data: https://www.fmrib.ox.ac.uk/datasets/netsim/.
- Lotka–Volterra system simulation is based on the code from https://github.com/smkalami/lotka-volterra-in-python.

Code for the baseline models, apart from VAR, is not included into this project and is available in the following repositories:
- cMLP and cLSTM: https://github.com/iancovert/Neural-GC
- TCDF: https://github.com/M-Nauta/TCDF
- eSRU: https://github.com/sakhanna/SRU_for_GCI

### Authors
- Ričards Marcinkevičs ([ricards.marcinkevics@inf.ethz.ch](mailto:ricards.marcinkevics@inf.ethz.ch))
- Julia E. Vogt ([julia.vogt@inf.ethz.ch](mailto:julia.vogt@inf.ethz.ch))

### References

Below are some references helpful for understanding our method:
- C. W. J. Granger. Investigating causal relations by econometric models and cross-spectral methods. *Econometrica*, 37(3):424–438, 1969.
- A. Arnold, Y. Liu, and N. Abe. Temporal causal modeling with graphical Granger methods. In *Proceedings of the 13th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, KDD ’07, pages 66–75, 2007.
- L. Song, M. Kolar, and E. Xing. Time-varying dynamic Bayesian networks. In *Advances in Neural Information Processing Systems 22*, pp. 1732–1740. Curran Associates, Inc., 2009.
- A. Tank, I. Covert, N. Foti, A. Shojaie, and E. Fox. Neural Granger causality for nonlinear time series, 2018. arXiv:1802.05842.
- D. Alvarez-Melis and T. Jaakkola. Towards robust interpretability with self-explaining neural networks. In *Advances in Neural Information Processing Systems 31*, pp. 7775–7784. Curran Associates, Inc., 2018.

### Citation

```
@inproceedings{Marcinkevics2021,
  title={Interpretable Models for Granger Causality Using Self-explaining Neural Networks},
  author={Ri{\v{c}}ards Marcinkevi{\v{c}}s and Julia E Vogt},
  booktitle={International Conference on Learning Representations},
  year={2021},
}
```
