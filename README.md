# variational Latent Gaussian Process with JAX

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?style=flat-square)]()
[![python 3.8](https://img.shields.io/badge/python-3.8-blue.svg?style=flat-square)]()

## Introduction

This repo contains a JAX implementation of [variational Latent Gaussian Process (vLGP)](https://doi.org/10.1162/NECO_a_00953) 
([arXiv](https://arxiv.org/abs/1604.03053)) 
([video](https://youtu.be/CrY5AfNH1ik)) by 
Yuan Zhao ([yuan.zhao@stonybrook.edu](yuan.zhao@stonybrook.edu)) and 
Il Memming Park ([memming.park@stonybrook.edu](memming.park@stonybrook.edu)).
It has been developed with the goal of recovering low-dimensional dynamics from neural population recordings. 

## Installation

```bash
pip install git+https://github.com/yuanz271/vlgpax.git
```

## Get started

Learn how to use it in the [example](script/example.py).

### Data structure
- `Trial` A single trial
  - `y` spike train
  - `x` regressors
  - `t` timing of bins
  - `z` posterior mean
  - `v` posterior variances
- `Session` A container of trials
  - `trials` list of `Trial`
  - `binsize` binwidth if evenly spaced
- `Params` Parameters
  - `n_factors` number of latent factors
  - `C` loading matrix, (n_factors + n_regressors, n_neurons)
  - `K` kernel matrices

### Training
```python
from vlgpax import Session, vi
from vlgpax.kernel import RBF

binsize = 1e-2
session = Session(binsize)
# Add trials to the session
# session.add_trial(tid=1, y=y)  # ID, spike train, ...
# ...

# Kernel function k(x, y) = scale * exp(-0.5 ||x/lengthscale - y/lengthsale||^2)
kernel = RBF(scale=1., lengthscale=100.)  # lengthscale has the same unit as that of binsize

session, params = vi.fit(session, n_factors=2, kernel=kernel)
```

## Citation
```
@Article{Zhao2017,
  author    = {Yuan Zhao and Il Memming Park},
  title     = {Variational Latent Gaussian Process for Recovering Single-Trial Dynamics from Population Spike Trains},
  journal   = {Neural Computation},
  year      = {2017},
  volume    = {29},
  number    = {5},
  pages     = {1293--1316},
  month     = {may},
  doi       = {10.1162/neco_a_00953},
  publisher = {{MIT} Press - Journals},
}
```
