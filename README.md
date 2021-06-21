# variational Latent Gaussian Process (ported to JAX)

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?style=flat-square)]()
[![python 3.8](https://img.shields.io/badge/python-3.8-blue.svg?style=flat-square)]()

## Introduction

This repo contains the implementation of [variational Latent Gaussian Process (vLGP)](https://doi.org/10.1162/NECO_a_00953) 
([arXiv](https://arxiv.org/abs/1604.03053)) 
([video](https://youtu.be/CrY5AfNH1ik)) by 
Yuan Zhao ([yuan.zhao@stonybrook.edu](yuan.zhao@stonybrook.edu])) and 
Il Memming Park ([memming.park@stonybrook.edu](memming.park@stonybrook.edu)).
It has been developed with the goal of recovering low-dimensional dynamics from neural population recordings. 

## Installation

```bash
pip install git+https://github.com/catniplab/vlgp.git
```

## Get started

Try the [example](script/example.py).

### Data structure
- `Trial`
  - `y` spike train
  - `x` regressors
  - `z` latent factors
  - `v` posterior variances
  - `w` diagonals of W matrices  
- `Session`
  - `trials` list of `Trial`s
  - `binsize` 
  - `unit` *str*, unit of time
- `Params`
  - `n_factors` number of latent factors
  - `K` kernel matrices
  - `logdet` log determinant of `K`s  
- `Inference`
  - `session`
  - `params`
  - `kernel` kernel functions
    
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

## Changes

2021

- port to JAX

2018

- New uniform data structure
- Support trials of unequal duration
- Faster
- Use NumPy data format

2017

- New ```fit``` function now only requires observation and the number of latent.
- Save snapshots if *path* is passed to ```fit```.
- You can access the iterations via *callback*.