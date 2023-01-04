#!/bin/bash

conda create -n animations
conda init bash
conda activate animations
conda install -c conda-forge manim numpy scipy matplotlib 
conda env export > dev_environment.yml
conda deactivate