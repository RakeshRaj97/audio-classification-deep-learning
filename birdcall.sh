#!/bin/bash

## script for building virtual environment for birdcall identification

module load anaconda3/5.1.0
module load gcc/6.4.0

conda create -n birdcall python=3.7
source activate birdcall

conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
pip install albumentations
pip install pydub
conda install -c conda-forge/label/gcc7 ffmpeg
pip install pretrainedmodels
pip install matplotlib
pip install seaborn
pip install Pillow
pip install scikit-learn
pip install librosa
pip install -q pydub
pip install opencv-python
pip install tqdm
pip install scikit-learn
pip install pandas
conda install ipykernel
python -m ipykernel install --user --name bird --display-name bird

source deactivate

