# Pytorch_MuseGAN_HSEstudy
Study project for the HSE "Master of Data Science" program

The project is based on paper [MuseGAN: Multi-track Sequential Generative Adversarial Networks for Symbolic Music Generation and Accompaniment](https://arxiv.org/abs/1709.06298) and GitHub [repository](https://github.com/akanametov/musegan)

## Example 
The example of model usage is shown in [notebook](MuseGAN_example.ipynb). It is recommended to open in jupyter notebook to listen inlne sound player. 

The example of generated melody could be found in [notebook](MuseGAN_example.ipynb) and in the file [fake_midi.midi](fake_midi.midi)

## The model implementation
The model implementation are under [musegan](musegan) folder. To train model it is needed to initialize MuseGAN model and run it's train method 
- [dataset.py](musegan/dataset.py) contains MidiDataset class and postprocessind function to trasform tensors to midi format
- [generator.py](musegan/generator.py) contains all submodels needed to implement MuseGAN Generator network and Generator class itself.
- [critic.py](musegan/critic.py) contains implementaion of Critic for MuseGAN
- [loss.py](musegan/loss.py) contains Wasserstein loss and gradient penalty implementaion
- [musegan.py](musegan/musegan.py) contains resulting MuseGAN model with train method. 

## Packages requirements
- [requirements.txt](requirements.txt)

## Dataset
In that project the dataset of 229 chorales of Bach for four tracks is used.
The dataset could be found [here](data/Jsb16thSeparated.npz)
