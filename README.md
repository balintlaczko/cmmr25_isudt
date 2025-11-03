# Image Sonification as Unsupervised Domain Transfer
This is the code repository for our paper "Image Sonification as Unsupervised Domain Transfer", to be presented on CMMR 2025, London.

This paper introduces a novel image sonification method, framed as a domain transfer problem, in a fully unsupervised framework. It is a combination of two FactorVAE models that learn representations of image and sound datasets, and an MLP Mapper network that projects latent points from the image model's latent space to the sound model's. 

Importantly, the Mapper aims to preserve the proportional distances observed in the sound model's latent space, so that when the projected latent points are decoded as sounds, the perceived auditory differences between samples should reflect the (learned) semantic differences between their corresponding images. 

The following figure summarizes the whole system. The semi-transparent red path indicates how the system is used during inference: images -> encoded to latent points -> projected -> decoded as sounds.

<img width="2178" height="1446" alt="LSM net" src="https://github.com/user-attachments/assets/6efda2f3-a674-49d2-8e95-e38f622d15d4" />


# Install

First, it is recommended that you make an environment, for example, via conda:
```
conda create -n cmmr25_isudt python=3.12 pip
conda activate cmmr25_isudt
```
Then got into the *cmmr25_isudt* folder (where the cloned repository resides on your computer) and install the **cmmr25_isudt** local package as editable:
```
pip install -e .
```
This will also install all the dependencies.

# Evaluation

Pretrained model checkpoints are included in the *ckpt* folder. In the *scripts* folder you will find three scripts and a Max patch for the evaluation process:
- *eval_img_vae.py*: this will load the image model and render scatter plots of its latent space, color coded with the parameters used for generating the input image data (which square X and Y position),
- *eval_sinewave_vae.py*: the same process applied on the sound model and the input sound generating parameters (sinewave MIDI pitch and amplitude)
- *eval_mapper.py*: this will load the mapper (whose checkpoint actually includes pickled versions of the other two models as well) and start an OSC server. If you then open *eval_mapper.maxpat* (which needs the [free Max runtime]([url](https://cycling74.com/downloads)) to be installed) you can try the whole image sonification system system in real-time.

Note that you should change directory to the *scripts* folder before running any of the above eval scripts (using `cd scripts`).

Here is how the Max patch *eval_mapper.maxpat* looks like:
<img width="837" height="1010" alt="image" src="https://github.com/user-attachments/assets/1e35cc88-87fc-4efe-a71c-8bf75a93fe6b" />


# Training

In the *scripts* folder there are also training scripts for training all models from scratch (initialized to the last-used hyperparameters). Not that training the mapper model (in *train_mapper.py*) requires that you train the image and sound models first (in *train_img_vae.py* and *train_sinewave_vae.py*, respectively).

The training scripts are expected to be invoked from the root folder of this repository.

# How to cite

[Link to paper](https://doi.org/10.5281/zenodo.17497987)

## APA

```
Laczkó, B., Rognes, M. E., & Jensenius, A. R. (2025). Image Sonification as Unsupervised Domain Transfer. Proceedings of the 17th International Symposium on Computer Music Multidisciplinary Research, 596–607. https://doi.org/10.5281/zenodo.17497987
```

## BibTeX

```
@inproceedings{laczkoImageSonificationUnsupervised2025,
	address = {London, UK},
	title = {Image {Sonification} as {Unsupervised} {Domain} {Transfer}},
	isbn = {979-10-97498-06-1},
	url = {https://doi.org/10.5281/zenodo.17497987},
	doi = {10.5281/zenodo.17497987},
	booktitle = {Proceedings of the 17th {International} {Symposium} on {Computer} {Music} {Multidisciplinary} {Research}},
	publisher = {The Laboratory PRISM “Perception, Representations, Image, Sound, Music”},
	author = {Laczkó, Bálint and Rognes, Marie E and Jensenius, Alexander Refsum},
	month = nov,
	year = {2025},
	pages = {596--607},
}
```

