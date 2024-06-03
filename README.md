# Differentiable and Stable Long-Range Tracking of Multiple Posterior Modes

## Abstract

Particle filters flexibly represent multiple posterior modes nonparametrically, via a collection of weighted samples, but have classically been applied to tracking problems with known dynamics and observation likelihoods. Such generative models may be inaccurate or unavailable for high-dimensional observations like images. We instead leverage training data to discriminatively learn particle-based representations of uncertainty in latent object states, conditioned on arbitrary observations via deep neural network encoders. While prior discriminative particle filters have used heuristic relaxations of discrete particle resampling, or biased learning by truncating gradients at resampling steps, we achieve unbiased and low-variance gradient estimates by representing posteriors as continuous mixture densities. Our theory and experiments expose dramatic failures of existing reparameterization-based estimators for mixture gradients, an issue we address via an importance-sampling gradient estimator. Unlike standard recurrent neural networks, our mixture density particle filter represents multimodal uncertainty in continuous latent states, improving accuracy and robustness. On a range of challenging tracking and robot localization problems, our approach achieves dramatic improvements in accuracy, while also showing much greater stability across multiple training runs.

## Disclaimer
> :warning: **Sorry for the messy code.  We do not currently have time to clean this up for public release but thought it is necessary to release the code at this time.**


## Data Download and Pre-Processing

#### DeepMind Dataset
1. Download the data using the instructions from [link](https://github.com/tu-rbo/differentiable-particle-filters)
2. Place the data in the "./data" directory
3. Run the data-preprocessing script:
	```
	cd data_preprocessing/Deepmind/splits && python3 main.py
	```

#### House3D Dataset
1. Download the data using the instructions from [link](https://github.com/AdaCompNUS/pfnet)
2. Place the data in the "./data" directory
3. Run the data-preprocessing scripts:
	```
	cd data_preprocessing/House3D/splits && python3 main.py
	cd data_preprocessing/House3D/splits && python3 main.py
	cd data_preprocessing/House3D/splits && python3 generate_ordering_and_split.py
	```


## Running Experiments
The config files and run files for all models for all experiments is located in "./src/experiments/".

Each method/experiment directory contains a "run.bash" file that can be used to run the experiment. Each method/experiment directory also contains a "config.yaml" file that is used to configure the experiment and may reference other .yaml config files present in this repo. 


#### Bearings Only:






Note that the names of experiments in the code may differ from that of the paper (aligning the names would take a lot of time and could break much of the code). Below we give a mapping of method names in the paper to that in the code:
| Paper Method Name   | Code Directory Name |
| --------     | ------- |
| TG-MDPF      | experiment0001    |
| IRG-MDPF     | experiment0002_implicit     |
| MDPF         | experiment0002_importance    |
| A-MDPF       | experiment0003_importance_init   |



## BibTex citation

Please consider citing our work if you use any code from this repo or ideas presented in the paper:
```
@inproceedings{younis2023mdpf,
  author    = {Ali Younis and
               Erik Sudderth},
  title     = {{Differentiable and Stable Long-Range Tracking of Multiple Posterior Modes}},
  booktitle = {Neurips},
  year      = {2023},
}
```

