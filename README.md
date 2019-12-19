# Synthesizing Fingerprint Images

The code is an implementation of the paper "Fingerprint Synthesis: Evaluating Fingerprint Search at Scale, in ICB 2018" by Kai Cao and Anil K. Jain.


## Dependencies

* Python 3
* Tensorflow 
* Tensorpack 0.5
* Numpy
* Scipy

## Pretrained Model

Please download the model [here](https://drive.google.com/file/d/1deYCP2THgISKvF27idbKvMvH-hpwlgEY/view?usp=sharing). Please extract the zipped file into ``model`` directory in root.

## Usage

The bash script 'Generate_Fingerprints.sh' will run the code for synthesizing fingerprint images. Running the bash script can be done simply by:

```bash
bash Generate_Fingerprints.sh
```

The script has 3 flags for specifying the following:
* The model to load (--load).
* The directory to store the generated fingerprint images (--sample_dir).
* The number of fingerprints to synthesize (--num_images).

You can change the values of these flags in the bash script.

## Citations

Please cite the following paper:

"Fingerprint Synthesis: Evaluating Fingerprint Search at Scale, in ICB 2018" by Kai Cao and Anil K. Jain.

```
@inproceedings{8411200,
author={K. {Cao} and A. {Jain}},
booktitle={2018 International Conference on Biometrics (ICB)},
title={Fingerprint Synthesis: Evaluating Fingerprint Search at Scale},
year={2018}}
```
