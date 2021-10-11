# Simu-GAN


This implementation is to train a Simu-GAN model that simulates the noisy speech from clean speech. 
## File system
datasets: customized speech datasets for Simu-GAN 
<br> &ensp; -- AudioTask: task name
<br> &ensp; &ensp; &ensp; -- trainA: the directory of clean speech
<br> &ensp; &ensp; &ensp; -- trainB: the directory of noisy speech
<br>

data: data preprocessing and loading for Simu-GAN  

options: options for Simu-GAN
<br> &ensp; -- base_options.py: base configurations for model
<br> &ensp; -- train_options.py: base configurations for training
<br> &ensp; -- test_options.py: base configurations for testing

models: architectures of GAN models 

utils: tools and functions for data processing. loudness normalization, visualization.

train.py: python script to train Simu-GAN 
<br> &ensp; -- example: python train.py --dataroot ./datasets/AudioTask --name AudioTask --CUT_mode CUT --checkpoints_dir checkpoints_AudioTask

test.py: python script to generate simulated speech by trained Simu-GAN
<br> &ensp; -- example: python test.py --dataroot ./datasets/AudioTask --name AudioTask --CUT_mode CUT --checkpoints_dir AudioTask --state Test 

requirements.txt 
<br> &ensp; -- environment installation: pip install -r requirements.txt 


## Listening example

The simulated noisy and its corresponding clean speech and real noisy speech are available at: https://chrisole.github.io/ICASSP2022-demo/

## Reference

The implementation of image translation using GAN: https://github.com/taesungp/contrastive-unpaired-translation