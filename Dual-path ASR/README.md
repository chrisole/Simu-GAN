#  Dual path ASR


The implementation for dual path ASR model that make use of clean speech and simulated noisy speech.

The espnet_model.py is the edited model file in Espnet0.9.1.


## Architecture

[filename](https://github.com/chrisole/Simu-GAN/blob/main/Dual-path%20ASR/figure/Two_channel.pdf)

The augmented dual-path ASR architecture. Two data flow are fed into Conformer-based ASR network as two independent batches, and the KL loss is computed between the outputs of decoder

## Reference
Espnet:
https://github.com/espnet/espnet
