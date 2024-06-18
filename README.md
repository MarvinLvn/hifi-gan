# HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis

### Jungil Kong, Jaehyeon Kim, Jaekyoung Bae

In our [paper](https://arxiv.org/abs/2010.05646), 
we proposed HiFi-GAN: a GAN-based model capable of generating high fidelity speech efficiently.<br/>
We provide our implementation and pretrained models as open source in this repository.

**Abstract :**
Several recent work on speech synthesis have employed generative adversarial networks (GANs) to produce raw waveforms. 
Although such methods improve the sampling efficiency and memory usage, 
their sample quality has not yet reached that of autoregressive and flow-based generative models. 
In this work, we propose HiFi-GAN, which achieves both efficient and high-fidelity speech synthesis. 
As speech audio consists of sinusoidal signals with various periods, 
we demonstrate that modeling periodic patterns of an audio is crucial for enhancing sample quality. 
A subjective human evaluation (mean opinion score, MOS) of a single speaker dataset indicates that our proposed method 
demonstrates similarity to human quality while generating 22.05 kHz high-fidelity audio 167.9 times faster than 
real-time on a single V100 GPU. We further show the generality of HiFi-GAN to the mel-spectrogram inversion of unseen 
speakers and end-to-end speech synthesis. Finally, a small footprint version of HiFi-GAN generates samples 13.4 times 
faster than real-time on CPU with comparable quality to an autoregressive counterpart.

Visit our [demo website](https://jik876.github.io/hifi-gan-demo/) for audio samples.


## Pre-requisites
1. Python >= 3.6
2. Clone this repository.
3. Install python requirements. Please refer [requirements.txt](requirements.txt)
4. Download and extract this [multi-speaker/multi-lingual dataset](https://huggingface.co/datasets/mbarnig/lb-de-fr-en-pt-12800-TTS-CORPUS).
And move all wav files to `/my/path/lb-de-fr-en-pt-12800-TTS-CORPUS/wavs`
5. Download and extract PB2009 (ask Thomas Hueber)


## A) Training

We'll first train a model on our multi-speaker/multi-lingual dataset.

```sh
DATA_PATH=/my/path/lb-de-fr-en-pt-12800-TTS-CORPUS
# 1) Create train/val/test split
python split_data --data_path ${DATA_PATH}/wavs --test_prop 0.1 --val_prop 0.1
# 2) Extract mel-spectrograms 
python extract_mel.py --data_path ${DATA_PATH}/wavs \
   --out_path ${DATA_PATH}/mels \
   --config config_16k.json
# 3) Train HiFi-GAN
python train.py --input_wavs_dir ${DATA_PATH}/wavs --input_training_file ${DATA_PATH}/train.txt \
  --input_validation_file ${DATA_PATH}/val.txt --input_mels_dir ${DATA_PATH}/mels \
  --config config_16k.json
```

Checkpoints and copy of the configuration will be saved in `cp_hifigan`.

### B) Fine-tuning

We'll fine-tune it on PB2009

```sh
DATA_PATH=/linkhome/rech/genscp01/uow84uh/agent/datasets/pb2009
# 1) Create train/val/test split
python split_data.py --data_path ${DATA_PATH}/wav --test_prop 0.1 --val_prop 0.1
# 2) Extract mel-spectrogram
# You can use the same instruction as above, but should already be generated using preprocess_datasets.py
# 2) Train HiFi-GAN
python train.py --input_wavs_dir ${DATA_PATH}/wav --input_training_file ${DATA_PATH}/train.txt \
  --input_validation_file ${DATA_PATH}/val.txt --input_mels_dir ${DATA_PATH}/mel \
  --config config_16k.json --fine_tuning True
```

## Inference from wav file
1. Make `test_files` directory and copy wav files into the directory.
2. Run the following command.
    ```
    python inference.py --checkpoint_file [generator checkpoint file path]
    ```
Generated wav files are saved in `generated_files` by default.<br>
You can change the path by adding `--output_dir` option.


## Inference for end-to-end speech synthesis
1. Make `test_mel_files` directory and copy generated mel-spectrogram files into the directory.<br>
You can generate mel-spectrograms using [Tacotron2](https://github.com/NVIDIA/tacotron2), 
[Glow-TTS](https://github.com/jaywalnut310/glow-tts) and so forth.
2. Run the following command.
    ```
    python inference_e2e.py --checkpoint_file [generator checkpoint file path]
    ```
Generated wav files are saved in `generated_files_from_mel` by default.<br>
You can change the path by adding `--output_dir` option.


## Acknowledgements
We referred to [WaveGlow](https://github.com/NVIDIA/waveglow), [MelGAN](https://github.com/descriptinc/melgan-neurips) 
and [Tacotron2](https://github.com/NVIDIA/tacotron2) to implement this.

