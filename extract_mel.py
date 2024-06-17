import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io.wavfile import read
from tqdm import tqdm

from env import AttrDict
from meldataset import mel_spectrogram

MAX_WAV_VALUE = 32768.0

def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate

def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)

def main(argv):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default='/home/engaclew/DATA/articulatory/lb-de-fr-en-pt-12800-TTS-CORPUS/wavs',
                        help='Path to the data folder.')
    parser.add_argument('--out_path', type=str,
                        default='/home/engaclew/DATA/articulatory/lb-de-fr-en-pt-12800-TTS-CORPUS/mels',
                        help='Path to the output folder.')
    parser.add_argument('--config', type=str,
                        default='/home/engaclew/agent/hifi-gan/config_v1.json',
                        help='Path to the config file (used for mel extraction).')
    args = parser.parse_args(argv)

    data_path = Path(args.data_path)
    out_path = Path(args.out_path)

    # Load mel extraction configuration
    with open(args.config) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)
    out_path.mkdir(parents=True, exist_ok=True)
    for wav_file in tqdm(data_path.glob('*.wav')):
        print(wav_file)
        audio, sampling_rate = load_wav(wav_file)
        audio = audio / MAX_WAV_VALUE
        audio = torch.FloatTensor(audio).unsqueeze(0)
        mel = get_mel(audio).squeeze(0).numpy()
        out_file = out_path / f'{wav_file.stem}.npy'
        np.save(out_file, mel)


if __name__ == "__main__":
    # execute only if run as a script
    args = sys.argv[1:]
    main(args)