import argparse
import random
import sys
from pathlib import Path

import numpy as np

random.seed(42)
np.random.seed(42)


def main(argv):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default='/home/engaclew/DATA/articulatory/lb-de-fr-en-pt-12800-TTS-CORPUS/wavs',
                        help='Path to the data folder.')
    parser.add_argument('--test_prop', type=int, default=0.1,
                        help='Proportion of the test set not used in training.')
    parser.add_argument('--val_prop', type=int, default=0.1,
                        help='Proportion of the val set not used in training.')
    args = parser.parse_args(argv)

    data_path = Path(args.data_path)
    wav_files = list(data_path.glob('*.wav'))
    np.random.shuffle(wav_files)
    NB_VAL, NB_TEST = int(args.val_prop*len(wav_files)), int(args.test_prop*len(wav_files))
    val_files = wav_files[:NB_VAL]
    test_files = wav_files[NB_VAL:NB_VAL+NB_TEST]
    train_files = wav_files[NB_VAL+NB_TEST:]

    for label, file_list in zip(['train', 'val', 'test'], [train_files, val_files, test_files]):
        with open(data_path.parent / f'{label}.txt', 'w') as fin:
            for file in file_list:
                fin.write(f'{file.stem}\n')

if __name__ == "__main__":
    # execute only if run as a script
    args = sys.argv[1:]
    main(args)