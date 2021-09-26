# -*- coding: utf-8 -*-
"""
-----------------------------------------------------------------------------
Copyright (c) 2019 Descript Inc.

Please see LICENSE-melgan-neurips.md.
-----------------------------------------------------------------------------

Add notes
Change 2021/9/25
"""

from mel2wav import MelVocoder

from pathlib import Path
from tqdm import tqdm
import argparse
import librosa
import torch

import soundfile as sf  # add


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=Path, required=True)
    parser.add_argument("--save_path", type=Path, required=True)
    parser.add_argument("--folder", type=Path, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    vocoder = MelVocoder(args.load_path)

    args.save_path.mkdir(exist_ok=True, parents=True)

    for i, fname in tqdm(enumerate(args.folder.glob("*.wav"))):
        wavname = fname.name
        wav, sr = librosa.core.load(fname)  # convert to sr=22050

        ###mel, _ = vocoder(torch.from_numpy(wav)[None]) # chg 
        mel = vocoder(torch.from_numpy(wav)[None]) # chg 
        recons = vocoder.inverse(mel).squeeze().cpu().numpy()

        ##librosa.output.write_wav(args.save_path / wavname, recons, sr=sr)
        sf.write(args.save_path / wavname, recons, sr) # chg


if __name__ == "__main__":
    main()
