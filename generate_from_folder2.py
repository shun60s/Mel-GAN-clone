# -*- coding: utf-8 -*-
"""
-----------------------------------------------------------------------------
Copyright (c) 2019 Descript Inc.

Please see LICENSE-melgan-neurips.md.
-----------------------------------------------------------------------------

Add notes
Change 2021/9/27
"""

from mel2wav import MelVocoder

from pathlib import Path
from tqdm import tqdm
import argparse
import librosa
import torch

import soundfile as sf  # add
from mel2wav.dataset import files_to_list # add
import os # add


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=Path, required=True)
    parser.add_argument("--save_path", type=Path, required=True)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--data_path", default=None, type=Path)  # add: specify --data_path when use file names list text file.
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    vocoder = MelVocoder(args.load_path)

    args.save_path.mkdir(exist_ok=True, parents=True)
    
    #-- add
    if args.data_path is not None:
         flist= files_to_list( Path(args.data_path) / "valid_files.txt")
         print ('ignore folder arugment, due to data_path is specified')
    else:
         flist= list(args.folder.glob("*.wav"))
    # -- end of add
    
    for i, fname in tqdm(enumerate(flist)): # chg
        wavname = os.path.basename(fname) #.name # chg
        #print (i, fname) # add
        wav, sr = librosa.core.load(fname)  # convert to sr=22050

        ###mel, _ = vocoder(torch.from_numpy(wav)[None]) # chg 
        mel = vocoder(torch.from_numpy(wav)[None]) # chg 
        recons = vocoder.inverse(mel).squeeze().cpu().numpy()

        ##librosa.output.write_wav(args.save_path / wavname, recons, sr=sr)
        sf.write(args.save_path / wavname, recons, sr) # chg


if __name__ == "__main__":
    main()
