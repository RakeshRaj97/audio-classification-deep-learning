# dataloader.py
import torch
import numpy as np
import albumentations
from pydub import AudioSegment
from arguments import args
from augmentations import AudioTransform, NoiseInjection, ShiftingTime, PitchShift, TimeStretch, RandomAudio, MelSpectrogram, SpecAugment, SpectToImage

class LoadAudio:
    def load_audio(path):
        try:
            sound = AudioSegment.from_mp3(path)
            sound = sound.set_frame_rate(args.sample_rate)
            sound_array = np.array(sound.get_array_of_samples(), dtype=np.float32)
        except:
            sound_array = np.zeros(args.sample_rate * args.max_duration, dtype=np.float32)

        return sound_array, args.sample_rate


class BirdDataset:
    def __init__(self, df, valid=False):

        self.filename = df.filename.values
        self.ebird_label = df.ebird_label.values
        self.ebird_code = df.ebird_code.values

        train_audio_augmentation = albumentations.Compose([
            RandomAudio(seconds=args.max_duration, always_apply=True),
            NoiseInjection(p=0.33),
            MelSpectrogram(parameters=args.melspectrogram_parameters, always_apply=True),
            SpecAugment(p=0.33),
            SpectToImage(always_apply=True)
        ])

        valid_audio_augmentation = albumentations.Compose([
            RandomAudio(seconds=args.max_duration, always_apply=True),
            MelSpectrogram(parameters=args.melspectrogram_parameters, always_apply=True),
            SpectToImage(always_apply=True)
        ])

        if valid:
            self.aug = valid_audio_augmentation
        else:
            self.aug = train_audio_augmentation

    def __len__(self):
        return len(self.filename)

    def __getitem__(self, item):

        filename = self.filename[item]
        ebird_code = self.ebird_code[item]
        ebird_label = self.ebird_label[item]

        data = LoadAudio.load_audio(f"{args.ROOT_PATH}/{ebird_code}/{filename}")
        spect = self.aug(data=data)["data"]

        target = ebird_label

        return {
            "spect": torch.tensor(spect, dtype=torch.float),
            "target": torch.tensor(target, dtype=torch.long)
        }


