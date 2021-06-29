import numpy as np
from librosa.core import stft, istft


class STFT(object):
    def __init__(self, frame_length, frame_step):
        self.frame_length = frame_length
        self.frame_step = frame_step

    def stft(self, wave):
        result = []
        # 按列为主序存储，也就是按通道为主序
        wave = np.asfortranarray(wave)
        channels = wave.shape[-1]
        for c in range(channels):
            data = wave[..., c]
            spectrogram = stft(data, n_fft=self.frame_length, hop_length=self.frame_step, center=False)
            spectrogram = np.expand_dims(spectrogram.T, axis=-1)
            result.append(spectrogram)
        result = np.concatenate(result, axis=-1)
        return result

    def istft(self, spectrogram, length):
        result = []
        # 按列为主序存储，也就是按通道为主序
        spectrogram = np.asfortranarray(spectrogram)
        channels = spectrogram.shape[-1]
        for c in range(channels):
            data = spectrogram[..., c].T
            # wave = istft(data, hop_length=self.frame_step, center=False)
            wave = istft(data, hop_length=self.frame_step, center=False, length=length)
            wave = np.expand_dims(wave.T, axis=1)
            result.append(wave)
        result = np.concatenate(result, axis=-1)
        return result