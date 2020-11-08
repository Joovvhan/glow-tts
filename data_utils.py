import random
import numpy as np
import torch
import torch.utils.data

import commons 
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence, cmudict

import librosa
import os


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        # self.load_mel_from_disk = hparams.load_mel_from_disk
        self.load_mel_from_disk = True
        self.add_noise = hparams.add_noise
        self.add_space = hparams.add_space
        if getattr(hparams, "cmudict_path", None) is not None:
          self.cmudict = cmudict.CMUDict(hparams.cmudict_path)
        self.stft = commons.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)

        # Copy hparams to be used later
        self.hparams = hparams

        self.stat_mean, self.stat_std = np.load('stats.npy')
        self.stat_mean = self.stat_mean.reshape(80, 1)
        self.stat_std = self.stat_std.reshape(80, 1)
	
        # print(self.audiopaths_and_text)

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        return (text, mel)

    def get_mel(self, filename):
        if self.load_mel_from_disk:
            # melspec = torch.from_numpy(np.load(filename))
            try:
                melspec = torch.from_numpy(np.load(filename + '.npy'))
                assert melspec.size(0) == self.stft.n_mel_channels, (
                    'Mel dimension mismatch: given {}, expected {}'.format(
                        melspec.size(0), self.stft.n_mel_channels))

                return melspec
            except:
                print(filename + '.npy' + ' not found ... ')

        audio, sampling_rate = load_wav_to_torch(filename)
        
        if sampling_rate != self.stft.sampling_rate:
            # print(sampling_rate, self.stft.sampling_rate)
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.stft.sampling_rate))
        if self.add_noise:
            audio = audio + torch.rand_like(audio)
        audio_norm = audio / self.max_wav_value
        # audio_norm = audio_norm.unsqueeze(0)

        # Implement TensorFlowTTS style audio normalization
        audio_norm = audio_norm.numpy()

        # Implement TensorFlowTTS style trim
        '''
        audio_norm, _ = librosa.effects.trim(
            audio_norm,
            top_db=30, # top_db=config["trim_threshold_in_db"],
            frame_length=2048, # frame_length=config["trim_frame_size"],
            hop_length=512, # hop_length=config["trim_hop_size"],
        )
        '''
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        '''
        # Implement TensorFlowTTS style mel-spectrogram
        D = librosa.stft(
            audio_norm,
            n_fft=self.hparams.filter_length, # n_fft=config["fft_size"],
            hop_length=self.hparams.hop_length, # hop_length=hop_size,
            win_length=self.hparams.filter_length, # win_length=config["win_length"],
            window="hann", # window=config["window"],
            pad_mode="reflect",
        )
        S, _ = librosa.magphase(D)  # (#bins, #frames)

        # get mel basis
        fmin = 0 if self.hparams.mel_fmin is None else self.hparams.mel_fmin
        fmax = self.hparams.sampling_rate // 2 if self.hparams.mel_fmax is None else self.hparams.mel_fmax
        mel_basis = librosa.filters.mel(
            sr=self.hparams.sampling_rate,
            n_fft=self.hparams.filter_length,
            n_mels=self.hparams.n_mel_channels,
            fmin=fmin,
            fmax=fmax,
        )
        # mel = np.log10(np.maximum(np.dot(mel_basis, S), 1e-10)).T  # (#frames, #bins)
        melspec = np.log10(np.maximum(np.dot(mel_basis, S), 1e-10))  # (#bins, #frames)

        # Implement TensorFlowTTS style mel-spectrogram normalization
        melspec = (melspec - self.stat_mean) / self.stat_std
        if not os.path.isfile(filename + '.npy'):
            np.save(filename + '.npy', melspec)

        melspec = torch.tensor(melspec)
        '''

        return melspec

    def get_text(self, text):
        if self.add_space:
          text = " " + text.strip() + " "
        text_norm = torch.IntTensor(
            text_to_sequence(text, self.text_cleaners, getattr(self, "cmudict", None)))
        return text_norm

    def __getitem__(self, index):
        # print(index, len(self.audiopaths_and_text))
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per step
    """
    def __init__(self, n_frames_per_step=1):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded, output_lengths


"""Multi speaker version"""
class TextMelSpeakerLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_sid_text, hparams):
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.add_noise = hparams.add_noise
        self.add_space = hparams.add_space
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)
        if getattr(hparams, "cmudict_path", None) is not None:
          self.cmudict = cmudict.CMUDict(hparams.cmudict_path)
        self.stft = commons.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)

        self._filter_text_len()
        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)

    def _filter_text_len(self):
      audiopaths_sid_text_new = []
      for audiopath, sid, text in self.audiopaths_sid_text:
        if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
          audiopaths_sid_text_new.append([audiopath, sid, text])
      self.audiopaths_sid_text = audiopaths_sid_text_new

    def get_mel_text_speaker_pair(self, audiopath_sid_text):
        # separate filename, speaker_id and text
        audiopath, sid, text = audiopath_sid_text[0], audiopath_sid_text[1], audiopath_sid_text[2]
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        sid = self.get_sid(sid)
        return (text, mel, sid)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            if self.add_noise:
                audio = audio + torch.rand_like(audio)
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text):
        if self.add_space:
          text = " " + text.strip() + " "
        text_norm = torch.IntTensor(
            text_to_sequence(text, self.text_cleaners, getattr(self, "cmudict", None)))
        return text_norm

    def get_sid(self, sid):
        sid = torch.IntTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        return self.get_mel_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)


class TextMelSpeakerCollate():
    """ Zero-pads model inputs and targets based on number of frames per step
    """
    def __init__(self, n_frames_per_step=1):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded & sid
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            output_lengths[i] = mel.size(1)
            sid[i] = batch[ids_sorted_decreasing[i]][2]

        return text_padded, input_lengths, mel_padded, output_lengths, sid
