import numpy as np
import os
import pandas as pd
import librosa as lb
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as signal
from librosa.core import fft_frequencies
from librosa.util.exceptions import ParameterError
from typing import List, Tuple
from torch import Tensor
import torch as tr
# import torchaudio
from scipy.signal import medfilt






# function to extract reference onsets
def get_reference_onsets(file_txt):
    """Extract reference onsets from a txt file.

    Args:
        file_txt (str): Path to the txt file.

    Returns:
        list: List of reference onsets.

    """
    gt_onsets = []
    with open(file_txt, "r",  encoding='latin-1') as file:
            rows = file.readlines()

    for row in rows:
        columns = row.split()

        if columns:
            first_value = float(columns[0])
            gt_onsets.append(first_value) 
    assert gt_onsets, "File cannot be read!"
    return np.array(gt_onsets)


def get_reference_offsets(file_txt):
    gt_offsets = []
    with open(file_txt, "r",  encoding='latin-1') as file:
            rows = file.readlines()

    for row in rows:
        columns = row.split()

        if columns:
            second_value = float(columns[1])
            gt_offsets.append(second_value) 
    assert gt_offsets, "File cannot be read!"

    return np.array(gt_offsets)




#This  function was built to calculate the duration of the calls from the ground truth onsets and offsets
def calculate_durations(events): 
    '''This function calculates the duration of each event in a list of events.
    '''  
    durations = []
    for event in events:
        duration = event[1] - event[0]
        durations.append(duration)
    return durations



def calculate_intercalls_intervals(audio_y, events, sr):
    '''This function calculates the inter-call intervals between consecutive events in a list of events.'''
    inter_calls_intervals = []
    for i in range(len(events) - 1):  # Iterate over onsets up to the second-to-last one
        inter_calls_interval = events[i + 1][0] - events[i][1]
        inter_calls_intervals.append(inter_calls_interval)

    total_experiment_duration = lb.get_duration(y=audio_y, sr=sr)
    total_number_of_calls = len(events)
    # Calculate temporal analysis of calls and mean inter-call interval
    mean_inter_call_interval = total_experiment_duration / total_number_of_calls

    return inter_calls_intervals, mean_inter_call_interval




def bp_filter(audio_y, sr=44100, lowcut=2050, highcut=13000):
    # Apply the bandpass filter
    '''Here, the Nyquist frequency is used to normalise the filter cut-off frequencies
      with respect to the sampling rate of the audio signal. This is important because 
      digital filters operate at normalised frequencies, so it is necessary to express 
      the filter's cut-off frequencies in relation to the Nyquist frequency to ensure 
      that the filter works correctly with the sampled audio signal.
      '''
    nyquist = 0.5 * sr
    low = lowcut / nyquist        
    high = highcut / nyquist
    b, a = signal.butter(5, [low, high], btype='band')
    audio_y = signal.filtfilt(b, a, audio_y)

    return audio_y




def skewness(audio_y, sr=44100):
    '''Skewness is a measure of symmetry in a distribution.
    A standard normal distribution is perfectly symmetrical 
    and has zero skew. Skewness is measured by multiplying
    the difference between mean and median by three and dividing
     by the standard deviation. Skewness = 3(mean-median)/(standard deviation)3.'''
    # 
    mean = np.mean(audio_y)
    skew = np.sum((audio_y - mean) ** 3) / (len(audio_y) * np.std(audio_y) ** 3)
    return skew




def kurtosis(audio_y, sr=44100):
    '''Kurtosis is a measure of the "tailedness" of a distribution.
    Kurtosis is calculated by dividing the fourth central moment by the square of the variance'''
    #β2 = (E(x)4 / (E(x)2)2) − 3, 
    #where E(x)4 is the fourth central moment and E(x)2 is the second central moment.
    fourth_moment = np.mean((audio_y - np.mean(audio_y))**4)
    variance = np.mean((audio_y - np.mean(audio_y))**2)
    kurt = fourth_moment / variance**2 - 3
    return kurt



def segment_spectrogram(spectrogram, onsets, offsets, sr=44100):

    # Initialize lists to store spectrogram slices
    calls_S = []
    # Loop through each onset and offset pair
    for onset, offset in zip(onsets, offsets):
        # Convert time (in seconds) to sample indices
        onset_frames = lb.time_to_frames(onset, sr=sr)
        offset_frames = lb.time_to_frames(offset, sr=sr)

        #Extract the spectrogram slice from onset to offset 
        # REVIEW THIS value of epsilon
        # epsilon = duration*0.0001
        # epsilon_samples = lb.time_to_samples(epsilon, sr=44100)

        call_spec = spectrogram[:, onset_frames: offset_frames]#+ epsilon_samples]

        # Append the scaled log-spectrogram slice to the calls list
        calls_S.append(call_spec)
    
    return calls_S
    




def get_calls_waveform(audio_y, onsets, offsets, sr=44100):
    '''
    Extracts waveform segments from an audio signal 
    based on onset and offset times.
    '''
    calls = []

    # Loop through each onset and offset pair
    for onset, offset in zip(onsets, offsets):
        # Convert time (in seconds) to frame indices
        onset_sample = lb.time_to_samples(onset, sr=sr)
        offset_sample = lb.time_to_samples(offset, sr=sr)

        onset_sample = int(onset_sample)
        offset_sample = int(offset_sample)

        # Extract the waveform segment from onset to offset
        call_waveform = audio_y[onset_sample:offset_sample]

        # Append the waveform segment to the calls list
        calls.append(call_waveform)

    return calls




def get_calls_waveform_padded(audio_y, onsets, offsets, sr=44100):
    '''
    Extracts waveform segments from an audio signal 
    based on onset and offset times, with additional padding.
    '''
    calls = []
  
    padding = 0.2
    padding_samples = lb.time_to_samples( padding, sr=44100)


    # Loop through each onset and offset pair
    for i, (onset, offset) in enumerate(zip(onsets, offsets)):
        # Converti il tempo (in secondi) in indici di frame
        onset_frame = lb.time_to_frames(onset, sr=sr)
        offset_frame = lb.time_to_frames(offset, sr=sr)

        # add padding to the onset and offset frames
        onset_frame_padded = max(0, onset_frame - padding_samples)
        offset_frame_padded = min(len(audio_y), offset_frame + padding_samples)

        # Estrai il segmento di forma d'onda dall'inizio alla fine
        call_waveform = audio_y[onset_frame_padded:offset_frame_padded]

        # Aggiungi il segmento di forma d'onda alla lista delle chiamate
        calls.append(call_waveform)

    return calls







def save_F0_segments(segments, filenames):
    for segment, filename in zip(segments, filenames):
        #save a csv file with the F0 values
        segment.to_csv(filename, index=False)



        

def get_calls_F0(F0_in_frames, onsets, offsets, sr=44100, hop_length=512, n_fft=2048*2):
    # Initialize an empty list to store F0 slices
    calls = []

    # Loop through each onset and offset pair
    for onset, offset in zip(onsets, offsets):
        # Find the indices corresponding to the onset and offset times
        onset_index = lb.time_to_frames(onset, sr=sr, hop_length=hop_length, n_fft=n_fft)
        offset_index = lb.time_to_frames(offset, sr=sr, hop_length=hop_length, n_fft=n_fft)

        # Extract the F0 values within the specified time window
        call_F0 = F0_in_frames[onset_index:offset_index]
        
        # Append the F0 slice to the calls list
        calls.append(call_F0)

    return calls



def calc_morlet_psi(t: Tensor, s: float = 1.0, w: float = 6.0) -> Tensor:
    """
    Args:
        :param t: time values in seconds centered at 0
        :param s: scale of the wavelet (proportional to the fourier period, inversely proportional to the center frequency)
        :param w: Morlet wavelet hyperparameter (usually greater than 5), controls the Q factor
        :return: a complex tensor of wavelet coefficients in the time domain
    """
    assert t.ndim == 1
    x = t / s
    gaussian = tr.exp(-0.5 * (x ** 2))
    complex_sinusoid = tr.exp(1j * w * x)
    corrective_term = tr.exp(-0.5 * tr.tensor(w ** 2))
    amplitude_constant = -2 * tr.exp(-0.75 * tr.tensor(w ** 2))
    amplitude_constant += 1 + tr.exp(-tr.tensor(w ** 2))
    amplitude_constant = tr.pi ** -0.25 * (amplitude_constant ** -0.5)
    psi = amplitude_constant * (complex_sinusoid - corrective_term) * gaussian
    return psi


def segment_calls(audio, sr, onsets_sec, offsets_sec):
    '''
    Segment an audio waveform based on onset and offset times.

    Args:
    - audio (Tensor): Input audio waveform tensor.
    - sr (int): Sample rate of the audio.
    - onsets_sec (list): List of onset times in seconds.
    - offsets_sec (list): List of offset times in seconds.

    Returns:
    - segments (list): List of segmented audio waveforms.
    '''
    calls = []

    # Convert onset and offset times to samples
    onsets_samples = lb.time_to_samples(tr.tensor(onsets_sec), sr=sr)
    offsets_samples = lb.time_to_samples(tr.tensor(offsets_sec), sr=sr)

    # Loop through each onset and offset pair
    for onset_sample, offset_sample in zip(onsets_samples, offsets_samples):
        # Convert sample indices to integers
        onset_sample = int(onset_sample.item())
        offset_sample = int(offset_sample.item())

        # Extract the waveform segment from onset to offset
        segment = audio[:, onset_sample:offset_sample]

        # Append the waveform segment to the segments list
        calls.append(segment)

    return calls


def median_filter(signal, kernel_size=3):
    """
    Apply a 1D median filter to a multi-channel signal.

    Parameters:
    - signal: The input signal (either a 1D or 2D NumPy array).
    - kernel_size: The size of the median filter kernel (must be an odd integer).

    Returns:
    - Filtered signal: A NumPy array with the same shape as the input signal.
    """
    if signal.ndim == 1:  # Mono signal (1D)
        return medfilt(signal, kernel_size=kernel_size)
    else:  # Multi-channel signal (2D or more)
        return np.array([medfilt(channel, kernel_size=kernel_size) for channel in signal])