import os
import numpy as np
import librosa as lb
import pandas as pd
import madmom
import glob
from scipy.io import wavfile
from scipy.signal import medfilt
from src.analysis.visualization import plot_signal, visualize_activation_and_gt
import matplotlib.pyplot as plt


MIN_DURATION =0.02 #seconds
MAX_DURATION = 0.54
AV_DURATION = 0.19



def high_frequency_content(file_name, hop_length=441, sr=44100, spec_num_bands=15, spec_fmin=2500, spec_fmax=5000, 
                           spec_fref=2800, pp_threshold=1.8, pp_pre_avg=25, pp_post_avg=1, pp_pre_max=3, pp_post_max=2, visualise_activation=False):
    '''Compute the onsets using the high frequency content algorithm with madmom.
    Args:
        file_name (str): Path to the audio file.
        hop_length (int): Hop length in samples.
        sr (int): Sample rate.
        spec_num_bands (int): Number of filter bands.
        spec_fmin (int): Minimum frequency.
        spec_fmax (int): Maximum frequency.
        spec_fref (int): Reference frequency.
        pp_threshold (float): Threshold for peak picking.
        pp_pre_avg (int): Number of frames to average before peak.
        pp_post_avg (int): Number of frames to average after peak.
        pp_pre_max (int): Number of frames to search for local maximum before peak.
        pp_post_max (int): Number of frames to search for local maximum after peak.
    Returns:
        list: Onsets in seconds.
    '''

    spec_mdm = madmom.audio.spectrogram.FilteredSpectrogram(file_name,num_bands=spec_num_bands, fmin=spec_fmin , fmax=spec_fmax, fref=spec_fref, norm_filters=True, unique_filters=True)
    # Compute onset based on High frequency content with madmom
    activation = madmom.features.onsets.high_frequency_content(spec_mdm)
    # Applying the peak picking function to count number of onsets
    peaks = madmom.features.onsets.peak_picking(activation,threshold=pp_threshold, smooth=None, pre_avg=pp_pre_avg, post_avg=pp_post_avg, pre_max=pp_pre_max, post_max=pp_post_max)

    hfc_onsets_seconds =[(peak * hop_length / sr ) for peak in peaks ]    
    if visualise_activation:
        return np.array(hfc_onsets_seconds), activation
    else:
        return np.array(hfc_onsets_seconds)
    



# here add gt_offsets as parameter if need to evaluate the model
def offset_detection_first_order(file_name, onsets, min_duration= MIN_DURATION, max_duration= MAX_DURATION, av_duration= AV_DURATION):
    
    y, sr= lb.load(file_name, sr=44100)
    # plot_signal(y)
    spectrogram= lb.feature.melspectrogram(y=y, sr=44100, hop_length=512, n_fft=2048 * 2, window=0.12, fmin= 2050, fmax=8000, n_mels= 15)
    
    min_duration_from_onsets = onsets + min_duration
    max_duration_from_onsets = onsets + max_duration

    #window to look for the offset
    # expected_window = max_duration_from_onsets- min_duration_from_onsets

    # transform into frames 

    min_duration_from_onsets_frames = lb.time_to_frames(min_duration_from_onsets, sr=44100, hop_length=512)
    max_duration_from_onsets_frames = lb.time_to_frames(max_duration_from_onsets, sr=44100, hop_length=512)
    
    start_window = min_duration_from_onsets_frames

    end_window = max_duration_from_onsets_frames

    # gt_offsets_fr = lb.time_to_frames(gt_offsets, sr=44100, hop_length=512)
    # expected_window = expected_window * sr
    offsets= []

    for i, startw in enumerate(start_window):
        endw = end_window[i]
        

        # if any onset is detected in the window, replace endw by the onset
        onsets_frames = lb.time_to_frames(onsets, sr=44100, hop_length=512)
        for onset in onsets_frames:
            if onset > startw and onset < endw:
                endw = onset - 1     # 1 frame =librosa.frames_to_time(1, sr=44100, hop_length=512)  0.011609977324263039
                break

        # get the spectrogram portion selecting all the frequencies from the start to the end of the window
        spectrogram_window = spectrogram[:, startw:endw]  # shape should be (15, 45)
        # average the spectrogram inside the spectrogram window
        average_spectrogram = np.mean(spectrogram_window, axis=0) # shape should be (1, 45)

        #first order difference  y (n+ h)-y (n)
        y_diff = np.diff(average_spectrogram, n=1)
        #plot_signal(y_diff)

        
        n_min = np.argmin(y_diff)# This returns the first ocorrence of the minimum value, to consider if this is the best approach
        
        offset_in_frames = startw + n_min
        offset_seconds = lb.frames_to_time(offset_in_frames, sr=44100, hop_length=512)

        offsets.append(offset_seconds)


    return np.array(offsets)    




def median_filter(signal, kernel_size=11):
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


    

def global_shift_correction(predicted_onsets, shift):
    '''subtract shift second to all the predicted onsets.
    Args:
        predicted_onsets (list): List of predicted onsets.
        shift (float): Global shift in seconds.
    Returns:
        list: Corrected predicted onsets.
    '''
    # compute global shift
    corrected_predicted_onsets = []
    for po in predicted_onsets:
        #subtract a global shift of 0.01 ms or more  to all the predicted onsets
        if po - shift > 0: # to avoid negative onsets
            corrected_predicted_onsets.append(po - shift)
        else:
            continue

    return np.array(corrected_predicted_onsets)


def normalise_audio(file_name):
    '''Normalise the audio file to have a maximum amplitude of 1.
    Args:
        file_name (str): Path to the audio file.
    Returns:
        np.array: Normalised audio signal.
    '''
    # Read the audio file
    sample_rate, audio_data = wavfile.read(file_name)

    # Compute the normalization factor (Xmax)
    max_amplitude = np.max(np.abs(audio_data))

    # Normalize the audio
    normalized_audio = audio_data / max_amplitude

    return normalized_audio, sample_rate


def double_onset_correction(onsets_predicted, correction= 0.020):
    '''Correct double onsets by removing onsets which are less than a given threshold in time.
    Args:
        onsets_predicted (list): List of predicted onsets.
        gt_onsets (list): List of ground truth onsets.
        correction (float): Threshold in seconds.
    Returns:
        list: Corrected predicted onsets.
    '''    
    # Calculate interonsets difference
    #gt_onsets = np.array(gt_onsets, dtype=float)

    # Convert to numpy array if not already
    onsets_predicted = np.array(onsets_predicted, dtype=float)
    
    # Handle edge cases
    if len(onsets_predicted) <= 1:
        return onsets_predicted
    
    # Sort onsets to ensure they are in chronological order
    onsets_predicted = np.sort(onsets_predicted)

    # Calculate the difference between consecutive onsets
    differences = np.diff(onsets_predicted)

    # Create a list to add the filtered onset and add a first value
    filtered_onsets = [onsets_predicted[0]]  #Add the first onset

    # Subtract all the onsets which are less than fixed threshold in time
    for i, diff in enumerate(differences):
      if diff >= correction:
      # keep the onset if the difference is more than the given selected time
        filtered_onsets.append(onsets_predicted[i + 1])
        #print the number of onsets predicted after correction
    return np.array(filtered_onsets)





def filter_calls_within_experiment(onsets, offsets, start_exp, end_exp):
    """
    Filters onsets and offsets between start and end of experiment, ensuring paired consistency.
    
    Parameters:
        onsets (list or np.array): List of onset times in seconds.
        offsets (list or np.array): List of offset times in seconds.
        start_exp (float): Start time of the experiment in seconds.
        end_exp (float): End time of the experiment in seconds.

    Returns:
        np.array: Filtered onsets.
        np.array: Filtered offsets.
    """
    filtered_onsets = []
    filtered_offsets = []
    
    for onset, offset in zip(onsets, offsets):
        # Check if both onset and offset are within the experimental window
        if start_exp <= onset <= end_exp and start_exp <= offset <= end_exp:
            filtered_onsets.append(onset)
            filtered_offsets.append(offset)
    
    return np.array(filtered_onsets), np.array(filtered_offsets)


