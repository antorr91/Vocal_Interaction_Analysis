import numpy as np
import librosa as lb
import os
import pandas as pd
from visualization import plot_signal
import matplotlib.pyplot as plt


MIN_DURATION =0.02 #seconds
MAX_DURATION = 0.54
AV_DURATION = 0.19



def offset_detection_local_minimum(file_name, onsets, min_duration= MIN_DURATION, max_duration= MAX_DURATION, av_duration= AV_DURATION):
    
    y, sr= lb.load(file_name, sr=44100)
    
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
    # expected_window = expected_window * sr
    offsets= []

    for i, startw in enumerate(start_window):
        endw = end_window[i]
        # get the spectrogram portion selecting all the frequencies from the start to the end of the window
        spectrogram_window = spectrogram[:, startw:endw]
        # average the spectrogram inside the spectrogram window
        average_spectrogram = np.mean(spectrogram_window, axis=0) # shape should be (1, 45)
        
         
        n_min = np.argmin(average_spectrogram)# This returns the first ocorrence of the minimum value, to consider if this is the best approach
        
        offset_in_frames = startw + n_min
        offset_seconds = lb.frames_to_time(offset_in_frames, sr=44100, hop_length=512)

        offsets.append(offset_seconds)
    return np.array(offsets)    




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
        # plot the signal with the ground thruth offsets

        # plt.figure(figsize=(10, 5))
        # plt.plot(np.arange(len(average_spectrogram)), average_spectrogram, alpha=0.7)
        
                  
        # # Filter gt_offsets within the window and plot them
        # gt_offsets_in_window = [offset for offset in gt_offsets_fr if startw < offset < endw]
        # for offset in gt_offsets_in_window:
        #     plt.axvline(x=offset - startw, alpha=0.8, color="r")  # Plot only the gt_offsets within the window
        
        
        # plt.xlabel('Frame index')
        # plt.ylabel('Average Spectrogram')
        # plt.title('Average Spectrogram with Ground Truth Offsets')
        # plt.show()


        #first order difference  y (n+ h)-y (n)
        y_diff = np.diff(average_spectrogram, n=1)
        #plot_signal(y_diff)

        
        n_min = np.argmin(y_diff)# This returns the first ocorrence of the minimum value, to consider if this is the best approach
        
        offset_in_frames = startw + n_min
        offset_seconds = lb.frames_to_time(offset_in_frames, sr=44100, hop_length=512)

        offsets.append(offset_seconds)

        # plt.figure(figsize=(10, 5))
        # plt.plot(np.arange(len(y_diff)), y_diff, alpha=0.7)
        # Filter gt_offsets within the window and plot them
        # gt_offsets_in_window = [offset for offset in gt_offsets_fr if startw < offset < endw]
        # for offset in gt_offsets_in_window:
        #     plt.axvline(x=offset - startw, alpha=0.8, color="r") 
        #     plt.axvline(x=offset_in_frames - startw, alpha=0.8, color="g") 
        # #rf_offset_in_window = [offset for offset in offset_in_frames if startw < offset < endw]    
        # plt.xlabel('Frame index')
        # plt.ylabel('Amplitude')
        # plt.title('Signal after 1st order differencing with  Ground Truth Offset ( red) and Predicted Offset (green)')
        # plt.show()


    return np.array(offsets)    



# This function compute the offset finding the local minimun + the second order of differenece of energy
def offset_detection_second_order(file_name, onsets,  min_duration= MIN_DURATION, max_duration= MAX_DURATION, av_duration= AV_DURATION):
    
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
        # plot the signal with the ground thruth offsets

        # second order x"t = x't - x't-1 = (xt - xt-1) - (xt-1 - xt-2) = xt - 2xt-1+xt-2
        #first order difference  y (n+ h)-y (n)
        y_diff = np.diff(average_spectrogram, n=2)
        #plot_signal(y_diff)


        
        n_min = np.argmin(y_diff)# This returns the first ocorrence of the minimum value, to consider if this is the best approach
        
        offset_in_frames = startw + n_min
        offset_seconds = lb.frames_to_time(offset_in_frames, sr=44100, hop_length=512)

        offsets.append(offset_seconds)
    return np.array(offsets)    