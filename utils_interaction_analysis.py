import os
import glob
from tqdm import tqdm
import pandas as pd
import json
import numpy as np

def load_stimulus_log_absolute(filepath, recorded_start_time, verbose=False):
    """
    Loads stimulus log, parses stimulus type and time,
    returns list of dicts with 'type' and absolute 'time'.
    
    Args:
        filepath (str): Path to stimulus log file.
        recorded_start_time (float): Recording start time in seconds to convert relative to absolute times.
        verbose (bool): If True, prints detailed parsing information.
    
    Returns:
        list of dict: Each dict contains 'type' and 'time' (absolute seconds).
    """
    stimuli = []
    
    if verbose:
        print(f"Reading stimulus log from: {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        if verbose:
            print(f"Total lines read: {len(lines)}")
            
        for idx, line in enumerate(lines):
            line = line.strip()
            
            # Skip empty lines
            if not line:
                if verbose:
                    print(f"Skipping empty line {idx}")
                continue
                
            # Check for colon separator
            if ':' not in line:
                if verbose:
                    print(f"Skipping line {idx}, no colon found: {line}")
                continue
                
            # Parse stimulus type and time
            parts = line.split(':', 1)  # split at first colon only
            stim_type = parts[0].strip()
            time_part = parts[1].strip()
            
            try:
                # Extract time (number before 's')
                time_str = time_part.split('s')[0].strip()
                time_sec = float(time_str)
                absolute_time = recorded_start_time + time_sec
                
                stimuli.append({
                    'type': stim_type, 
                    'time': absolute_time
                })
                
                if verbose:
                    print(f"Line {idx}: Parsed stim_type='{stim_type}', time={absolute_time:.3f}s")
                    
            except (ValueError, IndexError) as e:
                print(f"Warning: Failed to parse time in line {idx}: '{line}' - Error: {e}")
                continue
                
    except FileNotFoundError:
        print(f"Error: Stimulus log file not found: {filepath}")
        return []
    except Exception as e:
        print(f"Error reading stimulus log: {e}")
        return []
    
    if verbose:
        print(f"Successfully parsed {len(stimuli)} stimuli")
        
    return stimuli

# def filter_calls_produced(onsets, offsets, stimuli, buffer=0.02):
#     """

#    Filters out from th detected calls (onset/offset) those that coincide temporally with stimuli,
#      to keep only the vocalisations of the chicks.
    

#     Args:
#         onsets (array-like): Detected onset times in seconds.
#         offsets (array-like): Detected offset times in seconds.
#         stimuli (list of dict): Stimuli with absolute times.
#         buffer (float): Time window (seconds) around stimulus time to exclude.

#     Returns:
#         tuple: (filtered_onsets, filtered_offsets) as numpy arrays.
#     """
#     filtered_onsets = []
#     filtered_offsets = []
#     for o, off in zip(onsets, offsets):
#         is_stimulus = False
#         for stim in stimuli:
#             if stim['time'] - buffer <= o <= stim['time'] + buffer:
#                 is_stimulus = True
#                 break
#         if not is_stimulus:
#             filtered_onsets.append(o)
#             filtered_offsets.append(off)
#     return np.array(filtered_onsets), np.array(filtered_offsets)


# def filter_calls_produced(onsets, offsets, stimuli, biological_condition):
#     """
#     Filters out from the detected calls (onset/offset) those that coincide temporally with stimuli,
#     to keep only the vocalisations of the chicks.
#     Now customized for biological condition to use exact stimulus durations without buffer.

#     Args:
#         onsets (array-like): Detected onset times in seconds.
#         offsets (array-like): Detected offset times in seconds.
#         stimuli (list of dict): Stimuli with absolute times and types.
#         biological_condition (str): The biological condition ('pleasure', 'contact', 'cluck').

#     Returns:
#         tuple: (filtered_onsets, filtered_offsets) as numpy arrays.
#     """
#     # Duration mapping for each biological condition (in milliseconds)
#     dur_stimuli_ms = { 
#         'pleasure': {   # Biological_Condition = Pleasure
#             'pleasure': 107,  # biological token
#             'white':    107,
#             'silence':  107
#         },
#         'contact': {    # Biological_Condition = Contact
#             'contact':  223,  # biological token
#             'white':    223,
#             'silence':  223
#         },
#         'cluck': {      # Biological_Condition = Cluck
#             'cluck':    107,  # biological token
#             'white':    107,
#             'silence':  107
#         }
#     }
    
#     # Normalize biological_condition to lowercase for consistent matching
#     bio_condition_lower = biological_condition.lower().strip()
    
#     # Check if biological condition is valid
#     if bio_condition_lower not in dur_stimuli_ms:
#         print(f"Warning: Unknown biological condition '{biological_condition}'. Using default 200ms duration.")
#         default_duration_s = 0.2  # 200ms default
    
#     filtered_onsets = []
#     filtered_offsets = []
    
#     for o, off in zip(onsets, offsets):
#         is_stimulus = False
#         for stim in stimuli:
#             stim_type_lower = stim['type'].lower().strip()
#             stim_time = stim['time']
            
#             # Skip 'start' stimuli as they don't represent actual acoustic stimuli
#             if stim_type_lower == 'start':
#                 continue
            
#             # Get stimulus duration based on biological condition and stimulus type
#             if bio_condition_lower in dur_stimuli_ms:
#                 stimulus_duration_ms = dur_stimuli_ms[bio_condition_lower].get(stim_type_lower, 107)  # default to 107ms
#                 stimulus_duration_s = stimulus_duration_ms / 1000.0
#             else:
#                 stimulus_duration_s = default_duration_s
            
#             # Calculate exact exclusion window: stimulus start to stimulus end
#             exclusion_start = stim_time
#             exclusion_end = stim_time + stimulus_duration_s
            
#             # Check if onset falls within exact stimulus duration
#             if exclusion_start <= o <= exclusion_end:
#                 is_stimulus = True
#                 break
        
#         if not is_stimulus:
#             filtered_onsets.append(o)
#             filtered_offsets.append(off)
    
#     return np.array(filtered_onsets), np.array(filtered_offsets)


import numpy as np

def filter_calls_produced(onsets, offsets, stimuli, biological_condition):
    """
    Remove calls that overlap in time with playback stimuli, keeping only
    the chick's own vocalisations.

    This function assumes:
    - `onsets` and `offsets` are absolute times in seconds (same reference
      as the stimulus log and recording).
    - `stimuli` is a list of dicts with:
        * 'type': e.g. 'START', 'white', 'silence', 'pleasure', 'contact', 'cluck'
        * 'time': absolute onset time of the stimulus in seconds
    - `biological_condition` is the session's biological condition:
        'pleasure', 'contact' or 'cluck'.

    The 'START' event (the synchronisation beep) is ignored and never used
    for exclusion. All other events in the stimulus log (white, silence,
    biological tokens) are treated as true playback stimuli.

    Returns
    -------
    filtered_onsets : np.ndarray
        Onset times (s) of calls produced by the chick, after filtering.
    filtered_offsets : np.ndarray
        Offset times (s) of calls produced by the chick, after filtering.
    """

    # Stimulus duration mapping in milliseconds for each biological condition.
    # These values come from the stimulus design: each biological token has
    # a matched white-noise and silence control with identical duration.
    dur_stimuli_ms = {
        'pleasure': {   # Session with pleasure calls
            'pleasure': 107,   # biological token
            'white':    107,   # matched white noise
            'silence':  107,   # matched silence
        },
        'contact': {    # Session with contact calls
            'contact':  223,   # biological token
            'white':    223,
            'silence':  223,
        },
        'cluck': {      # Session with cluck calls
            'cluck':    107,   # biological token (hen cluck)
            'white':    107,
            'silence':  107,
        },
    }

    # Normalise the biological condition string
    bio = str(biological_condition).strip().lower()

    # Handle unexpected biological conditions gracefully
    if bio not in dur_stimuli_ms:
        print(
            f"Warning: Unknown biological condition '{biological_condition}'. "
            "Falling back to a default stimulus duration of 200 ms."
        )
        use_default_duration = True
        default_dur_s = 0.2  # 200 ms in seconds
    else:
        use_default_duration = False

    # Ensure onsets and offsets are NumPy arrays of floats
    onsets = np.asarray(onsets, dtype=float)
    offsets = np.asarray(offsets, dtype=float)

    filtered_onsets = []
    filtered_offsets = []

    # Loop over each detected call
    for onset, offset in zip(onsets, offsets):
        is_stimulus = False  # will be set to True if call overlaps with any stimulus

        # Compare this call with each stimulus event in the log
        for stim in stimuli:
            stim_type = str(stim.get('type', '')).strip()
            stim_type_lower = stim_type.lower()

            # Skip the START beep: it is only a synchronisation marker, not a stimulus
            if stim_type_lower == 'start':
                continue

            # Get the stimulus onset time; if missing or invalid, skip this entry
            try:
                stim_time = float(stim.get('time', np.nan))
            except (TypeError, ValueError):
                continue
            if np.isnan(stim_time):
                continue

            # Determine stimulus duration in seconds
            if not use_default_duration:
                dur_dict = dur_stimuli_ms[bio]

                if stim_type_lower not in dur_dict:
                    # If the stimulus label is unexpected, fall back to the
                    # biological token duration for this session (e.g. 'pleasure')
                    if bio in dur_dict:
                        dur_ms = dur_dict[bio]
                    else:
                        # As a last resort, skip this stimulus entry
                        continue
                else:
                    dur_ms = dur_dict[stim_type_lower]

                stim_dur_s = dur_ms / 1000.0  # convert ms to seconds
            else:
                stim_dur_s = default_dur_s

            stim_start = stim_time
            stim_end = stim_time + stim_dur_s

            # ---- Overlap criterion -----------------------------------------
            # Here we treat a call as "stimulus-driven" if it overlaps in time
            # with the playback window [stim_start, stim_end].
            #
            # Overlap occurs if the call does NOT:
            #   - finish before the stimulus starts, nor
            #   - start after the stimulus ends.
            #
            # i.e. we remove the call whenever there is any temporal overlap
            # with the speaker playback.
            # ----------------------------------------------------------------
            if not (offset <= stim_start or onset >= stim_end):
                is_stimulus = True
                break  # no need to check further stimuli for this call

        # If the call did not overlap with any stimulus, keep it
        if not is_stimulus:
            filtered_onsets.append(onset)
            filtered_offsets.append(offset)

    return np.asarray(filtered_onsets), np.asarray(filtered_offsets)




def calculate_call_suppression_ratio(onsets, stimuli, biological_condition, pre_window=10, post_window=10):
    """
    Calculates call suppression ratio per stimulus.
    Ratio = (calls in post-window) / (calls in pre-window).
    Post-window starts after stimulus ends.

    Args:
        onsets (array-like): Onset times of calls in seconds.
        stimuli (list of dict): Stimuli with absolute times and types.
        biological_condition (str): Biological condition ('pleasure', 'contact', 'cluck').
        pre_window (float): Seconds before stimulus to count calls.
        post_window (float): Seconds after stimulus end to count calls.

    Returns:
        list of dict: Each dict contains stimulus info and suppression ratio.
    """
    # Map biological condition to stimulus duration (seconds)
    duration_map = {
        'pleasure': 0.107,  # 107ms
        'contact': 0.223,   # 223ms  
        'cluck': 0.107      # 107ms
    }
    
    bio_condition_lower = biological_condition.lower().strip()
    stimulus_duration = duration_map.get(bio_condition_lower, 0.2)  # default 200ms
    
    if bio_condition_lower not in duration_map:
        print(f"Warning: Unknown biological condition '{biological_condition}'. Using 200ms default.")
    
    results = []
    for stim in stimuli:
        stim_time = stim['time']
        stim_type = stim['type'].lower().strip()
        
        # Skip start markers
        if stim_type == 'start':
            continue
        
        # Count calls in pre-window: [stim_time - pre_window, stim_time)
        calls_pre = np.sum((onsets >= stim_time - pre_window) & (onsets < stim_time))
        
        # Count calls in post-window: [stim_time + duration, stim_time + duration + post_window]
        post_start = stim_time + stimulus_duration
        post_end = post_start + post_window
        calls_post = np.sum((onsets >= post_start) & (onsets <= post_end))
        
        # Count calls during stimulus
        calls_during = np.sum((onsets >= stim_time) & (onsets <= stim_time + stimulus_duration))
        
        # Calculate ratio
        ratio = calls_post / calls_pre if calls_pre > 0 else np.nan

        results.append({
            'Stimulus_type': stim['type'],
            'Stimulus_time': stim_time,
            'Stimulus_duration': stimulus_duration,
            'Calls_pre_10s': calls_pre,
            'Calls_post_10s': calls_post,
            'Calls_during_stimulus': calls_during,
            'Call_Suppression_Ratio': ratio
        })
        
    return results




def group_stimuli_sequences_simple(stimuli, biological_condition):
    """
    Groups consecutive stimuli of the same type into sequences.
    Uses biological condition-specific duration for all stimuli in that condition.
    
    Args:
        stimuli (list of dict): Stimuli ordered by time, with 'type' and 'time' keys.
        biological_condition (str): Biological condition ('pleasure', 'contact', 'cluck').
    
    Returns:
        list of dict: Each dict with 'type', 'start_time', 'end_time'.
    """
    # Map biological condition to stimulus duration (seconds)
    duration_map = {
        'pleasure': 0.107,  # 107ms
        'contact': 0.223,   # 223ms  
        'cluck': 0.107      # 107ms
    }
    
    grouped = []
    if not stimuli:
        return grouped
    
    bio_condition_lower = biological_condition.lower().strip()
    stimulus_duration = duration_map.get(bio_condition_lower, 0.2)  # default 200ms
    
    if bio_condition_lower not in duration_map:
        print(f"Warning: Unknown biological condition '{biological_condition}'. Using 200ms default.")
    
    current_type = stimuli[0]['type'].lower()
    start_time = stimuli[0]['time']
    end_time = start_time + stimulus_duration
    
    for i in range(1, len(stimuli)):
        stim = stimuli[i]
        stim_type = stim['type'].lower()
        stim_time = stim['time']
        
        if stim_type == current_type:
            # Extend current sequence to last stimulus + duration
            end_time = stim_time + stimulus_duration
        else:
            # Finish current sequence
            grouped.append({
                'type': current_type,
                'start_time': start_time,
                'end_time': end_time
            })
            
            # Start new sequence
            current_type = stim_type
            start_time = stim_time
            end_time = stim_time + stimulus_duration
    
    # Add final sequence
    grouped.append({
        'type': current_type,
        'start_time': start_time,
        'end_time': end_time
    })
    
    return grouped


def map_stimulus_to_condition(stimulus_type, biological_condition):
    stimulus_mapping = {
        'pleasure': 'Biological',
        'contact': 'Biological', 
        'cluck': 'Biological',
        'white': 'White_Noise',
        'silence': 'Silence'
    }
    if stimulus_type.lower() == 'start':
        return None
    return {
        'Stimulus_Category': stimulus_mapping.get(stimulus_type.lower(), 'Unknown'),
        'Biological_Type': biological_condition if stimulus_type.lower() in ['pleasure', 'contact', 'cluck'] else None
    }

def create_comprehensive_dataset(calls_onsets, calls_offsets, stimuli, sequences, chick_id, biological_condition, sequence_type, second_pre_window=10, second_post_window=10):
    all_data = []
    # Ensure both arrays are aligned, same length and sorted by time
    onsets  = np.asarray(calls_onsets, dtype=float)
    offsets = np.asarray(calls_offsets, dtype=float)
    if onsets.size != offsets.size:
        raise ValueError("Each onset must have a matching offset.")
    idx = np.argsort(onsets)  # sort by onset time
    onsets, offsets = onsets[idx], offsets[idx]

    # Prepare ICI pairs: interval = next onset − current offset
    # Store current offset(i), next onset(i+1), and the gap (ICI)
    pair_offsets, pair_onsets, icis = [], [], []

    for i in range(len(onsets) - 1):
        gap = onsets[i+1] - offsets[i]   # pause between consecutive calls
        if gap >= 0:                     # ignore/ remove overlaps (negative gaps)
            pair_offsets.append(offsets[i])
            pair_onsets.append(onsets[i+1])
            icis.append(gap)

    # Convert lists back to arrays for easy masking
    pair_offsets = np.asarray(pair_offsets)
    pair_onsets  = np.asarray(pair_onsets)
    icis         = np.asarray(icis)

    for seq_idx, seq in enumerate(sequences):
        if str(seq['type']).lower() == 'start':
            continue
        condition_mapping = map_stimulus_to_condition(seq['type'], biological_condition)
        start_time, end_time = float(seq['start_time']), float(seq['end_time'])
        
        # Define analysis windows
        pre_start, pre_end   = start_time - second_pre_window, start_time
        post_start, post_end = end_time, end_time + second_post_window

        # Count calls in each window
        calls_pre    = int(np.sum((onsets >= pre_start)  & (onsets <  start_time)))
        calls_post   = int(np.sum((onsets >  end_time)   & (onsets <= post_end)))
        calls_during = int(np.sum((onsets >= start_time) & (onsets <= end_time)))
        csr = (calls_post / calls_pre) if calls_pre > 0 else np.nan

        # Compute mean ICI (pause between calls) before and after stimulus
        pre_mask  = (pair_offsets >= pre_start)  & (pair_onsets <= pre_end)
        post_mask = (pair_offsets >= post_start) & (pair_onsets <= post_end)
        mean_pre_ici  = float(np.mean(icis[pre_mask]))  if np.any(pre_mask)  else np.nan
        mean_post_ici = float(np.mean(icis[post_mask])) if np.any(post_mask) else np.nan

        record = {
            'Chick_ID': chick_id,
            'Biological_Condition': biological_condition,
            'Sequence_Type': sequence_type,
            'Sequence_Position': seq_idx + 1,
            'Stimulus_Type': condition_mapping['Stimulus_Category'],
            'Biological_Type': condition_mapping['Biological_Type'],
            'Stimulus_Start': start_time,
            'Stimulus_End': end_time,
            'Stimulus_Duration': end_time - start_time,
            'Calls_Pre_10s': calls_pre,
            'Calls_Post_10s': calls_post,
            'Calls_During_Stim': calls_during,
            'Call_Suppression_Ratio': csr,
            'Mean_ICI_Pre_Stim': mean_pre_ici,
            'Mean_ICI_Post_Stim': mean_post_ici,
            'Total_Calls_Session': len(onsets),
            'Trial_Order': seq_idx + 1
        }
        all_data.append(record)

        # Calculate actual ISI from previous sequence just to check
        if seq_idx > 0:
            prev_seq = sequences[seq_idx - 1]
            actual_isi = start_time - prev_seq['end_time']
            record['Actual_ISI'] = actual_isi
        else:
            record['Actual_ISI'] = np.nan

    return all_data

def latency_results_export(latency_results, biological_condition, chick_id, experiment_date, sequence_type):
    """
    Enriches and standardizes latency results for export across chicks and conditions.

    Adds contextual metadata such as biological condition, stimulus classification,
    chick ID, and experiment date — preparing a clean table for cross-session analysis.

    Args:
        latency_results (list of dict): Output of calculate_sequence_latency_response().
        biological_condition (str): Biological condition (Pleasure, Contact, Cluck).
        chick_id (str): Identifier for the chick.
        experiment_date (str): Date of the experimental session.
        sequence_type (str): Trial sequence label from metadata.

    Returns:
        list of dict: Clean, enriched latency records ready for CSV export.
    """
    enriched = []
    if not latency_results:
        return enriched

    # Helper function to classify stimulus type
    def classify_stimulus_type(seq_type):
        name = str(seq_type).lower().strip()
        if 'white' in name:
            return 'White_Noise'
        elif 'silence' in name:
            return 'Silence'
        elif name in ['pleasure', 'contact', 'cluck']:
            return 'Biological'
        else:
            return 'Unknown'

    for entry in latency_results:
        seq_type = entry.get('Sequence_Type', '').lower().strip()

        enriched_entry = {
            'Chick_ID': chick_id,
            'Experiment_Date': experiment_date,
            'Biological_Condition': biological_condition,
            'Sequence_Type': sequence_type,              # from metadata
            'Sequence_Category': seq_type,               # raw sequence label (e.g. 'white', 'pleasure')
            'Stimulus_Type': classify_stimulus_type(seq_type),
            'Sequence_Index': entry.get('Sequence_Index'),
            'Sequence_End': entry.get('Sequence_End'),
            'First_Call_After_Seq': entry.get('First_Call_After_Seq'),
            'Latency_Response_Sec': entry.get('Latency_Response_Sec')
        }

        enriched.append(enriched_entry)

    return enriched


def calculate_intercalls_intervals_stat(calls_onsets, calls_offsets, sequences):
    """
    Calculate session-level metrics with ICI defined as onset(call_i+1) - offset(call_i).
    """
    metrics = {}
    
    # Corrected ICI calculation: onset(call_i+1) - offset(call_i)
    if len(calls_onsets) > 1 and len(calls_offsets) >= len(calls_onsets):
        ici_intervals = []
        min_length = min(len(calls_onsets), len(calls_offsets))
        
        for i in range(min_length - 1):
            ici = calls_onsets[i + 1] - calls_offsets[i]
            if ici > 0:  # Only valid intervals
                ici_intervals.append(ici)
        
        if len(ici_intervals) > 0:
            metrics['Mean_ICI'] = np.mean(ici_intervals)
            metrics['Std_ICI'] = np.std(ici_intervals)
            # metrics['Median_ICI'] = np.median(ici_intervals)
        else:
            metrics['Mean_ICI'] = np.nan
            metrics['Std_ICI'] = np.nan
            # metrics['Median_ICI'] = np.nan
    else:
        metrics['Mean_ICI'] = np.nan
        metrics['Std_ICI'] = np.nan
        # metrics['Median_ICI'] = np.nan
    
    # Session call rate
    if len(sequences) > 0:
        session_duration = sequences[-1]['end_time'] - sequences[0]['start_time']
        metrics['Call_Rate_Per_Min'] = len(calls_onsets) / (session_duration / 60) if session_duration > 0 else np.nan
    else:
        metrics['Call_Rate_Per_Min'] = np.nan
    
    return metrics


def analyze_post_stimulus_responses(onsets, offsets, stimuli, biological_condition, 
                                   pre_window=5, post_window=10, max_latency=10):
    """
    Comprehensive analysis of chick responses to stimuli including latency and call patterns.
    
    Args:
        onsets (array-like): All detected call onsets in seconds.
        offsets (array-like): All detected call offsets in seconds.
        stimuli (list of dict): Stimuli with absolute times and types.
        biological_condition (str): Biological condition ('pleasure', 'contact', 'cluck').
        pre_window (float): Seconds before stimulus to analyze baseline.
        post_window (float): Seconds after stimulus end to analyze response.
        max_latency (float): Maximum latency to consider for first response.
    
    Returns:
        dict: Comprehensive response analysis for each stimulus.
    """
    # Duration mapping
    duration_map = {
        'pleasure': 0.107,  # 107ms
        'contact': 0.223,   # 223ms  
        'cluck': 0.107      # 107ms
    }
    
    bio_condition_lower = biological_condition.lower().strip()
    default_duration = duration_map.get(bio_condition_lower, 0.2)
    
    results = []
    
    for i, stim in enumerate(stimuli):
        if stim['type'].lower() == 'start':
            continue
            
        stim_time = stim['time']
        stim_type = stim['type'].lower().strip()
        stim_duration = default_duration
        stim_end = stim_time + stim_duration
        
        # Define analysis windows
        pre_start = stim_time - pre_window
        post_start = stim_end  
        post_end = stim_end + post_window
        latency_end = stim_end + max_latency
        
        # Count calls in different windows
        calls_pre = np.sum((onsets >= pre_start) & (onsets < stim_time))
        calls_during = np.sum((onsets >= stim_time) & (onsets <= stim_end))
        calls_post = np.sum((onsets > stim_end) & (onsets <= post_end))
        
        # Calculate first response latency
        calls_after_stim = onsets[onsets > stim_end]
        calls_within_latency = calls_after_stim[calls_after_stim <= latency_end]
        
        if len(calls_within_latency) > 0:
            first_response_time = calls_within_latency[0]
            response_latency = first_response_time - stim_end
            has_response = True
        else:
            first_response_time = np.nan
            response_latency = np.nan
            has_response = False
        
        # Calculate call rate changes
        pre_rate = calls_pre / pre_window if pre_window > 0 else 0
        post_rate = calls_post / post_window if post_window > 0 else 0
        rate_change = post_rate - pre_rate
        
        # Response probability (binary: responded or not within latency window)
        response_prob = 1 if has_response else 0
        
        result = {
            'Stimulus_Index': i,
            'Stimulus_Type': stim['type'],
            'Stimulus_Time': stim_time,
            'Stimulus_End': stim_end,
            'Stimulus_Duration': stim_duration,
            
            # Call counts
            'Calls_Pre_Stimulus': calls_pre,
            'Calls_During_Stimulus': calls_during, 
            'Calls_Post_Stimulus': calls_post,
            
            # Response metrics
            'First_Response_Time': first_response_time,
            'Response_Latency_Sec': response_latency,
            'Has_Response_Within_Latency': has_response,
            'Response_Probability': response_prob,
            
            # Rate analysis
            'Call_Rate_Pre': pre_rate,
            'Call_Rate_Post': post_rate,
            'Call_Rate_Change': rate_change,
            
            # Suppression analysis
            'Call_Suppression_Ratio': post_rate / pre_rate if pre_rate > 0 else np.nan,
            'Is_Suppressed': post_rate < pre_rate if pre_rate > 0 else False
        }
        
        results.append(result)
    
    return results


def calculate_sequence_latency_response(onsets, sequences):
    """
    Compute the latency of the first vocal response after each stimulus sequence.

    Specifically, it measures the time difference between the end of the last
    feedback (offset of the final stimulus in a trial sequence) and the onset
    of the first call that occurs after that point.

    Args:
        onsets (array-like): Array of detected call onset times (in seconds).
        sequences (list of dict): List of stimulus sequences, typically obtained
            from `group_stimuli_sequences_simple()`, each containing 'type',
            'start_time', and 'end_time'.

    Returns:
        list of dict: Each dictionary contains:
            - 'Sequence_Index': Sequential number of the sequence.
            - 'Sequence_Type': Type of the stimulus sequence (e.g., pleasure, white, silence).
            - 'Sequence_End': End time of the sequence (in seconds).
            - 'First_Call_After_Seq': Absolute onset time of the first call after the sequence.
            - 'Latency_Response_Sec': Latency value in seconds between sequence end
              and first call onset (NaN if no call occurs after the sequence).
    """
    results = []
    onsets = np.sort(np.asarray(onsets))

    for seq_idx, seq in enumerate(sequences):
        # Skip technical markers such as 'start'
        if seq['type'].lower() == 'start':
            continue

        seq_end = seq['end_time']

        # Find the first call that occurs after the end of the sequence
        following_calls = onsets[onsets > seq_end]
        if len(following_calls) > 0:
            first_call_after = following_calls[0]
            latency = first_call_after - seq_end
        else:
            first_call_after = np.nan
            latency = np.nan

        results.append({
            'Sequence_Index': seq_idx + 1,
            'Sequence_Type': seq['type'],
            'Sequence_End': seq_end,
            'First_Call_After_Seq': first_call_after,
            'Latency_Response_Sec': latency
        })

    return results

