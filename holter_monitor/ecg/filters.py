import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfilt, sosfiltfilt, sosfilt_zi, savgol_filter
from scipy import stats
import pywt  # You'll need to install PyWavelets if not already installed

def filter_ecg_signal(data, fs=250, powerline_freq=None, adaptive=True):
    """
    Enhanced ECG signal filtering with improved baseline correction,
    adaptive noise handling, and automatic powerline frequency detection.
    
    Parameters:
    -----------
    data : array-like
        Raw ECG data
    fs : int
        Sampling frequency in Hz
    powerline_freq : float or None
        Powerline frequency (50 or 60 Hz). If None, will attempt to auto-detect.
    adaptive : bool
        Whether to use adaptive filtering based on signal quality
        
    Returns:
    --------
    array-like: Filtered ECG signal
    """
    if len(data) < 20:
        return data
    
    # Work with data as numpy array
    data = np.asarray(data, dtype=float)
    
    # Step 0: Improve initial preprocessing with more robust outlier handling
    median_val = np.median(data)
    mad = np.median(np.abs(data - median_val)) * 1.4826  # Approximates standard deviation for normal distribution
    upper_bound = median_val + 5 * mad
    lower_bound = median_val - 5 * mad
    data = np.clip(data, lower_bound, upper_bound)
    
    # Step 1: Enhanced baseline wander removal with segmented polynomial approach
    # This handles non-stationary baseline drift better than a single polynomial
    x = np.arange(len(data))
    
    # For long signals, use segmented approach to handle non-stationary baseline
    if len(data) > fs * 5:
        # Segment length (5 seconds)
        segment_length = int(fs * 5)
        num_segments = len(data) // segment_length + 1
        baseline = np.zeros_like(data)
        
        for i in range(num_segments):
            start_idx = i * segment_length
            end_idx = min((i + 1) * segment_length, len(data))
            
            if end_idx - start_idx < 10:  # Skip very short segments
                continue
                
            try:
                # Use lower degree for shorter segments
                degree = min(3, (end_idx - start_idx) // 100)
                z = np.polyfit(x[start_idx:end_idx], data[start_idx:end_idx], degree)
                p = np.poly1d(z)
                baseline[start_idx:end_idx] = p(x[start_idx:end_idx])
            except:
                # Use median for segments where fitting fails
                baseline[start_idx:end_idx] = np.median(data[start_idx:end_idx])
        
        # Remove estimated baseline
        data = data - baseline
    
    # Now apply high-pass filter for remaining baseline wander
    nyq = 0.5 * fs
    hp_cutoff = 0.5 / nyq  # 0.5 Hz cutoff for baseline wander
    sos_hp = butter(3, hp_cutoff, btype='high', output='sos')  # Use SOS for better numerical stability
    data_hp = sosfiltfilt(sos_hp, data)
    
    # Step 2: Enhanced multi-stage bandpass filtering 
    # First stage - wide band to preserve overall morphology
    low1 = 0.5 / nyq
    high1 = 45 / nyq
    sos1 = butter(4, [low1, high1], btype='band', output='sos')
    filtered_wide = sosfiltfilt(sos1, data_hp)
    
    # Second stage - narrow band optimized for QRS complex
    low2 = 8 / nyq   # Slightly higher to better isolate QRS complex
    high2 = 20 / nyq  # Focus more narrowly on QRS energy
    sos2 = butter(2, [low2, high2], btype='band', output='sos')
    filtered_qrs = sosfiltfilt(sos2, data_hp)
    
    # Third stage - special filter for P and T waves
    low3 = 0.5 / nyq
    high3 = 10 / nyq  # Lower high cut to focus on slower P and T waves
    sos3 = butter(3, [low3, high3], btype='band', output='sos')
    filtered_pt = sosfiltfilt(sos3, data_hp)
    
    # Advanced adaptive combination of the filtered signals
    # First estimate QRS presence for weighting
    qrs_energy = np.abs(filtered_qrs)
    
    # Use more robust QRS detection with adaptive thresholding
    window_len = int(fs * 1.5)  # 1.5 second window
    qrs_threshold = np.zeros_like(qrs_energy)
    
    # Calculate adaptive threshold along signal
    for i in range(0, len(qrs_energy), window_len//2):
        window_end = min(i + window_len, len(qrs_energy))
        window_slice = qrs_energy[i:window_end]
        local_threshold = np.mean(window_slice) + 1.8 * np.std(window_slice)
        qrs_threshold[i:window_end] = local_threshold
    
    # Smooth threshold to avoid abrupt changes
    qrs_threshold = savgol_filter(qrs_threshold, min(21, len(qrs_threshold)//10*2+1), 2)
    
    # Create QRS mask
    qrs_mask = (qrs_energy > qrs_threshold).astype(float)
    
    # Smooth the mask to avoid abrupt transitions
    window = int(fs * 0.1)  # 100ms window
    if window > 1:
        kernel = np.ones(window) / window
        qrs_mask = np.convolve(qrs_mask, kernel, mode='same')
    
    # Create refined combination weights considering three signal components
    qrs_weight = qrs_mask
    p_t_weight = 1 - qrs_mask
    
    # Advanced weighted combination with three filters
    filtered = (filtered_wide * 0.3 + 
                filtered_qrs * qrs_weight * 0.5 + 
                filtered_pt * p_t_weight * 0.2)
    
    # Step 3: Enhanced powerline noise detection and removal
    if powerline_freq is None:
        # More robust powerline detection
        from scipy import signal
        if len(data) > fs * 2:
            # Use multiple spectral analysis techniques
            detected_frequencies = []
            
            # Welch's method with multiple window sizes
            for nperseg in [fs*2, fs*4, fs]:
                if nperseg > len(data) // 2:
                    continue
                    
                f, psd = signal.welch(data, fs, nperseg=nperseg)
                # Normalize PSD to enhance peaks
                psd_norm = psd / (np.mean(psd) + 1e-10)
                
                # Find peaks with stricter criteria
                peak_indices = signal.find_peaks(psd_norm, height=2.0, distance=5)[0]
                peak_freqs = f[peak_indices]
                
                # Check for powerline frequencies with more precise detection
                for freq in peak_freqs:
                    if abs(freq - 50) < 1.5:
                        detected_frequencies.append(50)
                    elif abs(freq - 60) < 1.5:
                        detected_frequencies.append(60)
            
            # Use most common detected frequency
            if detected_frequencies:
                from collections import Counter
                powerline_freq = Counter(detected_frequencies).most_common(1)[0][0]
    
    # Step 3b: Enhanced adaptive notch filter for powerline noise
    if powerline_freq in (50, 60):
        # Apply cascaded notch filters for better stopband performance
        for freq_multiple in [1, 2, 3]:  # Remove fundamental and harmonics
            notch_freq = powerline_freq * freq_multiple
            if notch_freq < fs/2 - 5:  # Only if below Nyquist
                w0 = notch_freq / nyq
                # Use cascaded notch filters with different Q factors
                for Q_factor in [35.0, 50.0]:
                    b_notch, a_notch = iirnotch(w0, Q=Q_factor)
                    filtered = filtfilt(b_notch, a_notch, filtered)
    
    # Step 4: Enhanced wavelet-based denoising tailored for ECG
    try:
        # Use wavelet specifically suited for ECG signals
        wavelet = 'sym6'  # Better captures ECG morphology than sym4
        level = min(8, int(np.log2(len(filtered))))
        
        if level >= 3:  # Only apply if we have enough samples
            # Wavelet decomposition
            coeffs = pywt.wavedec(filtered, wavelet, level=level)
            
            # Advanced level-dependent thresholding
            for i in range(1, len(coeffs)):
                # Level-dependent threshold (more aggressive at higher frequencies)
                sigma = np.median(np.abs(coeffs[i])) / 0.6745
                if i <= 2:  # Preserve low frequency components (important for ECG)
                    threshold = sigma * 1.5
                else:  # More aggressive for high frequencies (mostly noise)
                    threshold = sigma * 2.5
                
                # Apply soft thresholding
                coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='soft')
            
            # Reconstruct signal
            filtered = pywt.waverec(coeffs, wavelet)
            
            # Ensure correct length
            filtered = filtered[:len(data)]
    except Exception:
        # Fall back gracefully
        pass
    
    # Step 5: Enhanced adaptive smoothing with optimized parameters
    if adaptive:
        # Calculate robust signal quality metrics
        diff_signal = np.diff(filtered)
        noise_level = np.median(np.abs(diff_signal - np.median(diff_signal))) * 1.4826
        signal_level = np.std(filtered)
        
        # Calculate signal-to-noise ratio
        snr = signal_level / (noise_level + 1e-10)
        
        if snr < 5:  # Very noisy signal
            # Use Savitzky-Golay with optimized parameters for ECG
            window_length = min(int(fs * 0.04) * 2 + 1, len(filtered) // 5)  # ~40ms window (must be odd)
            if window_length >= 5 and window_length % 2 == 1:
                filtered = savgol_filter(filtered, window_length, 3)
        elif snr < 10:  # Moderately noisy
            # Gentle smoothing with optimized cutoff
            sos_smooth = butter(3, 35/nyq, btype='low', output='sos')
            filtered = sosfiltfilt(sos_smooth, filtered)
    
    # Step 6: Morphological feature preservation using ECG-specific processing
    # Detect and enhance R peaks
    r_peak_candidates = []
    if len(filtered) > fs:
        # Use QRS energy to locate R peaks
        from scipy.signal import find_peaks
        
        # Enhance R peaks for better detection
        r_detector = sosfiltfilt(sos2, filtered)  # Use QRS filter
        
        # Find peaks with adaptive height threshold
        r_height = np.median(r_detector) + 0.6 * np.std(r_detector)
        r_peaks, _ = find_peaks(r_detector, height=r_height, distance=int(fs*0.2))
        
        if len(r_peaks) > 0:
            # For each detected peak, ensure its amplitude is well preserved
            for peak in r_peaks:
                start = max(0, peak - int(fs * 0.05))
                end = min(len(filtered), peak + int(fs * 0.05))
                
                # Preserve peak shape by limiting smoothing in its vicinity
                r_peak_candidates.append((start, end))
    
    # Step 7: Improved outlier detection and replacement with peak preservation
    window_size = min(fs // 3, len(filtered) // 10)  # 1/3 second window
    if window_size >= 5:
        # Calculate rolling median and MAD with edge handling
        rolling_med = np.zeros_like(filtered)
        rolling_mad = np.zeros_like(filtered)
        
        for i in range(len(filtered)):
            # Skip processing for detected R peak regions
            if any(start <= i < end for start, end in r_peak_candidates):
                continue
                
            start = max(0, i - window_size)
            end = min(len(filtered), i + window_size)
            window_data = filtered[start:end]
            rolling_med[i] = np.median(window_data)
            rolling_mad[i] = np.median(np.abs(window_data - rolling_med[i]))
        
        # Adaptive threshold with lower values for detected peak regions
        threshold_factor = 4.5  # Slightly less aggressive
        outliers = np.abs(filtered - rolling_med) > threshold_factor * (rolling_mad + 1e-10)
        
        # Don't process outliers in R peak regions
        for start, end in r_peak_candidates:
            outliers[start:end] = False
        
        # Replace outliers with local median or interpolated values
        if np.sum(outliers) > 0:
            for i in np.where(outliers)[0]:
                start = max(0, i - window_size)
                end = min(len(filtered), i + window_size)
                
                # Find non-outlier points in the window
                valid_points = ~outliers[start:end]
                if np.sum(valid_points) > 0:
                    # Replace with local median of non-outlier points
                    filtered[i] = np.median(filtered[start:end][valid_points])
    
    # Step 8: Final centering and scaling
    # Center around middle of ADC range with small offset to avoid clipping
    filtered = filtered - np.mean(filtered) + 2048
    
    # Clip to valid ADC range
    return clip(filtered)


def clip(values):
    """Clip values to valid ADC range (0-4095)."""
    return np.clip(np.round(values), 0, 4095).astype(int).tolist()