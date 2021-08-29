import numpy as np
import scipy
from scipy import signal


# TODO: remove convenience wrappers

def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    lowcut :    float - the lowest frequency for the band pass filter
    highcut :   float - the highest frequency for the band pass filter
    fs :        int - the sampling frequency of the signal

    Default_
    order :     int - the order of the filter (>=1)
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    data :      DataFrame - The data containing the signals to filter
    lowcut :    float - the lowest frequency for the band pass filter
    highcut :   float - the highest frequency for the band pass filter
    fs :        int - the sampling frequency of the signal

    Default_
    order :     int - the order of the filter (>=1)
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def test_valid_window(window, test_level=5):
    """
    window : array_like - the window in the signal that has to be tested

    Default_
    test_level : float - the test level for the test_vqlid_window function (>1)

    This funtion tests the window to insure that it doesn't contain the signal of interest (spike)
    """
    try:
        second = np.percentile(window, 2)
        thirtyth = np.percentile(window, 30)
        return abs(second/thirtyth) < test_level
    except ValueError:
        print('Beware, your window may be empty...')
        return False


def init_noise_levels(signal, fs,
                      noise_window_size=0.01,
                      required_valid_windows=100,
                      old_noise_level_propagation=0.8,
                      test_level=5,
                      estimator_type="RMS",
                      percentile_value=25,
                      plot_estimator_graph=True):
    """
    estimator_type : string - the estimator to be used to compute the noise level window by window
    others : see in the subfunction

    This funtion calls the subfuntion corresponding to the chosen estimator
    """

    if estimator_type == "RMS":
        return init_noise_levels_RMS(signal, fs,
                      noise_window_size = noise_window_size,
                      required_valid_windows = required_valid_windows,
                      old_noise_level_propagation = old_noise_level_propagation,
                      test_level = test_level,
                      percentile_value = percentile_value,
                      plot_estimator_graph = plot_estimator_graph)

    elif estimator_type == "MAD":
        return init_noise_levels_MAD(signal, fs,
                      noise_window_size = noise_window_size,
                      required_valid_windows = required_valid_windows,
                      old_noise_level_propagation = old_noise_level_propagation,
                      test_level = test_level,
                      percentile_value = percentile_value,
                      plot_estimator_graph = plot_estimator_graph)

    else: return None


def init_noise_levels_RMS(signal, fs,
                      noise_window_size = 0.01,
                      required_valid_windows = 100,
                      old_noise_level_propagation = 0.8,
                      test_level = 5,
                      percentile_value = 25,
                      plot_estimator_graph = True):
    """
    signal :                        data_frame - the signal after denoising
    fs :                            int - the sampling frequency of the signal

    Default_
    noise_window_size :             float - the size of the windows that will be used to compute the noise level (in s)
    required_valid_windows :        int - minimum number of valid windows to first compute a noise level
    old_noise_level_propagation :   float - coef of propagation of the old noise level when a new valid window is found (0<=.<1)
    test_level :                    float - the test level for the test_vqlid_window function (>1)
    percentile_value :              int - the percentile to be used to compute the global noise (0<.<100)
    plot_estimator_graph :          bool - True to plot the estimtor level through the signal

    This function will return an array of noise level by window of the lenght noise_window_size in seconds
    estimated by RMS in each valid window. Then, a percentile will be apply to the list of initial noise levels.
    """

    nb_valid_windows = 0
    list_RMS = []
    noise_levels = []

    noise_level = -1

    #the signal is split in window of size noise_window_size
    for window_index in range(0,len(signal),int(fs*noise_window_size)):
        test = test_valid_window(signal.iloc[window_index: window_index + int(fs*noise_window_size)], test_level)
        #we count the number of valid windows we need to have to first compute the global noise
        if nb_valid_windows < required_valid_windows:
            if test == True :
                RMS = np.sqrt(np.mean(signal.iloc[window_index: window_index + int(fs*noise_window_size)]**2))
                list_RMS.append(RMS)
                nb_valid_windows += 1

            #when the required number is reach, we compute the global noise
            if nb_valid_windows == required_valid_windows:
                noise_level = np.percentile(list_RMS, percentile_value)
                for elm in range(0, window_index, int(fs*noise_window_size)):
                    noise_levels.append(noise_level)

        #for each new valid window after the required ones, the global noise level is updated
        else :
            if test == True:
                if (window_index + int(fs*noise_window_size)) > (len(signal)-1) :
                    RMS = np.sqrt(np.mean(signal.iloc[window_index:]**2))
                else :
                    RMS = np.sqrt(np.mean(signal.iloc[window_index: window_index + int(fs*noise_window_size)]**2))
                list_RMS.append(RMS)
                NX = np.percentile(list_RMS, percentile_value)
                new_noise_level = old_noise_level_propagation*noise_level + (1-old_noise_level_propagation)*NX
                noise_level = new_noise_level
            noise_levels.append(noise_level)

    #in the case where there is less valid windows than required_valid_windows :
    if noise_level == -1:

        #if no valid window was found, an array of 0 will be returned
        if noise_levels == []:
            for elm in range(0, len(signal), int(fs*noise_window_size)):
                noise_levels.append(0)

        #else,the global noise level is computed based on what was found
        else:
            noise_level = np.percentile(list_RMS, percentile_value)
            for elm in range(0, len(signal), int(fs*noise_window_size)):
                noise_levels.append(noise_level)

    noise_levels.append(noise_level)

    return noise_levels


def init_noise_levels_MAD(signal, fs,
                      noise_window_size = 0.01,
                      required_valid_windows = 100,
                      old_noise_level_propagation = 0.8,
                      test_level = 5,
                      percentile_value = 25,
                      plot_estimator_graph = True):
    """
    signal :                        data_frame - the signal after denoising
    fs :                            int - the sampling frequency of the signal

    Default_
    noise_window_size :             float - the size of the windows that will be used to compute the noise level (in s)
    required_valid_windows :        int - minimum number of valid windows to first compute a noise level
    old_noise_level_propagation :   float - coef of propagation of the old noise level when a new valid window is found (0<=.<1)
    test_level :                    float - the test level for the test_vqlid_window function (>1)
    percentile_value :              int - the percentile to be used to compute the global noise (0<.<100)
    plot_estimator_graph :          bool - True to plot the estimtor level through the signal

    This function will return an array of noise level by window of the lenght noise_window_size in seconds
    estimated by MAD in each valid window. Then, a percentile will be apply to the list of initial noise levels.
    """

    nb_valid_windows = 0
    list_MAD = []
    noise_levels = []

    noise_level = -1

    #the signal is split in window of size noise_window_size
    for window_index in range(0,len(signal),int(fs*noise_window_size)):
        test = test_valid_window(signal.iloc[window_index: window_index + int(fs*noise_window_size)], test_level)
        #we count the number of valid windows we need to have to first compute the global noise
        if nb_valid_windows < required_valid_windows:
            if test == True :
                MAD = scipy.stats.median_absolute_deviation(signal.iloc[window_index: window_index + int(fs*noise_window_size)])
                list_MAD.append(MAD)
                nb_valid_windows += 1

            #when the required number is reach, we compute the global noise
            if nb_valid_windows == required_valid_windows:
                noise_level = np.percentile(list_MAD, percentile_value)
                for elm in range(0, window_index, int(fs*noise_window_size)):
                    noise_levels.append(noise_level)

        #for each new valid window after the required ones, the global noise level is updated
        else :
            if test == True:
                if (window_index + int(fs*noise_window_size)) > (len(signal)-1) :
                    MAD = scipy.stats.median_absolute_deviation(signal.iloc[window_index:])
                else :
                    MAD = scipy.stats.median_absolute_deviation(signal.iloc[window_index: window_index + int(fs*noise_window_size)])
                list_MAD.append(MAD)
                NX = np.percentile(list_MAD, percentile_value)
                new_noise_level = old_noise_level_propagation*noise_level + (1-old_noise_level_propagation)*NX
                noise_level = new_noise_level
            noise_levels.append(noise_level)

    #in the case where there is less valid windows than required_valid_windows :
    if noise_level == -1:

        #if no valid window was found, an array of 0 will be returned
        if noise_levels == []:
            for elm in range(0, len(signal), int(fs*noise_window_size)):
                noise_levels.append(0)

        #else,the global noise level is computed based on what was found
        else:
            noise_level = np.percentile(list_MAD, percentile_value)
            for elm in range(0, len(signal), int(fs*noise_window_size)):
                noise_levels.append(noise_level)

    noise_levels.append(noise_level)

    return noise_levels
