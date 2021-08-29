import numpy as np
import pandas as pd
import scipy
from scipy.signal import *


# TODO: base on gradient information, look for negative gradient above a
# threshold and then look for positive threshold in window

def find_spike(signal, initial_index, noise_levels, fs, spike_info,
               window_size = 0.002,
               noise_window_size = 0.01,
               threshold_factor = 3.5,
               reduct_factor  = 0.33,
               maxseparation = 0.0008,
               time_checkmaxlocal = 0.0002,
               burst_threshold = 7):
    """
    signal :                    data_frame - the signal after denoising
    initial_index :             int - the index where we initiate the search for a spike
    noise_levels :              array_like - list of the noise levels window by window
    fs :                        int - the sampling frequency of the signal
    spike_info :                array_like - the array to be updated containing the infos for each found spike

    Default_
    window_size :               float - minimum time between 2 spike (in s)
    noise_window_size :         float - the size of the windows that will be used to compute the noise level (in s)
    threshold_factor :          float - multiplication factor applied to the noise threshold level to find spike
    reduct_factor  :            float - multiplication factor applied to the threshold level to find a max in a spike
                                (in addition to the threshold_factor) (0<=.<=1)
    maxseparation :             float - maximum time between the min and the max of a spike (used to find max) (in s)
    time_checkmaxlocal :        float - time scale where a value has to be a local max to be considered max of the spike (used to find max) (in s)
    burst_threshold :           float - minimum amplitude threshold defining if a spike is in a burst phenomenom or not

    This function browse the signal begining at the initial index, searching for a spike and recording the usefull information of the fund spike
    """

    offset_index = int(np.round(signal.index[0]*fs/1000))

    #the function is triggered when the signal crosses the negative threshold
    if initial_index < len(signal) + offset_index:
        i = initial_index
        for value in signal.iloc[initial_index-offset_index:]:
            threshold = threshold_factor*noise_levels[int(np.round((i/fs)//noise_window_size))]
            if value < -threshold:
                indice_1er_depass = i
                while(True):
                    if i < len(signal)+offset_index-1:
                        if signal.iloc[i-offset_index + 1]>signal.iloc[i-offset_index]:
                            break
                        else :
                            i+=1
                    else :
                        break

                #we check if the signal wasn't already below the threshold due to the side effect of the previous spike
                if signal.iloc[initial_index-offset_index-1] < -threshold:
                    for value_ in signal.iloc[initial_index-offset_index:]:
                        if value_ > -threshold:
                            return i
                        i += 1

                #we look for a max at the right of the depolarization
                i_max_right = 'nan'
                for k in range(int(np.round(maxseparation*fs))):
                    if (i-offset_index + k) < len(signal)-1:
                        if signal.iloc[i-offset_index+k] > threshold * reduct_factor and signal.iloc[i-offset_index+k]>signal.iloc[i-offset_index+k+1]:
                            if checkmaxlocal(signal, "right",i+k,offset_index,int(np.round(time_checkmaxlocal*fs))):
                                i_max_right = i+k
                                break

                #we look for a max at the left of the depolarization
                i_max_left = 'nan'
                for k in range(int(np.round(maxseparation*fs))):
                    if (i-offset_index - k) > 0:
                        if signal.iloc[i-offset_index-k] > threshold * reduct_factor and signal.iloc[i-offset_index-k]>signal.iloc[i-offset_index-k-1]:
                            if checkmaxlocal(signal, "left",i-k,offset_index,int(np.round(time_checkmaxlocal*fs))):
                                i_max_left = i-k
                                break


                #if there is no max, we consider there is no spike and resume the search
                if i_max_left == 'nan' and i_max_right == 'nan':
                    while signal.iloc[i-offset_index] < -threshold:
                        i += 1
                    return i

                #elsem we record the spike's informations
                else:
                    amplitude = 0
                    if i_max_left != 'nan' and i_max_right != 'nan':
                        if signal.iloc[i_max_left-offset_index] < signal.iloc[i_max_right-offset_index]:
                            amplitude = signal.iloc[i_max_right-offset_index] - signal.iloc[i-offset_index]
                        else :
                            amplitude = signal.iloc[i_max_left-offset_index] - signal.iloc[i-offset_index]
                    elif i_max_left != 'nan':
                            amplitude = signal.iloc[i_max_left-offset_index] - signal.iloc[i-offset_index]
                    elif i_max_right != 'nan':
                            amplitude = signal.iloc[i_max_right-offset_index] - signal.iloc[i-offset_index]

                    if amplitude >= 2*burst_threshold:
                        is_it_burst = True
                    elif amplitude < 2*burst_threshold:
                        is_it_burst = False

                    #corresponding labels :
                        # indice min, indice 1er depasssement
                        # max gauche, max droite
                        # variation d'amplitude entre min et max
                        # is_it_burst

                    spike_info.append([i, indice_1er_depass,
                                        i_max_left, i_max_right,
                                        amplitude, is_it_burst])
                    return i+int(np.round(window_size*fs))

                break
            i += 1

    return -44


def checkmaxlocal(local_signal, sens, supposed_i_max, offset_index, nb_index_research=3):
    """
    local_signal :      data_frame - the signal after denoising
    sens :              string - the orientation in wich we have to check if the value is a max
    supposed_i_max :    int - index that is supposed to be a local max of the signal
    offset_index :      int - begining index of the local signal

    Default_
    nb_index_research : int - index size where the value has to be local max

    This function check if the supposed maximum is really the max in the direction of sens for the span of nb_index_research
    """
    try:
        if(sens == "right"):
            k = 0
            while k <= nb_index_research:
                try:
                    if((local_signal.iloc[supposed_i_max-offset_index + k]) < (local_signal.iloc[supposed_i_max-offset_index + k + 1])):
                        return False
                except ValueError:
                    k = nb_index_research
                    print('Beware, one of your spike may be at the very end of your signal')
                k += 1
            return True

        elif(sens == "left"):
            k = 0
            while k <= nb_index_research:
                try:
                    if((local_signal.iloc[supposed_i_max-offset_index - k]) < (local_signal.iloc[supposed_i_max-offset_index - k - 1])):
                        return False
                except ValueError:
                    k = nb_index_research
                    print('Beware, one of your spike may be at the very begining of your signal')
                k+=1
            return True
    except:
        print('Beware, the direction you want to explore is not authorized, please chose "right" or "left"')
        return False


def find_spikes(signal, noise_levels, fs,
               window_size = 0.002,
               noise_window_size = 0.01,
               threshold_factor = 3.5,
               positive_threshold_factor = 0.33,
               maxseparation = 0.0008,
               time_checkmaxlocal = 0.0002,
               burst_threshold = 7):
    """
    signal :                    data_frame - the signal after denoising
    noise_levels :              array_like - list of the noise levels window by window
    fs :                        int - the sampling frequency of the signal

    Default_
    window_size :               float - minimum time between 2 spike (in s)
    noise_window_size :         float - the size of the windows that will be used to compute the noise level (in s)
    threshold_factor :          float - multiplication factor applied to the noise threshold level to find spike
    positive_threshold_factor : float - multiplication factor applied to the threshold level to find a max in a spike
                                (in addition to the threshold_factor) (0<=.<=1)
    maxseparation :             float - maximum time between the min and the max of a spike (used to find max) (in s)
    time_checkmaxlocal :        float - time scale where a value has to be a local max to be considered max of the spike (used to find max) (in s)
    burst_threshold :           float - minimum amplitude threshold defining if a spike is in a burst phenomenom or not

    This function browse the signal, searching for spikes and recording the usefull information of the fund spikes in a DataFrame
    """

    initial = int(np.round(signal.index[0]*fs/1000))
    spike_info = []

    while initial != -44:
        initial = find_spike(signal, initial, noise_levels, fs, spike_info,
                             window_size = window_size,
                             noise_window_size = noise_window_size,
                             threshold_factor = threshold_factor,
                             maxseparation = maxseparation,
                             time_checkmaxlocal = time_checkmaxlocal,
                             burst_threshold = burst_threshold)

    df_spike_info = pd.DataFrame(spike_info)
    df_spike_info.columns = ['indice_min', 'indice_1er_depass','indice_max_gauche','indice_max_droite','Delta_amplitudes', 'burst?']

    return df_spike_info


def record_spikes(signal, fs, spike_info,
                  align_method,
                  t_before = 0.001,
                  t_after = 0.002):
    """
    signal :        data_frame - the signal after denoising
    fs :            int - the sampling frequency of the signal
    spike_info :    data_frame - the DataFrame containing the informations about the spikes to record
    align_method :  string - the specific point in spike_info to use for spike's alignement

    Default_
    t_before :      float - time to record before the alignement point (in s)
    t_after :       type - time to record after the alignement point (in s)

    This function record the spikes aligned with align_method and store them in a DataFrame
    """

    if (align_method in spike_info.columns) == False:
        print("align_method is incorrect, please choose one of the following :" + str(spike_info.columns))
        return None

    else:
        spike_centers = spike_info[align_method].values

    t_b = int(np.round(fs*(t_before)))
    t_a = int(np.round(fs*(t_after)))

    data = np.array([[float(x) for x in range(t_b+t_a+1)]])

    initial_index = int(np.round(signal.index[0]*fs/1000))

    for center in spike_centers:
        if center < t_b + initial_index:
            spike = [0 for i in range(0, t_b-(center-initial_index))]
            spike = np.concatenate((spike, signal.values[:center + t_a - initial_index]))
            data = np.insert(data, len(data), spike, axis=0)

        elif center > len(signal)-t_a + initial_index:
            spike = signal.values[center - t_b - initial_index:]
            spike = np.concatenate((spike,[0 for i in range(0, t_a - (len(signal) + initial_index-center)+1)]))
            data = np.insert(data, len(data), spike, axis=0)

        else :
            spike = signal.values[center - t_b - initial_index: center + t_a + 1 - initial_index]
            data = np.insert(data, len(data), spike, axis=0)

    data = np.delete(data, 0, axis = 0)
    data = data.transpose()
    spike_data = pd.DataFrame(data)

    return spike_data


def record_spikes_oneline(signal, fs, spike_info,
                  align_method,
                  t_before = 0.001,
                  t_after = 0.002):
    """
    signal :        data_frame - the signal after denoising
    fs :            int - the sampling frequency of the signal
    spike_info :    data_frame - the DataFrame containing the informations about the spikes to record
    align_method :  string - the specific point in spike_info to use for spike's alignement

    Default_
    t_before :      float - time to record before the alignement point (in s)
    t_after :       type - time to record after the alignement point (in s)

    This function record the spikes aligned with align_method and store them in a single signal of the same size as signal
    """

    if (align_method in spike_info.columns) == False:
        print("align_method is incorrect, please choose one of the following :" + str(spike_info.columns))
        return None

    else:
        spike_centers = spike_info[align_method].values

    offset_index = int(np.round(signal.index[0]*fs/1000))

    t_b = int(np.round(fs*(t_before)))
    t_a = int(np.round(fs*(t_after)))

    data = np.array(['NaN' for x in range(len(signal))])
    data = data.astype(float)
    times = np.array(['NaN' for x in range(len(signal))])
    times = times.astype(pd.Timestamp)

    for center in spike_centers:
        if center < t_b + offset_index:
            data[:center + t_a - offset_index] = signal.values[:center + t_a - offset_index]
            times[:center + t_a - offset_index] = signal.index[:center + t_a - offset_index]

        elif center > len(signal) - t_a + offset_index:
            data[center - t_b - offset_index:] = signal.values[center - t_b - offset_index:]
            times[center - t_b - offset_index:] = signal.index[center - t_b - offset_index:]

        else :
            data[center - t_b - offset_index: center + t_a + 1 - offset_index] = signal.values[center - t_b - offset_index: center + t_a + 1 - offset_index]
            times[center - t_b - offset_index: center + t_a + 1 - offset_index] = signal.index[center - t_b - offset_index: center + t_a + 1 - offset_index]

    spike_data_oneline = pd.DataFrame(data, index = times.astype(float))

    return spike_data_oneline


def record_spikes_clusterized_oneline(signal,
                                      fs,
                                      spike_info,
                                      align_method,
                                      labels = None,
                                      t_before = 0.001,
                                      t_after = 0.002):
    """
    signal :        data_frame - the signal after denoising
    fs :            int - the sampling frequency of the signal
    spike_info :    data_frame - the DataFrame containing the informations about the spikes to record
    align_method :  string - the specific point in spike_info to use for spike's alignement
    labels          array_like - the cluster's labels of the differents spikes, none id the labels are already in spike_info

    Default_
    t_before :      float - time to record before the alignement point (in s)
    t_after :       type - time to record after the alignement point (in s)

    This function record the spikes aligned with align_method and store them in a different signal of the same size as signal for each cluster
    """

    try:
        if labels != None:
            print('The labels located in labels will be used')
        else:
            try:
                labels = spike_info['cluster_label'].values
                print('The labels located in spike_info will be used')
            except KeyError as e:
                print('The labels has to be either in labels or in the column cluster_label in spike_info')
                raise e

    except ValueError:
         print('The labels located in labels will be used')




    if (align_method in spike_info.columns) == False:
        print("align_method is incorrect, please choose one of the following :" + str(spike_info.columns))
        return None

    else:
        spike_centers = spike_info[align_method].values

    nb_clusters = max(labels) + 1
    nb_spikes = len(labels)

    if (-1 in labels) == True:
        nb_clusters_ = nb_clusters + 1
    else:
        nb_clusters_ = nb_clusters

    offset_index = int(np.round(signal.index[0]*fs/1000))

    t_b = int(np.round(fs*(t_before)))
    t_a = int(np.round(fs*(t_after)))

    data = np.array([['NaN' for x in range(len(signal))] for i in range(nb_clusters_)])
    data = data.astype(float)
    columns = []

    for cluster_number in range(nb_clusters):
        columns.append('Cluster n°' + str(cluster_number))
        for center in spike_centers[labels == cluster_number]:
            if center < t_b + offset_index:
                data[cluster_number, :center + t_a - offset_index] = signal.values[:center + t_a - offset_index]

            elif center > len(signal) - t_a + offset_index:
                data[cluster_number, center - t_b - offset_index:] = signal.values[center - t_b - offset_index:]

            else :
                data[cluster_number, center - t_b - offset_index: center + t_a + 1 - offset_index] = signal.values[center - t_b - offset_index: center + t_a + 1 - offset_index]

    if (-1 in labels) == True:
        columns.append('Cluster de bruit')
        for center in spike_centers[labels == -1]:
            if center < t_b + offset_index:
                data[cluster_number, :center + t_a - offset_index] = signal.values[:center + t_a - offset_index]

            elif center > len(signal) - t_a + offset_index:
                data[cluster_number, center - t_b - offset_index:] = signal.values[center - t_b - offset_index:]

            else :
                data[cluster_number, center - t_b - offset_index: center + t_a + 1 - offset_index] = signal.values[center - t_b - offset_index: center + t_a + 1 - offset_index]


    data = data.transpose()
    spike_data_clusterized_oneline = pd.DataFrame(data, index = signal.index)
    spike_data_clusterized_oneline.columns = columns

    return spike_data_clusterized_oneline


def spike_fine_tuning(spike_info):
    true_before = spike_info.loc[spike_info['burst?'] == True]
    True_spikes = true_before.index.values
    true_before.index = [x for x in range(len(true_before))]
    for i in range(len(true_before)-1):
        if(true_before.loc[i+1]['indice_1er_depass']-true_before.loc[i]['indice_1er_depass']<5000):
            for j in range(True_spikes[i],True_spikes[i+1]):
                spike_info.at[j, 'burst?'] = True
