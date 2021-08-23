import numpy as np
import matplotlib.pyplot as plt
from random import randint
from mpl_toolkits.mplot3d import Axes3D

def print_spikes(spike_data, # TODO: add argument of time window
                 t_before_alignement = 0,
                 first_spike = 0,
                 last_spike = -1,
                 fs = 25000,
                 randomize = False,
                 nb_spike = 20,
                 y_lim_min = -50, # TODO: detect automatically
                 y_lim_max = 60):
    """
    spike_data :            data_frame - the DataFrame containing the recorded spikes

    Default_
    t_before_alignement :   float - time of recording before the alignement point (in s)
    first_spike :           int - first spike to print in indice if randomize == False
    last_spike :            int - last spike to print in indice if randomize == False
    fs :                    int - the sampling frequency of the signal
    randomize :             bool - allow to chose between printing random spikes or not
    nb_spike :              int - number of spikes to print if randomize == True
    y_lim_min :             float - negative limit in y axis in the plot
    y_lim_max :             float - negative limit in y axis in the plot
    
    This function print the spikes in a figure randomly or not
    """
    
    if randomize == True:        
        kept = []
        m = len(spike_data.values[0])
        if m <= nb_spike:
            kept = [i for i in range(m)]
        else:      
            i = 0  
            while i < nb_spike:
                r = randint(0,m-1)
                if (r in kept) == False:
                    kept.append(r)
                    i += 1
        
        x = spike_data.iloc[:,kept].values
        
    else:
        x = spike_data.iloc[:,first_spike:last_spike]
        
    figure = plt.figure()
    t_b = int(np.round(fs*(t_before_alignement)))
    axes = figure.add_subplot(1, 1, 1)
    axes.plot((np.array([ind for ind in range(len(spike_data))])-t_b)*1000/fs, x)
    axes.set_xlabel('Time in ms')
    axes.set_ylabel('Amplitude [µV]')
    axes.set_ylim(y_lim_min , y_lim_max)
    axes.grid()

# TODO: add time parameter (1 second)
#
def print_spikes_oneline(signal, spike_data_oneline):
    """
    signal :                data_frame - the signal after denoising
    spike_data_oneline :    data_frame - the recorded spikes stored in one signal
    
    This function print the spikes on the signal of origin
    """
    plt.figure()
    plt.plot(signal.index, signal, color = 'blue')
    plt.plot(signal.index, spike_data_oneline, color = 'red')
    plt.title('Filtered Signal with Detected Spikes with RMS')
    plt.xlabel('Time Windows')
    plt.ylabel('Amplitude [µV]')
    plt.grid(True)


# TODO : look at number of total spike plotting, maybe add to print_spikes_oneline
def print_spikes_clusterized(spike_data,
                             labels,
                             t_before_alignement = 0,
                             nb_spike = 20,
                             y_lim_min = -50,
                             y_lim_max = 60,
                             fs = 25000):
    """
    spike_data :            data_frame - the DataFrame containing the recorded spikes
    labels                  array_like - the cluster's labels of the differents spikes

    Default_
    t_before_alignement :   float - time of recording before the alignement point (in s)
    fs :                    int - the sampling frequency of the signal
    nb_spike :              int - number of spikes to print by cluster
    y_lim_min :             float - negative limit in y axis in the plot
    y_lim_max :             float - negative limit in y axis in the plot
    
    This function print the spikes sorted by cluster in differents subplots then all of them in one final subplot
    """
    
    nb_clusters = max(labels) + 1
    nb_spikes = len(labels)
    
    if (-1 in labels) == True:
        nb_clusters_ = nb_clusters + 1
    else:
        nb_clusters_ = nb_clusters
    
    nb_line = nb_clusters_//2
    if nb_clusters_%2 != 0:
        nb_line += 1
    
    t_b = int(np.round(fs*(t_before_alignement)))
    y = (spike_data.index-t_b)*1000/fs
        
    a = [i for i in range(len(labels))]
    b = np.transpose([a,list(labels)])
    
    figure = plt.figure()
    plt.gcf().subplots_adjust(left = 0.1, bottom = 0.1, right = 0.9, top = 0.9, wspace = 0.2, hspace = 0.5)
    for nb in range(nb_clusters):
        data = spike_data.iloc[:,[x for x,y in b if y==nb]]
        nb_spike_clusterized = len(data.values[0])
        
        kept = []
        
        if nb_spike_clusterized <= nb_spike:
            kept = [i for i in range(nb_spike_clusterized)]
        else:      
            i = 0  
            while i < nb_spike:
                r = randint(0,nb_spike_clusterized-1)
                if (r in kept) == False:
                    kept.append(r)
                    i += 1
        
        x = data.iloc[:,kept].values
        axes = figure.add_subplot(nb_line, 2, nb+1)
        axes.plot(y, x)
        axes.set_xlabel('Time in ms')
        axes.set_title('Cluster n° ' + str(nb) + ', nb_spikes = ' + str(nb_spike_clusterized) + ' representing ' + str(np.round(nb_spike_clusterized/nb_spikes*100)) + '\% of the total')
        axes.set_ylabel('Amplitude [µV]')
        axes.set_ylim(y_lim_min , y_lim_max)
        axes.grid(True)
        
    if (-1 in labels) == True:
        data = spike_data.iloc[:,[x for x,y in b if y==-1]]
        nb_spike_clusterized = len(data.values[0])
        
        kept = []
        
        if nb_spike_clusterized <= nb_spike:
            kept = [i for i in range(nb_spike_clusterized)]
        else:      
            i = 0  
            while i < nb_spike:
                r = randint(0,nb_spike_clusterized-1)
                if (r in kept) == False:
                    kept.append(r)
                    i += 1
        
        x = data.iloc[:,kept].values
        
        axes = figure.add_subplot(nb_line, 2, nb+2)
        axes.plot(y, x)
        axes.set_xlabel('Time in ms')
        axes.set_title('Cluster de bruit, nb_spikes = ' + str(nb_spike_clusterized) + ' representing ' + str(np.round(nb_spike_clusterized/nb_spikes*100)) + '\% of the total')
        axes.set_ylabel('Amplitude [µV]')
        axes.set_ylim(y_lim_min , y_lim_max)
        axes.grid(True)


def print_spikes_clusterized_oneline(signal, spike_data_clusterized_oneline,
                                     y_lim_min = -50,
                                     y_lim_max = 60):
    """
    signal :                            data_frame - the signal after denoising
    spike_data_clusterized_oneline :    data_frame - the recorded spikes stored in one different signal by cluster

    Default_
    y_lim_min :             float - negative limit in y axis in the plot
    y_lim_max :             float - negative limit in y axis in the plot
    
    This function print the spikes on the signal of origin cluster by cluster
    """

    nb_line = (len(spike_data_clusterized_oneline.columns) + 1)//2
    if (len(spike_data_clusterized_oneline.columns) + 1)%2 != 0:
        nb_line += 1

    ghost_array = np.array(['NaN' for x in range(len(signal))])
    ghost_array = ghost_array.astype(float)
    legend = ['Signal d\'origine']

    figure = plt.figure()
    plt.gcf().subplots_adjust(left = 0.1, bottom = 0.1, right = 0.9, top = 0.9, wspace = 0.2, hspace = 0.5)

    i = 0
    for cluster in spike_data_clusterized_oneline.columns:
        axes = figure.add_subplot(nb_line, 2, i+1)
        axes.plot(signal.index, signal)

        for ghost in range(i):
            axes.plot(signal.index, ghost_array)

        axes.plot(signal.index, spike_data_clusterized_oneline[cluster])
        axes.set_title(cluster)
        axes.set_xlabel('Time in ms')
        axes.set_ylabel('Amplitude [µV]')
        axes.set_ylim(y_lim_min , y_lim_max)
        axes.grid(True)

        legend.append(cluster)
        i += 1

    axes = figure.add_subplot(nb_line, 2, i+1)
    axes.plot(signal.index, signal)
    for cluster in spike_data_clusterized_oneline.columns:
        axes.plot(signal.index, spike_data_clusterized_oneline[cluster])
    axes.set_title('Every clusters on the signal of origin')
    axes.set_xlabel('Time in ms')
    axes.set_ylabel('Amplitude [µV]')
    axes.set_ylim(y_lim_min , y_lim_max)
    axes.legend(legend)
    axes.grid(True)

    plt.figure()
    plt.plot(signal.index, signal)
    for cluster in spike_data_clusterized_oneline.columns:
        plt.plot(signal.index, spike_data_clusterized_oneline[cluster])
    plt.title('Every clusters on the signal of origin')
    plt.xlabel('Time in ms')
    plt.ylabel('Amplitude [µV]')
    plt.ylim(y_lim_min , y_lim_max)
    plt.legend(legend)
    plt.grid(True)


def print_clusters_3d(labels, X):
    """
    labels :    array_like - the cluster's labels of the differents spikes
    X :         ndarray - the position of each spike after the PCA
    
    This function print the spikes colored by cluster in the 3D space of the 3 most important directions of the PCA
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    nb_clusters = max(labels) + 1

    a = [i for i in range(len(labels))]
    b = np.transpose([a,list(labels)])

    for nb in range(nb_clusters):
        data = X[[x for x,y in b if y==nb],:]
        ax.scatter(data[:,0], data[:,1], data[:,2])
    data = X[[x for x,y in b if y==-1],:]
    ax.scatter(data[:,0], data[:,1], data[:,2], c='black')    
    
    ax.set_title('Nombre de cluster(s) :' + str(nb_clusters))
    plt.show()


def PCA_plot(PCA_X):
    """
    PCA_X : ndarray - the position of each spike after the PCA
    
    This function print the spikes in the 3D space of the 3 most important directions of the PCA
    """
    
    fig = plt.figure()
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    plt.cla()
  
    ax.scatter(PCA_X[:, 0], PCA_X[:, 1], PCA_X[:, 2], cmap=plt.cm.nipy_spectral,
           edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

    plt.show()
