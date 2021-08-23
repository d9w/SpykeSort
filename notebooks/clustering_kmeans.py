{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# KMEANS\n",
    "\n",
    "### 18/05/20"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Importing Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import all libraries needed for the tutorial\n",
    "\n",
    "# General syntax to import specific functions in a library: \n",
    "##from (library) import (specific library function)\n",
    "from pandas import DataFrame, read_csv\n",
    "\n",
    "# General syntax to import a library but no functions: \n",
    "##import (library) as (give the library a nickname/alias)\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd #this is how I usually import pandas\n",
    "import sys #only needed to determine Python version number\n",
    "import matplotlib #only needed to determine Matplotlib version number\n",
    "\n",
    "# Enable inline plotting\n",
    "%matplotlib inline\n",
    "\n",
    "import scipy\n",
    "import numpy as np\n",
    "#import scipy.signal as signal\n",
    "from scipy.signal import *\n",
    "import matplotlib.pyplot as plt\n",
    "from mpl_toolkits.mplot3d import Axes3D\n",
    "from sklearn.decomposition import PCA\n",
    "from sklearn.cluster import KMeans\n",
    "\n",
    "import sys  \n",
    "#sys.path.insert(0, '/Users/louiseplacidet/Desktop/PIR/GITPIR/GIT_29_04/PIR/AdabandFlt')\n",
    "#sys.path.insert(0, '/Users/SYL21/External_Drive/SUPAERO/PIR/AdabandFlt')\n",
    "sys.path.insert(0, '/Users/louiseplacidet/Desktop/PIR/GITPIR/GIT_29_04/PIR/AdabandFlt')\n",
    "#from AdaBandFlt import *\n",
    "from V2AdaBandFlt import *\n",
    "\n",
    "%matplotlib tk"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Loading Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "#load data\n",
    "# file path of csv file\n",
    "#Location = r'/Users/33614/ExternalDrive/SUPAERO/PIR_2A/Data/data_spikes/E18KABaseline_Bcut.txt'\n",
    "#Location = r'/Users/SYL21/D_Drive/SUPAERO/PIR_2A/Data/data_spikes/E18KABaseline_Bcut.txt'\n",
    "#Location = r'/Users/louiseplacidet/Desktop/PIR/Data/data_spikes/E18KABaseline_Bcut.txt'\n",
    "#Location = r'/Users/SYL21/External_Drive/SUPAERO/PIR/Data/Wetransfer_data/E18KABaseline_BcutV2groundAll.txt' ",
    "\n",
    "Location = r'/Users/louiseplacidet/Desktop/PIR/Data/new_spike_data/newdata/E18KABaseline_BcutV2groundAll.txt'\n",
    "\n",
    "# create dataframe\n",
    "df = pd.read_csv(Location, sep='\\t',skiprows=[0,1,3] , index_col='%t           ')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "#####################################################################################################################\n",
    "####  BANK OF PARTS OF DATA\n",
    "\n",
    "all_raw_data = df #Entire recording from all electrodes\n",
    "\n",
    "full_signal = df.iloc[:,1] #Entire recording from the first electrode\n",
    "\n",
    "def butter_bandpass(lowcut, highcut, fs, order=5):\n",
    "    nyq = 0.5 * fs\n",
    "    low = lowcut / nyq\n",
    "    high = highcut / nyq\n",
    "    b, a = butter(order, [low, high], btype='band')\n",
    "    return b, a\n",
    "\n",
    "\n",
    "def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):\n",
    "    b, a = butter_bandpass(lowcut, highcut, fs, order=order)\n",
    "    y = filtfilt(b, a, data)\n",
    "    return y\n",
    "\n",
    "# Sample rate and desired cutoff frequencies (in Hz).\n",
    "fs = 25000.0\n",
    "lowcut = 100.0\n",
    "highcut = 2500.0\n",
    "\n",
    "\n",
    "y = butter_bandpass_filter(df.iloc[:,1], lowcut, highcut, fs, order=6)\n",
    "\n",
    "\n",
    "filtereddf = pd.DataFrame(y)\n",
    "filtereddf.index = df.index\n",
    "\n",
    "signal_filtered = filtereddf.iloc[:,0] #Entire recording filtered by bandpass, for one electrode\n",
    "\n",
    "\n",
    "fs = 25000\n",
    "\n",
    "#xminnoise = int(np.round(11114*(fs/1000)))\n",
    "#xmaxnoise = int(np.round(18511*(fs/1000)))\n",
    "\n",
    "#noise_data = filtereddf.iloc[xminnoise:xmaxnoise,0]\n",
    "\n",
    "#xminspike = int(np.round(130826*(fs/1000)))\n",
    "#xmaxspike = int(np.round(131699*(fs/1000)))\n",
    "\n",
    "#burst_data = filtereddf.iloc[xminspike:xmaxspike,0]\n",
    "\n",
    "#begin_data = signal_filtered.iloc[:500000]\n",
    "\n",
    "########################################################\n",
    "y_ref = butter_bandpass_filter(df['El 15       '], lowcut, highcut, fs, order=6)\n",
    "\n",
    "filtereddf_ref = pd.DataFrame(y_ref)\n",
    "filtereddf_ref.index = df.index\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "################################################################################################\n",
    "####   TEST ADABANDFLT AVEC SIGNAL ORIGINAL (PASSE-BANDE+WIENER)\n",
    "\n",
    "# Choices:\n",
    "#  - full_signal : entire signal from first electrode\n",
    "#  - signal_filtered : entire signal from first electrode after bandpass\n",
    "#  - noise_data : part of signal where only noise (after bandpass)\n",
    "#  - spike_data : part of signal where burst (after bandpass)\n",
    "\n",
    "\n",
    "signal = signal_filtered\n",
    "signal_filtered_ref = filtereddf_ref.iloc[:,0] #Entire recording filtered by bandpass, for reference electrode"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Detecting and Aligning the Spikes"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Setting Up Thresholds adapted to Noise Levels"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "fs = 25000\n",
    "\n",
    "noise_levels = init_noise_levels(signal_filtered, fs, \n",
    "                      noise_window_size = 0.01,\n",
    "                      required_valid_windows = 100,\n",
    "                      old_noise_level_propagation = 0.8, \n",
    "                      test_level = 5,\n",
    "                      estimator_type = \"RMS\",\n",
    "                      percentile_value = 25)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, 'Noise Levels')"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "plt.figure()\n",
    "plt.plot(noise_levels)\n",
    "plt.grid(True)\n",
    "plt.xlabel('Time')\n",
    "plt.ylabel('Noise Amplitude [µV]')\n",
    "plt.title('Noise Levels')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Detecting Spikes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>indice_min</th>\n",
       "      <th>indice_1er_depass</th>\n",
       "      <th>indice_max_gauche</th>\n",
       "      <th>indice_max_droite</th>\n",
       "      <th>Delta_amplitudes</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>53016</td>\n",
       "      <td>53015</td>\n",
       "      <td>53007</td>\n",
       "      <td>nan</td>\n",
       "      <td>13.985842</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>53240</td>\n",
       "      <td>53239</td>\n",
       "      <td>nan</td>\n",
       "      <td>53266</td>\n",
       "      <td>13.827923</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>53286</td>\n",
       "      <td>53285</td>\n",
       "      <td>53278</td>\n",
       "      <td>nan</td>\n",
       "      <td>16.236913</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>53327</td>\n",
       "      <td>53325</td>\n",
       "      <td>nan</td>\n",
       "      <td>53352</td>\n",
       "      <td>17.075408</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>53375</td>\n",
       "      <td>53373</td>\n",
       "      <td>53352</td>\n",
       "      <td>53391</td>\n",
       "      <td>18.401228</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>660</th>\n",
       "      <td>478481</td>\n",
       "      <td>478478</td>\n",
       "      <td>nan</td>\n",
       "      <td>478501</td>\n",
       "      <td>17.741349</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>661</th>\n",
       "      <td>478578</td>\n",
       "      <td>478574</td>\n",
       "      <td>478542</td>\n",
       "      <td>nan</td>\n",
       "      <td>18.384711</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>662</th>\n",
       "      <td>478788</td>\n",
       "      <td>478788</td>\n",
       "      <td>478774</td>\n",
       "      <td>nan</td>\n",
       "      <td>14.207572</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>663</th>\n",
       "      <td>478927</td>\n",
       "      <td>478926</td>\n",
       "      <td>478909</td>\n",
       "      <td>nan</td>\n",
       "      <td>12.559793</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>664</th>\n",
       "      <td>512271</td>\n",
       "      <td>512270</td>\n",
       "      <td>nan</td>\n",
       "      <td>512303</td>\n",
       "      <td>13.494181</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>665 rows × 5 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     indice_min  indice_1er_depass indice_max_gauche indice_max_droite  \\\n",
       "0         53016              53015             53007               nan   \n",
       "1         53240              53239               nan             53266   \n",
       "2         53286              53285             53278               nan   \n",
       "3         53327              53325               nan             53352   \n",
       "4         53375              53373             53352             53391   \n",
       "..          ...                ...               ...               ...   \n",
       "660      478481             478478               nan            478501   \n",
       "661      478578             478574            478542               nan   \n",
       "662      478788             478788            478774               nan   \n",
       "663      478927             478926            478909               nan   \n",
       "664      512271             512270               nan            512303   \n",
       "\n",
       "     Delta_amplitudes  \n",
       "0           13.985842  \n",
       "1           13.827923  \n",
       "2           16.236913  \n",
       "3           17.075408  \n",
       "4           18.401228  \n",
       "..                ...  \n",
       "660         17.741349  \n",
       "661         18.384711  \n",
       "662         14.207572  \n",
       "663         12.559793  \n",
       "664         13.494181  \n",
       "\n",
       "[665 rows x 5 columns]"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "spike_info = find_spikes(signal, noise_levels, fs, \n",
    "                           window_size = 0.0012, \n",
    "                           noise_window_size = 0.01,\n",
    "                           threshold_factor = 3.2,\n",
    "                           maxseparation = 0.0015)\n",
    "\n",
    "#spike_info.columns = ['indice_max','indice_min','indice_depass_positif','indice_depass_negatif', 'indice_1er_depass','indice_zero_central','i_max-i_min','Delta_amplitudes']\n",
    "spike_info"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Recording the spikes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(666, 77)\n"
     ]
    }
   ],
   "source": [
    "spike_data = record_spikes(signal, fs, spike_info,\n",
    "                           'indice_1er_depass',\n",
    "                           t_before = 0.0015, \n",
    "                           t_after = 0.0015)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "spike_data_oneline = record_spikes_oneline(signal, fs, spike_info,\n",
    "                                           'indice_1er_depass',\n",
    "                                           t_before = 0.0015,\n",
    "                                           t_after = 0.0015)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Plotting the spikes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "No handles with labels found to put in legend.\n"
     ]
    }
   ],
   "source": [
    "#signal.plot(color = 'blue')\n",
    "#spike_data_oneline.plot(color = 'red')\n",
    "plt.plot(df.index, signal, color = 'blue')\n",
    "plt.plot(spike_data_oneline.index, spike_data_oneline, color = 'red')\n",
    "plt.title('Filtered Signal with Detected Spikes with RMS')\n",
    "plt.xlabel('Time Windows')\n",
    "plt.ylabel('Amplitude [µV]')\n",
    "plt.legend()\n",
    "plt.grid(True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "print_spikes(spike_data,\n",
    "             t_before_alignement = 0.0015,\n",
    "             first_spike = 1,\n",
    "             last_spike = 50,\n",
    "             fs = 25000)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Bilan PCA + KMEANS"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## PCA and KMEANS on spikes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "def PCA_and_KMEANS_spikes(spike_data, spike_info, nb_PCA_components=3, nb_clusters=3):\n",
    "    \n",
    "    ## PCA\n",
    "    pca_data = np.array(spike_data.iloc[:,1:].values).transpose()\n",
    "    pca = PCA(n_components=nb_PCA_components)\n",
    "    pca.fit(pca_data)\n",
    "    PCA_X = pca.transform(pca_data)\n",
    "    \n",
    "    ## KMEANS\n",
    "    kmeans = KMeans(n_clusters=nb_clusters, random_state=0).fit(PCA_X)\n",
    "\n",
    "    \n",
    "    ## Ajout du label des clusters dans spike info\n",
    "    spike_info['cluster_label'] = kmeans.labels_\n",
    "    \n",
    "    return PCA_X, kmeans, spike_info\n",
    "    "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Plotting the PCA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "## Fonction qui plot la PCA\n",
    "\n",
    "def PCA_plot(PCA_X):\n",
    "    \n",
    "    fig = plt.figure(4,figsize=(4,3))\n",
    "    plt.clf()\n",
    "    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)\n",
    "    plt.cla()\n",
    "  \n",
    "    ax.scatter(PCA_X[:, 0], PCA_X[:, 1], PCA_X[:, 2], cmap=plt.cm.nipy_spectral,\n",
    "           edgecolor='k')\n",
    "\n",
    "    ax.w_xaxis.set_ticklabels([])\n",
    "    ax.w_yaxis.set_ticklabels([])\n",
    "    ax.w_zaxis.set_ticklabels([])\n",
    "\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "## Importance des premiers coefficients de la PCA selon les différents indices d'alignement\n",
    "\n",
    "def print_pca_explained_variance_ratio(signal, fs, spike_infos):\n",
    "    legend = ['indice_max',\n",
    "     'indice_min',\n",
    "     'indice_depass_positif',\n",
    "     'indice_depass_negatif',\n",
    "     'indice_1er_depass',\n",
    "     'indice_zero_central']\n",
    "    explained_variance_ratio = []\n",
    "    \n",
    "    for method in legend:\n",
    "        spike_data = record_spikes(signal, fs, spike_infos,\n",
    "                  method,\n",
    "                  t_before = 0.001,\n",
    "                  t_after = 0.002)\n",
    "        pca_data = np.array(spike_data.iloc[:,1:].values).transpose()\n",
    "        pca = PCA(n_components=10)\n",
    "        pca.fit(pca_data)\n",
    "        X = pca.transform(pca_data)\n",
    "        explained_variance_ratio.append(pca.explained_variance_ratio_)\n",
    "        \n",
    "    plt.figure()\n",
    "    plt.plot(np.transpose(explained_variance_ratio))\n",
    "    plt.legend(legend)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Plotting the KMEANS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "## Fonction qui plot le KMEANS\n",
    "\n",
    "def print_clusters_3d(labels, PCA_X):\n",
    "    fig = plt.figure()\n",
    "    ax = fig.add_subplot(111, projection='3d')\n",
    "\n",
    "    nb_clusters = max(labels) + 1\n",
    "\n",
    "    a = [i for i in range(len(labels))]\n",
    "    b = np.transpose([a,list(labels)])\n",
    "\n",
    "    for nb in range(nb_clusters):\n",
    "        legend = 'Cluster n°'+str(nb)\n",
    "        data = PCA_X[[x for x,y in b if y==nb],:]\n",
    "        ax.scatter(data[:,0], data[:,1], data[:,2],label=legend)\n",
    "    data = PCA_X[[x for x,y in b if y==-1],:]\n",
    "    ax.scatter(data[:,0], data[:,1], data[:,2], c='black',label='Noise cluster')    \n",
    "    \n",
    "    ax.set_title('Nombre de cluster(s) :' + str(nb_clusters))\n",
    "    ax.legend()\n",
    "    plt.show()\n",
    "    \n",
    "#print_clusters_3d(kmeans.labels_, X)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Plotting the spikes from the different clusters after KMEANS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "def print_spikes_clusterized(spike_data,\n",
    "                             labels,\n",
    "                             t_before_alignement = 0,\n",
    "                             nb_spike = 20,\n",
    "                             y_lim_min = -50,\n",
    "                             y_lim_max = 60,\n",
    "                             fs = 25000):\n",
    "    \n",
    "    nb_clusters = max(labels) + 1\n",
    "        \n",
    "    if (-1 in labels) == True:\n",
    "        nb_clusters_ = nb_clusters + 1\n",
    "    else:\n",
    "        nb_clusters_ = nb_clusters\n",
    "    \n",
    "    nb_line = nb_clusters_//2\n",
    "    if nb_clusters_%2 != 0:\n",
    "        nb_line += 1\n",
    "    \n",
    "    #spike_data.iloc[:,first_spike:last_spike].plot()\n",
    "    t_b = int(np.round(fs*(t_before_alignement)))\n",
    "    y = (spike_data.iloc[:,0]-t_b)*1000/fs\n",
    "        \n",
    "    a = [i+1 for i in range(len(labels))]\n",
    "    b = np.transpose([a,list(labels)])\n",
    "    \n",
    "    figure = plt.figure()\n",
    "    plt.gcf().subplots_adjust(left = 0.1, bottom = 0.1, right = 0.9, top = 0.9, wspace = 0.2, hspace = 0.5)\n",
    "    for nb in range(nb_clusters):\n",
    "        data = spike_data.iloc[:,[x for x,y in b if y==nb]]\n",
    "        m = len(data.values[0])\n",
    "        nb_spikes_in_cluster = len(data.columns)\n",
    "\n",
    "        \n",
    "        kept = []\n",
    "        \n",
    "        if m <= nb_spike:\n",
    "            kept = [i for i in range(m)]\n",
    "        else:      \n",
    "            i = 0  \n",
    "            while i < nb_spike:\n",
    "                r = randint(0,m-1)\n",
    "                if (r in kept) == False:\n",
    "                    kept.append(r)\n",
    "                    i += 1\n",
    "        \n",
    "        x = data.iloc[:,kept].values\n",
    "        axes = figure.add_subplot(nb_line, 2, nb+1)\n",
    "        axes.plot(y, x)\n",
    "        axes.set_xlabel('Time in ms')\n",
    "        axes.set_title('Cluster numero ' + str(nb))\n",
    "        axes.set_ylim(y_lim_min , y_lim_max)\n",
    "        axes.grid(True)\n",
    "        axes.legend([str(nb_spikes_in_cluster)+' spikes'])\n",
    "        \n",
    "    if (-1 in labels) == True:\n",
    "        data = spike_data.iloc[:,[x for x,y in b if y==-1]]\n",
    "        m = len(data.values[0])\n",
    "        nb_spikes_in_cluster = len(data.columns)\n",
    "    \n",
    "        kept = []\n",
    "        \n",
    "        if m <= nb_spike:\n",
    "            kept = [i for i in range(m)]\n",
    "        else:      \n",
    "            i = 0  \n",
    "            while i < nb_spike:\n",
    "                r = randint(0,m-1)\n",
    "                if (r in kept) == False:\n",
    "                    kept.append(r)\n",
    "                    i += 1\n",
    "        \n",
    "        x = data.iloc[:,kept].values\n",
    "        \n",
    "        axes = figure.add_subplot(nb_line, 2, nb+2)\n",
    "        axes.plot(y, x)\n",
    "        axes.set_xlabel('Time in ms')\n",
    "        axes.set_title('Cluster de bruit')\n",
    "        axes.set_ylim(y_lim_min , y_lim_max)\n",
    "        axes.grid(True)\n",
    "        axes.legend([str(nb_spikes_in_cluster)+' spikes'])\n",
    "        \n",
    "#print_spikes_clusterized(spike_data,\n",
    "#                             dbscan.labels_,\n",
    "#                             t_before_alignement = 0.0015,\n",
    "#                             nb_spike = 20,\n",
    "#                             y_lim_min = -50,\n",
    "#                             y_lim_max = 60,\n",
    "#                             fs = 25000)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Printing the spikes from clusters on original signal"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "## Fonction qui affiche les spikes des différents clusters sur échelle temporelle\n",
    "\n",
    "def print_spikes_clusterized_oneline(signal,\n",
    "                                     updated_spike_infos,\n",
    "                                     align_method = 'indice_zero_central',\n",
    "                                     t_before = 0.001,\n",
    "                                     t_after = 0.002,\n",
    "                                     fs = 25000,\n",
    "                                     y_lim_min = -50,\n",
    "                                     y_lim_max = 60,\n",
    "                                     separate_plot = False):\n",
    "    \n",
    "    labels = updated_spike_infos['cluster_label'].values\n",
    "    nb_clusters = max(labels) + 1\n",
    "    \n",
    "    if (-1 in labels) == True:\n",
    "        nb_clusters_ = nb_clusters + 2\n",
    "    else:\n",
    "        nb_clusters_ = nb_clusters + 1\n",
    "    \n",
    "    nb_line = nb_clusters_//2\n",
    "    if nb_clusters_%2 != 0:\n",
    "        nb_line += 1\n",
    "    \n",
    "    ghost_array = np.array(['NaN' for x in range(len(signal))])\n",
    "    ghost_array = ghost_array.astype(float)\n",
    "    spike_data_clusterized_oneline = []\n",
    "    legend = ['Signald\\'origine']\n",
    "    \n",
    "    figure = plt.figure()\n",
    "    plt.gcf().subplots_adjust(left = 0.1, bottom = 0.1, right = 0.9, top = 0.9, wspace = 0.2, hspace = 0.5)\n",
    "    for nb in range(nb_clusters):\n",
    "        \n",
    "        spike_data_oneline = record_spikes_oneline(signal,\n",
    "                              fs,\n",
    "                              updated_spike_infos.loc[updated_spike_infos['cluster_label'] == nb],\n",
    "                              align_method,\n",
    "                              t_before = t_before,\n",
    "                              t_after = t_after)\n",
    "        spike_data_oneline = np.resize(spike_data_oneline.values,len(spike_data_oneline.values))\n",
    "        spike_data_clusterized_oneline.append(spike_data_oneline)\n",
    "        \n",
    "        \n",
    "        axes = figure.add_subplot(nb_line, 2, nb+1)\n",
    "        axes.plot(signal.index, signal.values)\n",
    "        for ghost in range(nb):\n",
    "            axes.plot(signal.index, ghost_array)\n",
    "        axes.plot(signal.index, spike_data_oneline)\n",
    "        axes.set_xlabel('Time in ms')\n",
    "        axes.set_title('Cluster n°' + str(nb))\n",
    "        axes.set_ylim(y_lim_min , y_lim_max)\n",
    "        axes.grid()\n",
    "        \n",
    "        legend.append('Cluster n°' + str(nb))\n",
    "        \n",
    "    if (-1 in labels) == True:\n",
    "        spike_data_oneline = record_spikes_oneline(signal,\n",
    "                              fs,\n",
    "                              updated_spike_infos.loc[updated_spike_infos['cluster_label'] == -1],\n",
    "                              align_method,\n",
    "                              t_before = t_before,\n",
    "                              t_after = t_after)\n",
    "        spike_data_oneline = np.resize(spike_data_oneline.values,len(spike_data_oneline.values))\n",
    "        spike_data_clusterized_oneline.append(spike_data_oneline)\n",
    "        \n",
    "        \n",
    "        axes = figure.add_subplot(nb_line, 2, nb+2)\n",
    "        axes.plot(signal.index, signal.values)\n",
    "        axes.plot(signal.index, spike_data_oneline)\n",
    "        axes.set_xlabel('Time in ms')\n",
    "        axes.set_title('Cluster de bruit')\n",
    "        axes.set_ylim(y_lim_min , y_lim_max)\n",
    "        axes.grid()\n",
    "                \n",
    "        legend.append('Cluster de bruit')\n",
    "        \n",
    "        axes = figure.add_subplot(nb_line, 2, nb+3)\n",
    "    else:\n",
    "        axes = figure.add_subplot(nb_line, 2, nb+2)\n",
    "    \n",
    "    axes.plot(signal.index, signal.values)\n",
    "    axes.plot(signal.index, np.transpose(spike_data_clusterized_oneline))\n",
    "    axes.legend(legend)\n",
    "    axes.set_xlabel('Time in ms')\n",
    "    axes.set_title('Tous les clusters')\n",
    "    axes.set_ylim(y_lim_min , y_lim_max)\n",
    "    axes.grid()\n",
    "        \n",
    "        \n",
    "        \n",
    "        \n",
    "\n",
    "#print_spikes_clusterized_oneline(signal,\n",
    "#                                 updated_spike_infos,\n",
    "#                                 align_method = 'indice_zero_central',\n",
    "#                                 t_before = 0.001,\n",
    "#                                 t_after = 0.002,\n",
    "#                                 fs = 25000,)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "## Fonction qui affiche les spikes des différents clusters sur échelle temporelle\n",
    "\n",
    "def print_spikes_clusterized_oneline_total(signal,\n",
    "                                     updated_spike_infos,\n",
    "                                     align_method = 'indice_zero_central',\n",
    "                                     t_before = 0.001,\n",
    "                                     t_after = 0.002,\n",
    "                                     fs = 25000,\n",
    "                                     y_lim_min = -50,\n",
    "                                     y_lim_max = 60,\n",
    "                                     separate_plot = False):\n",
    "    \n",
    "    labels = updated_spike_infos['cluster_label'].values\n",
    "    nb_clusters = max(labels) + 1\n",
    "    \n",
    "    if (-1 in labels) == True:\n",
    "        nb_clusters_ = nb_clusters + 2\n",
    "    else:\n",
    "        nb_clusters_ = nb_clusters + 1\n",
    "    \n",
    "    nb_line = nb_clusters_//2\n",
    "    if nb_clusters_%2 != 0:\n",
    "        nb_line += 1\n",
    "    \n",
    "    ghost_array = np.array(['NaN' for x in range(len(signal))])\n",
    "    ghost_array = ghost_array.astype(float)\n",
    "    spike_data_clusterized_oneline = []\n",
    "    legend = ['Signald\\'origine']\n",
    "    \n",
    "    figure = plt.figure()\n",
    "    plt.gcf().subplots_adjust(left = 0.1, bottom = 0.1, right = 0.9, top = 0.9, wspace = 0.2, hspace = 0.5)\n",
    "    for nb in range(nb_clusters):\n",
    "        \n",
    "        spike_data_oneline = record_spikes_oneline(signal,\n",
    "                              fs,\n",
    "                              updated_spike_infos.loc[updated_spike_infos['cluster_label'] == nb],\n",
    "                              align_method,\n",
    "                              t_before = t_before,\n",
    "                              t_after = t_after)\n",
    "        spike_data_oneline = np.resize(spike_data_oneline.values,len(spike_data_oneline.values))\n",
    "        spike_data_clusterized_oneline.append(spike_data_oneline)\n",
    "        \n",
    "        legend.append('Cluster n°' + str(nb))\n",
    "\n",
    "        \n",
    "    if (-1 in labels) == True:\n",
    "        spike_data_oneline = record_spikes_oneline(signal,\n",
    "                              fs,\n",
    "                              updated_spike_infos.loc[updated_spike_infos['cluster_label'] == -1],\n",
    "                              align_method,\n",
    "                              t_before = t_before,\n",
    "                              t_after = t_after)\n",
    "        spike_data_oneline = np.resize(spike_data_oneline.values,len(spike_data_oneline.values))\n",
    "        spike_data_clusterized_oneline.append(spike_data_oneline)\n",
    "        legend.append('Cluster de bruit')\n",
    "    \n",
    "    axes = figure.add_subplot(1, 1, 1)\n",
    "    axes.plot(signal.index, signal.values)\n",
    "    axes.plot(signal.index, np.transpose(spike_data_clusterized_oneline))\n",
    "    axes.legend(legend)\n",
    "    axes.set_xlabel('Time in ms')\n",
    "    axes.set_title('Tous les clusters')\n",
    "    axes.set_ylim(y_lim_min , y_lim_max)\n",
    "    axes.grid()\n",
    "        \n",
    "        \n",
    "        \n",
    "        \n",
    "\n",
    "#print_spikes_clusterized_oneline_total(signal,\n",
    "#                                 updated_spike_infos,\n",
    "#                                 align_method = 'indice_zero_central',\n",
    "#                                 t_before = 0.001,\n",
    "#                                 t_after = 0.002,\n",
    "#                                 fs = 25000,)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Histogram of Amplitudes and Time"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "## Histogramme de différences d'amplitude Peak-to-Peak des spikes\n",
    "\n",
    "def histogram_spikes_amplitude(spike_info):\n",
    "    plt.figure()\n",
    "    plt.hist(spike_info['Delta_amplitudes'], bins='auto')\n",
    "    plt.title(\"Histogram of Spike Peak-to-Peak Amplitude\")\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "## Histogramme de l'écartement temporel des spikes (imax-imin)\n",
    "\n",
    "#def histogram_spikes_time(spike_info):\n",
    "#    plt.figure()\n",
    "#    plt.hist(spike_info['i_max-i_min'], bins='auto')\n",
    "#    plt.title(\"Histogram of Spike i_max-i_min\")\n",
    "#    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "## Histogramme des différences d'amplitude Peak-to-Peak des spikes:\n",
    "\n",
    "## ¡¡¡ATTENTION!!! Il faut que le spike info en entrée, soit mis à jour avec les labels des clusters\n",
    "\n",
    "def histogram_amplitude_clusterized(updated_spike_info):\n",
    "    \n",
    "    labels = updated_spike_info['cluster_label'].values\n",
    "    nb_clusters = max(labels) + 1\n",
    "    \n",
    "    for nb in range(nb_clusters):\n",
    "        local_info = spike_info.loc[spike_info['cluster_label'] == nb]\n",
    "        plt.figure()\n",
    "        plt.hist(local_info['Delta_amplitudes'], bins='auto')\n",
    "        plt.title(\"Histogram of Spike peak-to-peak of cluster n°\"+str(nb))\n",
    "        plt.show()\n",
    "    \n",
    "    if(-1 in labels) == True:\n",
    "        local_info = spike_info.loc[spike_info['cluster_label']==-1]\n",
    "        plt.figure()\n",
    "        plt.hist(local_info['Delta_amplitudes'], bins='auto')\n",
    "        plt.title(\"Histogram of Spike peak-to-peak of cluster of noise\")\n",
    "        plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "## Histogramme des différences d'amplitude Peak-to-Peak des spikes:\n",
    "\n",
    "## ¡¡¡ATTENTION!!! Il faut que le spike info en entrée, soit mis à jour avec les labels des clusters\n",
    "\n",
    "def histogram_amplitude_clusterized_superposed(updated_spike_info):\n",
    "    \n",
    "    labels = updated_spike_info['cluster_label'].values\n",
    "    nb_clusters = max(labels) + 1\n",
    "    \n",
    "    for nb in range(nb_clusters,0,-1):\n",
    "        local_info = spike_info.loc[spike_info['cluster_label'] == nb]\n",
    "        plt.hist(local_info['Delta_amplitudes'], bins='auto')\n",
    "        plt.title(\"Superposition of histogram of Spike peak-to-peak of clusters\")\n",
    "        plt.show()\n",
    "    \n",
    "    if(-1 in labels) == True:\n",
    "        local_info = spike_info.loc[spike_info['cluster_label']==-1]\n",
    "        plt.hist(local_info['Delta_amplitudes'], bins='auto')\n",
    "        plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Tests des fonctions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function: PCA_and_KMEANS_spikes(spike_data, spike_info, nb_PCA_components=3, nb_clusters=3)\n",
    "\n",
    "PCA_X, KMEANS, updated_spike_info = PCA_and_KMEANS_spikes(spike_data, spike_info, 5, 4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function: PCA_plot(spike_data, nb_clusters=3)\n",
    "\n",
    "PCA_plot(PCA_X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function: print_clusters_3d(labels, PCA_X)\n",
    "\n",
    "print_clusters_3d(KMEANS.labels_, PCA_X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function: print_spikes_clusterized(spike_data,\n",
    "#                             labels,\n",
    "#                             t_before_alignement = 0,\n",
    "#                             nb_spike = 20,\n",
    "#                             y_lim_min = -50,\n",
    "#                             y_lim_max = 60,\n",
    "#                             fs = 25000)\n",
    "                        \n",
    "print_spikes_clusterized(spike_data,\n",
    "                             KMEANS.labels_,\n",
    "                             t_before_alignement = 0,\n",
    "                             nb_spike = 20,\n",
    "                             y_lim_min = -50,\n",
    "                             y_lim_max = 60,\n",
    "                             fs = 25000)                        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function: print_spikes_clusterized_oneline(signal,\n",
    "#                                 updated_spike_infos,\n",
    "#                                 align_method = 'indice_zero_central',\n",
    "#                                 t_before = 0.001,\n",
    "#                                 t_after = 0.002,\n",
    "#                                 fs = 25000,)\n",
    "\n",
    "print_spikes_clusterized_oneline(signal,\n",
    "                                 updated_spike_info,\n",
    "                                 align_method = 'indice_1er_depass',\n",
    "                                 t_before = 0.0015,\n",
    "                                 t_after = 0.0015,\n",
    "                                 fs = 25000,)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function: print_spikes_clusterized_oneline_total(signal,\n",
    "#                                 updated_spike_infos,\n",
    "#                                 align_method = 'indice_zero_central',\n",
    "#                                 t_before = 0.001,\n",
    "#                                 t_after = 0.002,\n",
    "#                                 fs = 25000,)\n",
    "\n",
    "print_spikes_clusterized_oneline_total(signal,\n",
    "                                 updated_spike_info,\n",
    "                                 align_method = 'indice_1er_depass',\n",
    "                                 t_before = 0.001,\n",
    "                                 t_after = 0.002,\n",
    "                                 fs = 25000,)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function: histogram_spikes_amplitude(spike_info)\n",
    "\n",
    "histogram_spikes_amplitude(spike_info)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function: histogram_spikes_time(spike_info)\n",
    "\n",
    "#histogram_spikes_time(spike_info)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/opt/anaconda3/lib/python3.7/tkinter/__init__.py:749: UserWarning: Creating legend with loc=\"best\" can be slow with large amounts of data.\n",
      "  func(*args)\n",
      "/opt/anaconda3/lib/python3.7/tkinter/__init__.py:1705: UserWarning: Creating legend with loc=\"best\" can be slow with large amounts of data.\n",
      "  return self.func(*args)\n"
     ]
    }
   ],
   "source": [
    "# Function: histogram_amplitude_clusterized(spike_info)\n",
    "\n",
    "histogram_amplitude_clusterized(updated_spike_info)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function: histogram_amplitude_clusterized_superposed(updated_spike_info)\n",
    "\n",
    "histogram_amplitude_clusterized_superposed(updated_spike_info)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print_pca_explained_variance_ratio(signal, fs, spike_infos)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
