import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

try:
    from sklearn.cluster import OPTICS
except Exception as e:
    print("Erreur dans l'importation d'OPTICS")



def PCA_and_AGGLOCLUST_spikes(spike_data, spike_info, nb_PCA_components=3,
                              n_clusters=5, metric='euclidean', linkage='ward'):
    
    ## PCA
    pca_data = np.array(spike_data.values).transpose()
    pca = PCA(n_components=nb_PCA_components)
    pca.fit(pca_data)
    PCA_X = pca.transform(pca_data)
    
    ## AGGLOMERATIVE CLUSTERING
    ## Different linkages: 'ward', 'average', 'complete', 'single'
    
    aggloclustering = AgglomerativeClustering(n_clusters=n_clusters, affinity = metric,
                                    linkage=linkage)
    aggloclustering.fit(PCA_X)
    
    labels = aggloclustering.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    ## Ajout du label des clusters dans spike info
    spike_info['cluster_label'] = aggloclustering.labels_
    
    return PCA_X, aggloclustering, spike_info


def PCA_and_OPTICS_spikes(spike_data, spike_info, nb_PCA_components=3, min_samples=5, max_eps=10, xi=0.05,min_cluster_size=5):
    
    ## PCA
    pca_data = np.array(spike_data.values).transpose()
    pca = PCA(n_components=nb_PCA_components)
    pca.fit(pca_data)
    PCA_X = pca.transform(pca_data)
    
    ## OPTICS
    
    optics = OPTICS(min_samples=min_samples, max_eps=max_eps, xi=0.05,min_cluster_size=min_cluster_size).fit(PCA_X)

    #core_samples_mask = np.zeros_like(optics.labels_, dtype=bool)
    #core_samples_mask[optics.core_sample_indices_] = True
    labels = optics.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print('')
    print('Number of spikes detected by AdaBandFlt: %d' % len(spike_info))
    print('Spikes placed into the clusters: %d' % (len(spike_info)-n_noise_))
    print('Percentage placed: %d ' % ((len(spike_info)-n_noise_)/len(spike_info)*100))
    
    ## Ajout du label des clusters dans spike info
    spike_info['cluster_label'] = optics.labels_
    
    return PCA_X, optics, spike_info


def PCA_and_KMEANS_spikes(spike_data, spike_info, nb_PCA_components=3, n_clusters=4, random_state=0):
    
    ## PCA
    pca_data = np.array(spike_data.values).transpose()
    pca = PCA(n_components=nb_PCA_components)
    pca.fit(pca_data)
    PCA_X = pca.transform(pca_data)
    
    ## OPTICS
    
    kmeans = KMeans(n_clusters = n_clusters, random_state = random_state).fit(PCA_X)

    #core_samples_mask = np.zeros_like(optics.labels_, dtype=bool)
    #core_samples_mask[optics.core_sample_indices_] = True
    labels = kmeans.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    ## Ajout du label des clusters dans spike info
    spike_info['cluster_label'] = labels
    
    return PCA_X, kmeans, spike_info