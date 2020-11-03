import hyperspy.api as hs
import sys, os
import numpy as np
import pandas as pd
import matplotlib as mpl
import sklearn
from matplotlib import pyplot as plt
from matplotlib import cm
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap
from matplotlib_scalebar.scalebar import ScaleBar
from sklearn.preprocessing import normalize as norms

def load_spectrum_image(file_name):
    '''
    This function loads a dm3/dm4 spectrum image. 
    
    -file_name: string. 
        Name of the file containing the dm3/dm4 spectrum image.
    
    Returns:
    --------
    Hyperspy spectrum image of shape (x_lenght,y_lenght,n_channels).
    '''
    spectra = hs.load(file_name)

    return spectra

def kmeans_clustering(spectra,n_clusters,pca=False,n_comp=8):
    '''
    This function applies clustering via kmeans to Digital Micrograph spectrum images. 
    
    -spectra: hyperspy spectrum images of shape (x_lenght,y_lenght,n_channels).
    -n_clusters = int. 
        Number of clusters choose in the kmeans clustering routine.
    -pca = Boolean. (default = False)
        True, it applies PCA to clean the spectra.
        False, it doesn't apply the PCA clean. 
    -n_comp = int. (default = 8)
        Number of principal components selected from the PCA routine.

    Returns:
    --------
    labels: array like of shape (x_lenght,y_lenght). 
    	Label per each spectrum in the spectrum image. 
    centres: array like of shape (n_clusters,n_channels)
    	Centroids of each cluster identified in the spectrum image. 
    '''
    if pca:
        spectra.decomposition(True)
        sPCA = spectra.get_decomposition_model(n_comp)
        print('PCA applied.')
    else:
        sPCA = spectra
        print('No PCA applied.')
        
    sclust = sPCA.deepcopy()
    sclust.unfold()
    norms(sclust,axis=1,copy=False)
    kmeans = KMeans(n_clusters=n_clusters,tol=1e-9,max_iter=700)
    fitted = kmeans.fit(sclust)
    centres = fitted.cluster_centers_
    labels = fitted.labels_.reshape(spectra.data.shape[:-1])
    sclust.fold()
    
    return labels, centres


def preprocesing_spectrum_image(spectrum_image,labels,indx_lbl,ax,bx,a_bck,b_bck,pca=False,n_comp=8):
    '''
    This function extracts an array of spectra from an initial spectrum images which is identified by the kmeans clustering. 
    You have to provide the value of the cluster in which you are interested to get the spectra. 

    spectrum_image: hyperspy spectrum image of shape (x_side,y_side,n_channels).
        Spectrum image. 
    labels: array like of shape (x_side,y_side).
        Labels of kemans clustering. 
    indx_lbl: int.
        Value of the label in which spectra we are interested. 
    ax: float. 
        Initial energy value to crop the energy axis of the spectra.
    bx: float.
        Ending energy value to crop the energy axis of the spectra.
    a_bck: float.
        Initial energy value used to apply a Power law fit and remove the background. 
    b_bck: float.
        Ending energy value used to apply a Power law fit and remove the background. 
    pca: boolean (default = False). 
        If True, it applies PCA to clean the spectra.
        If False, it does not apply the PCA clean. 
    n_comp: int (default = 8). 
        Number of principal components selected from the PCA routine.

    Returns: 
    --------
    end_spectra: array-like of shape (n_spectra,r_channels)
        n_spectra is the number of spectra resulting from the spectrum image. 
        r_channels is the number of channels chose from the (ax,bx) crop. 
    '''
    spec_norm,aux_spec,end_spectra = [], [], []
    spectrum = spectrum_image.deepcopy()
    
    if pca:
        sPCA = spectrum.get_decomposition_model(n_comp)
    else: 
        sPCA = spectrum
    
    #Crop the spectra:
    spec_reduced = sPCA.isig[ax:bx]
    #Remove background:
    spec_rmv_bck = spec_reduced.remove_background([a_bck,b_bck],fast=False)
    
    #Choose the cluster in which we are interested: 
    spectra = spec_rmv_bck.data[labels == indx_lbl]
    
    #Energy resolution of the spectrum image spectra. 
    resol_energy = spectrum.axes_manager[-1].scale
    
    #Normalize all the spectra:
    for i in range(0,spectra.shape[0]):
        spec_norm.append(renorm(spectra[i]))

    #Interpolate the spectra: 
    new_energy_axis = np.arange(ax,bx,step=.1)
    old_energy_axis = spectrum_image.isig[ax:bx].axes_manager[-1].axis
    
    spec_norm = np.array(spec_norm)
    
    print(new_energy_axis.shape,old_energy_axis.shape,spec_norm.shape)
    
    for i in range(0,spec_norm.shape[0]):
        end_spectra.append(np.interp(new_energy_axis,old_energy_axis,spec_norm[i]))
    
    end_spectra = np.array(end_spectra)
    return end_spectra


def renorm(spectrum):
    '''
    This funtion normalizes an spectrum beetween 0 and 1. 

    spectrum = array-like of shape (n_features)
        Spectrum to be normalized. 

    Returns:
    --------
    Normalized spectrum between 0 and 1. 
    '''
    norm_spectrum = (spectrum-spectrum.min())/(spectrum.max()-spectrum.min())
    
    return norm_spectrum


def apply_mask(spectrum_image,labels,mask_label):
    '''
    This functions applies a mask under the "mask_label" value. 
    
    spectrum_image: hyperspy spectrum image of shape (x_side,y_side,n_channels).
        Spectrum image. 
    labels: array like of shape (x_side,y_side).
        Labels achieved by the kmeans clustering. 
    mask_label: int. 
        Value of the label which we mask.

    Returns:
    --------
    Returns an spectrum image with 0s at the masked regions. 
    '''

    mask = spectrum_image.deepcopy()
    
    masked_spectra = np.full((spectrum_image.data.shape[0],spectrum_image.data.shape[1]),False)
    max_lbl = np.unique(labels).max()
    
    for i in range(0,max_lbl):
        if i == mask_label:
            masked_spectra[labels==i] = False # Masked element.
        else:
            masked_spectra[labels==i] = True #Conserved elements.
            
    mask.data[masked_spectra == False] = 0 #Add 0 in the masked region.
    
    return mask

def plot_clustering(labels,centroids,cm,pix_size=1e-9,fig_save=None):
    '''
    This function plots in a colormap the results obtained by the kmeans clustering. 

    labels: array-like of shape (x_side,y_side).
        Labels for each spectrum in the spectrum image.
    centroids: array-like of shape (n_clusters,n_channels)
        Centroid for each cluster.
    cm: ListedColormap.
        List of colors for the colormap. 
        i.e. cm = mpl.colors.ListedColormap(['black','red','green'])
    pix_size: float. (default = 1nm)
        Spatial resolution in meters of the spectrum image. 

    Return:
    -------
    Returns the colormap built by the labels provided.  
    '''
    n_max_lbl = np.unique(labels).max()
    
    plt.rc('font', family='serif')
    fig = plt.figure(figsize=(8,8),dpi = 200)
 
    plt.subplot(211)
    plt.title('Color map:')    
    currentAxis = plt.gca()
    plt.axis('off')
    scalebar = ScaleBar(pix_size)
    plt.imshow(labels,cm)
    scalebar.color = 'white'
    scalebar.box_alpha = 0.0
    scalebar.location = 'lower right'
    currentAxis.add_artist(scalebar)
    
    plt.subplot(212)
    plt.title('Centroids:')
    for i in range(0,centroids.shape[0]):
        plt.plot(centroids[i], label = str(i), color = cm(i) )
    plt.legend(title='Numerical label of each cluster:')
    
    plt.show()
    
    if fig_save is not None:
        fig.savefig(fig_save,bbox_inches = 'tight')
        print('Figure saved with name: ',fig_save)
    else:
        print('Figure not saved.')
    return None

def crop_spectra(spectra,min,max):
    '''
    This function crops the energy axis of an spectral dataset.

    spectra: array-like of shape (n_spectra,n_features).
        Spectral dataset. 
    min: float.
        Minimum value to crop the spectra.  
    max: float.
        Maximum value to crop the spectra. 

    Return:
    -------
    Returns the initial spectra cropped from the "min" to "max" index in the energy axis.
    '''
    
    spectra = spectra[:,min:max]

    return spectra