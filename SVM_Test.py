import sys
import hyperspy.api as hs
import os
import pickle
import sklearn
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedShuffleSplit, cross_val_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from matplotlib import cm
from random import randrange
import numpy as np
from sklearn.metrics import plot_confusion_matrix

def noise_PCA_spectra(model, X_test, y_test, noise_mult=5, scree_plot=None):
    '''
    This function gets a noise distribution from the low variance components of the Principal Component Analysis,
    and test the performance of a SVC model with the provided datasest with the noise distribution incorporated.

    -model: SVC estimator. 
        SVC trained model to evaluate the noise.
    -X_test: array-like of shape (n_spectra,n_features).
        Test samples.
    -y_test: array-like of shape (n_spectra)
        True labels for X_test.
    -noise_mult: int. 
        Maximum multiplier of the pca noise distribution.
    -scree_plot: boolean. 
		
    Returns:
    -------
    The resulting spectra with the PCA noise incorpored, and, 
    the resulting accuracy for each multiple of noise added.
    '''
    noise_test, noise_acc = [], []
    
    spectra = np.copy(X_test)
    mu = np.mean(spectra, axis=0) #Mean for intensity channel! 
    
    #Apply PCA to our dataset.
    pca = PCA()
    noise_model = pca.fit(spectra)
    
    #Criteria to chose the low variance pca components. 
    nCompTot = pca.components_.shape[0]
    nComp = 0
    while noise_model.explained_variance_[nComp]>0.05:
        nComp += 1
    print('Number of pca components used to calculate the noise ',nCompTot-nComp,' of a total of ',nCompTot,' pca components.')
    
    signals = np.dot(pca.transform(spectra)[:,:nComp], pca.components_[:nComp,:]) #High variance components or real spectra.
    signals += mu
    noise_lvl = np.dot(pca.transform(spectra)[:,nComp:], pca.components_[nComp:,:]) #Noise distribution. 
    
    num = int((noise_mult-1)*10)
    SNR = np.linspace(1,noise_mult,num)

    for i  in range(len(SNR)):
        noise_test.append(SNR[i]*noise_lvl + signals)
        j = 0
        for spectrum in noise_test[i]:
            noise_test[i][j] = renorm(spectrum)
            j = j+1
        noise_acc.append(model.score(noise_test[i].reshape(X_test.shape[0],X_test.shape[1]),y_test)) #Evaluate model performance. 
    
    noise_acc = np.array(noise_acc)
    
    if scree_plot == True:
        plt.figure()
        plt.title('Scree Plot')
        plt.plot(pca.explained_variance_ratio_, 'r.')
        plt.xlim(-0.1,30)
        plt.ylim(-0.01, 0.6)
        plt.xlabel('Principal Components')
        plt.show()
        
    return noise_test, noise_acc

def plot_PCA_noise(noisy_spec,num_spec,clean_spec,ax=700.,bx=730.,random_spectra=None,fig_save=None):
    '''
    This function plots a number "num_spec" of spectra for different multiples of the pca noise distribution. 

    noisy_spec: array-like of shape (n_spectra,n_features).
        Dataset containing the spectra with the PCA noise.
    num_spec: int.
        Number of spectra to be plotted in the plot. 
    clean_spec: array-like of shape (n_spectra,n_features).
        Spectral dataset with the PCA clean spectra. 
    ax: float. (default = 700 eV)
        Origin value of the energy axis. 
    bx: float. (default = 730 eV)
        Ending value of the energy axis. 
    random_spectra: int, or None. 
        Index of the spectrum in the spectral dataset. 
        If None index is provided a random spectrum will be selected. 
    fig_save: string. 
        Name of the file to save the plot. 
    
    Returns:
    --------
    Returns a plot with a number "num_spec" of spectra of the pca noise distribution.
    '''
    if random_spectra is not None: 
        spectro_ind = random_spectra
    else:
        spectro_ind = randrange(0,len(clean_spec))
        print('Random spectra selected with index: ',spectro_ind)

    fig = plt.figure(dpi=150,figsize=(3,5))
    
    plt.gca().set(xlim=(ax,bx),ylim=(-0.03,num_spec+1.03))
    
    k = int(len(noisy_spec)/num_spec)
    #PCA CLEANED SPECTRA
    plt.plot(np.arange(ax,bx,step=.1),clean_spec[spectro_ind],label='PCA Cleaned',linewidth=0.7,linestyle='-')
    
    for i in range(0,num_spec):
        plt.plot(np.arange(ax,bx,step=.1),noisy_spec[i*k][spectro_ind][:]+(1+i),linewidth=0.7,linestyle='-')

    plt.xticks(np.arange(ax,bx,step=5))
    plt.yticks([])
    plt.xlabel('Energy Loss (eV)')
    plt.ylabel('Normalized Intensity (a.u.)')
    
    plt.text(717, 0.5,'PCA Cleaned',fontdict= {'family' : 'Arial', 'fontsize' : 8, 'weight' : 'normal'})
    plt.text(717, 1.5,'Original',fontdict= {'family' : 'Arial', 'fontsize' : 8, 'weight' : 'normal'})
    plt.text(717, 4.5,'4 x Noise',fontdict= {'family' : 'Arial', 'fontsize' : 8, 'weight' : 'normal'})
    plt.text(717, 3.5,'3 x Noise',fontdict= {'family' : 'Arial', 'fontsize' : 8, 'weight' : 'normal'})
    plt.text(717, 2.5,'2 x Noise',fontdict= {'family' : 'Arial', 'fontsize' : 8, 'weight' : 'normal'})
    plt.text(717, 5.55,'5 x Noise',fontdict= {'family' : 'Arial', 'fontsize' : 8, 'weight' : 'normal'})
    
    plt.show()

    if fig_save is not None:
        fig.savefig(fig_save, bbox_inches = 'tight')
        print('Figure saved with name: ',fig_save)
    else:
        print('Figure not saved.')
    return None

def plot_PCA_acc(pca_acc,label,fig_save=None):
    '''
    This function plots the Cross Validated Mean Accuracy as a function of multiples of pca noise.
    
    pca_acc: array-like of shape (n_multipliers)
        Accuracy as a function of the PCA noise multiplier added. 
    label: string.
        Label to indicate the kernel and training dataset used. 
    fig_save: string. (Default=None)
        Name of the file to save the plot. 
        
    Returns:
    -------
    Plot with the accuracy of the SVM estimator as a function of the pca multiplier of noise added. 
    '''
    fig = plt.figure(dpi=150)

    plt.scatter(np.arange(1,5,step=.1),pca_acc,label=label,marker='o',s=8,color='fuchsia')
    plt.xlabel('Multiples of Noise')
    plt.grid()
    plt.ylabel('10-fold Cross Validated Mean Test Accuracy')

    plt.legend()
    plt.show()
    
    if fig_save is not None:
        fig.savefig(fig_save, bbox_inches = 'tight')
        print('Figure saved with name: ',fig_save)
    else:
        print('Figure not saved.')
    
    return None

def cross_fold_validation(model,x_val,y_val,x_test,y_test,train_size,shift=False,val_shift=0.,cv=10):
    '''
    This function evaluate a score by cross-validation.  

    -model: SVC estimator. 
        The SVC estimator to use to fit the data.
    -x_val: array-like of shape (n_spectra,n_features).
        Whole spectral dataset.
    -y_val: array-like of shape (n_spectra).
        Labels of the x_val.
    -x_test: array-like of shape (n_spectra,n_features).
        Larger energy axis spectral dataset.
    -y_test: array-like of shape (n_spectra).
        Labels of the x_test.
    -train_size: float from 0 to 1.
        It represents the proportion of the dataset to include in the train split.
    -shift: boolean. 
        If True a chemical shitf is included in the testing spectra. 
    -val_shift: int.
        Value of channel displacement in the test sets.
    -cv: int.
        Determines the number of folds in a Stratified KFold strategy. 

    Returns:
    --------
    score: float.
        Mean score of the estimator for all the run of the cross validation.
    '''

    scores, train, test = [], [], []
    train_size = train_size
    X = x_val
    y = np.array(y_val)
    y_test = np.array(y_test)
    
    #Generate a cv train&test configurations for fit and test the model:
    sss = StratifiedShuffleSplit(n_splits=cv, train_size=train_size)
    sss.get_n_splits(X)
    #Save each training and testing set:

    for train_index, test_index in sss.split(X, y):
        #The train sets are generated with the x_val values: 
        train.append([X[train_index],y[train_index]])
        #The test sets are generated with the x_test values in order to apply the shift:
        test.append([x_test[test_index],y_test[test_index]])
        
    if shift: #Chemical shift:
        for i in range(0,len(train[:])):
            model.fit(train[i][0],train[i][1])
            scores.append(model.score(test[i][0][:,50+val_shift:350+val_shift],test[i][1]))
            
    else: #Cross-fold validation:
        for i in range(0,len(train[:])):
            model.fit(train[i][0],train[i][1])
            scores.append(model.score(test[i][0][:,:],test[i][1]))
            
    return np.mean(scores)

def chemical_shifts(model,x_val,y_val,x_test,y_test,num_bins_translate,train_size,cv=10):
    '''
    This function evaluates the performance of the SVC estimator when energy shifts are incorpored in the spectra.  

    -model: SVC estimator.
    	SVC trained or fit model.
    -x_val: array-like of shape (n_spectra,n_features).
    	Whole spectral dataset.
    -y_val: array-like of shape (n_spectra).
    	Labels of the x_val.
    -x_test: array-like of shape (n_spectra,n_features).
    	Larger energy axis spectral dataset.
    -y_test: array-like of shape (n_spectra).
    	Labels of the x_test.
    -num_bins_translate: int. 
    	Total displacement per direction. Each shift corresponds with a displacement of 0.1eV.
    -train_size: float from 0 to 1.
    	It represents the proportion of the dataset to include in the train split.
    -cv: int.
    	Determines the number of folds in a Stratified KFold strategy. 

    Returns:
    --------
    accuracy: array of float, len=2*num_bins_translate.
    Array of scores of the estimator by cross-validation for each displacement.
    '''
    accuracy_shift_plus, accuracy_shift_minus = [], []
    
    for i in range(int(num_bins_translate/2)):
        accuracy_shift_plus.append(cross_fold_validation(model,x_val,y_val,x_test,y_test,train_size,True,val_shift=i,cv=cv))
    for j in range(int(num_bins_translate/2)):
        accuracy_shift_minus.append(cross_fold_validation(model,x_val,y_val,x_test,y_test,train_size,True,val_shift=-j,cv=cv))
    
    accuracy_shift_plus = np.array(accuracy_shift_plus)
    accuracy_shift_minus = np.array(accuracy_shift_minus)
    accuracy = np.concatenate((accuracy_shift_minus[::-1],accuracy_shift_plus),axis=0)

    return accuracy

def plot_chemical_shift(x_axis,shift_acc,labels,title,fig_save=None):

    '''
    This function plots the scores of the SVC estimator as a function of the chemical shifts.  

    x_axis: array-like of shape (n_shifts).
        Array with the values of the energy axis.
    shift_acc: array-like of shape (n_proportion,n_shifts).
        Array with the scores per energy displacements. 
    labels: array-like of strings. 
        Proportion of the dataset to train the SVC estimator. 
    title: string.
        Title of the plot. 
    fig_save: string. 
        Name of the file to save the plot.
    Returns:
    -------
    Returns the plot of the accuracy as a function of the energy shifts. 
    '''

    fig = plt.figure(figsize=(5,5),dpi=200)

    plt.title(title)
    for i in range(0,len(shift_acc)):
        plt.plot(x_axis,shift_acc[i],linewidth=.5,label=labels[i])
        
    plt.ylabel('Averaged 10-fold cross-validation accuracy')
    plt.xlabel('Chemical Shift (eV)')
    plt.yticks(np.arange(0,1.1,step=.1))
    plt.grid(b=True, which='both', axis='both')
    plt.legend(title='Training-set data \n percentage:',fontsize=9,title_fontsize=9)
    
    plt.show()

    if fig_save is not None:
        fig.savefig(fig_save,bbox_inches = 'tight')
        print('Figure saved with name: ',fig_save)
    else:
        print('Figure not saved.')
    
    return None

def add_gaus_shot_noise(spec,gaus=True,std=.15,pois=True):
    '''
    This function recieve an EELS spectral dataset, and return this spectral dataset 
    with Gaussian or Possonian noise added on it. If both types of noise are chosen,
    the function will return the combination of both.

    -spec: array-like of shape (n_spectra,n_features).
        Spectral dataset.
    -gaus: boolean (Default = True).
        Add Gaussian Noise in the spectral dataset. 
    -std: float. (Default = 0.15)
        Standard deviation of the gaussian noise.
    -pois: boolean (Default = True).
        Add Poissonian noise in the spectral dataset.

    Returns:
    --------
    Spectral dataset with the noise distribution selected added.
    Each resulting spectral dataset is an array-like of shape (n_spectra,n_features).    
    '''

    spectra = spec.copy()
    
    if gaus and pois:
        spectra_gaus = hs.signals.EELSSpectrum(spectra.copy())
        spectra_gaus.add_gaussian_noise(std=std)
        spectra_shot = hs.signals.EELSSpectrum(spectra.copy()*100)
        spectra_shot.add_poissonian_noise()
        spectra_shot = hs.signals.EELSSpectrum(spectra_shot.data/100)
        spectra_both = spectra_shot.deepcopy()
        spectra_both.add_gaussian_noise(std=std)
        print('Gaussian and Poisonian Noise added.')
        spectra_shot = np.array(spectra_shot.data)
        spectra_gaus = np.array(spectra_gaus.data)
        spectra_both = np.array(spectra_both.data)
        return spectra_shot, spectra_gaus, spectra_both
    elif gaus:
        spectra_gaus = hs.signals.EELSSpectrum(spectra.copy())
        spectra_gaus.add_gaussian_noise(std=std)
        print('Gaussian Noise added.')
        spectra_gaus = np.array(spectra_gaus.data)
        return spectra_gaus
    elif pois:
        spectra_shot = hs.signals.EELSSpectrum(spectra.copy()*100)
        spectra_shot.add_poissonian_noise()
        spectra_shot = hs.signals.EELSSpectrum(spectra_shot.data/100)
        print('Poison/Shot Noise added.')
        spectra_shot = np.array(spectra_shot.data)
        return spectra_shot
    else: 
        print('No added Noise.')
        return None

def plot_gaus_pois_noise(spectro_ind,clean_spec,shot_spec,gaus_spec,both_spec,ax,bx,fig_save=None):
    '''
    This function plots the clean spectra, the gaussian noise spectra, or the posonian noise spectra.

    spectro_ind: int, or None. 
        Index of the spectrum in the spectral dataset.
        If None index is provided a random spectrum will be selected. 
    clean_spec: array-like of shape (n_spectra,n_features).
        Spectral dataset with the PCA clean spectra. 
    shot_spec: array-like of shape (n_spectra,n_features).
        Spectral dataset with poissonian noise added. 
    gaus_spec: array-like of shape (n_spectra,n_features).
        Spectral dataset with guassian noise added. 
    both_spec: array-like of shape (n_spectra,n_features).
        Spectral dataset with the PCA clean spectra. 
    ax: float. (default = 700 eV)
        Origin value of the energy axis. 
    bx: float. (default = 730 eV)
        Ending value of the energy axis. 
    fig_save: string. 
        Name of the file to save the plot. 

    Returns:
    --------
    Returns a plot with spectra incorporing gaussian, possonian, a combination of both noises. 
    '''
    
    if spectro_ind==None:
        spectro_ind = randrange(0,len(clean_spec))
        print('Random spectra selected with index: ',spectro_ind)
    else: 
        spectro_ind = spectro_ind
    
    fig = plt.figure(dpi=150,figsize=(3,5))
    
    plt.gca().set(xlim=(ax,bx),ylim=(-0.03,4+1.03))
    plt.xticks(np.arange(ax,bx, step=5))
    plt.yticks([])
    plt.xlabel('Energy Loss (eV)')
    plt.ylabel('Normalized Intensity (a.u.)')

    plt.plot(np.arange(ax,bx,step=.1),clean_spec[spectro_ind],linewidth=0.7,linestyle='-')
    plt.plot(np.arange(ax,bx,step=.1),shot_spec[spectro_ind]+1,linewidth=0.7,linestyle='-')
    plt.plot(np.arange(ax,bx,step=.1),gaus_spec[spectro_ind]+2.2,linewidth=0.7,linestyle='-')
    plt.plot(np.arange(ax,bx,step=.1),both_spec[spectro_ind]+3.4,linewidth=0.7,linestyle='-')
    
    plt.text(716.5, 0.5,'PCA Cleaned',fontdict= {'family' : 'Arial', 'fontsize' : 8, 'weight' : 'normal'})
    plt.text(716, 1.7,'Poissonian Noise',fontdict= {'family' : 'Arial', 'fontsize' : 8, 'weight' : 'normal'})
    plt.text(713.5, 4.35,'Poisson + Gaussian (0.1)',fontdict= {'family' : 'Arial', 'fontsize' : 8, 'weight' : 'normal'})
    plt.text(715, 3.0,'Gaussian Noise (0.1)',fontdict= {'family' : 'Arial', 'fontsize' : 8, 'weight' : 'normal'})
    
    plt.show()

    if fig_save is not None:
        fig.savefig(fig_save,bbox_inches = 'tight')
        print('Figure saved with name: ',fig_save)
    else:
        print('Figure not saved.')

    return None

def plot_conf_matrix(model, x_test, y_test, class_names=['Fe$^{+2}$','Fe$^{+2}$Fe$^{+3}_2$','Mn$^{+2}$','Mn$^{+3}$','Mn$^{+4}$']):
    '''
    This function plots the confusion matrix for an SVC estimator.
     
    -model: SVC estimator. 
        SVC trained model.
    -x_test: array-like of shape (n_spectra,n_features).
        Spectral dataset.
    -y_test: array-like of shape (n_spectra).
        Labels of the Spectral dataset.

    Returns:
    --------
    Plots the confusion matrix for an estimator.
    '''
    
    np.set_printoptions(precision=2)
    #Plot normalized confusion matrix:
    title = 'Confusion matrix'
    disp = plot_confusion_matrix(model, x_test, y_test,
                                 display_labels=class_names,
                                 cmap='gray',
                                 values_format='.2f',
                                 normalize='all')
    disp.ax_.set_title(title)
    plt.show()

    return None

def renorm(spectrum):
    '''
    This funtion normalize an spectrum beetween 0 and 1. 

    spectrum = array-like of shape (n_features)
        Spectrum to be normalized. 

    Returns:
    --------
    Normalized spectra between 0 and 1. 
    '''
    norm_spectrum = (spectrum-spectrum.min())/(spectrum.max()-spectrum.min())
    
    return norm_spectrum

def test_svm_cv(model,x_test,y_test,cv=10):
    '''
    This function test a SVM estimator via cross-fold validation.
    
    -model: SVC estimator.
        SVC model.
    -x_test: array-like of shape (n_spectra,n_features).
        Testing dataset.
    -y_test: array-like of shape (n_spectra).
        Labels.
    -cv: int. (default=10)
        Cross validation fold.

    Returns:
    --------
    Returns the cross validation mean accuracy.
    '''
    acc = []   
    acc.append(cross_val_score(estimator=model,X=x_test,y=y_test,cv=cv))
    acc = np.array(acc)
    print('Averaged ',str(cv),'-fold cross-validation accuracy: %0.2f (+/- %0.2f)' % (acc.mean(), acc.std()*2))

    return acc

def test_svm(model,x_test,y_test):
    '''
    This function test an SVM estimator.
    
    -model: SVC estimator.
        SVC model.
    -x_test: array-like of shape (n_spectra,n_features).
        Testing dataset.
    -y_test: array-like of shape (n_spectra).
        Labels.

    Returns:
    --------
    Returns the resulting accuracy by the SVC estimator.
    '''
    acc = model.score(x_test,y_test)  
    print('Test accuracy: ', acc,'.')

    return acc