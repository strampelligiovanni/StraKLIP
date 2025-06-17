"""
set of ancillary routines needed by the pipeline or to perform analysis
"""
import math
import numpy as np
import pandas as pd
from straklip.stralog import getLogger

from numpy import unravel_index
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip
from random import random
from tkinter import Tk
from screeninfo import get_monitors
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp2d
from photutils import DAOStarFinder,CircularAperture
from texttable import Texttable
from latextable import draw_latex
from scipy import stats
from tabulate import tabulate
from scipy.odr import Model,RealData,ODR
from synphot import ExtinctionModel1D,Observation,SourceSpectrum,SpectralElement
from stsynphot import band
from synphot.units import FLAM
from dust_extinction.parameter_averages import CCM89
from synphot.reddening import ExtinctionCurve
from astropy import units as u
from scipy.interpolate import LinearNDInterpolator,Rbf
from plotly.subplots import make_subplots
from plotly.graph_objects import Scatter3d,Scatter
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from tqdm import tqdm
from pylab import Circle
from IPython.display import display
from astropy.coordinates import Angle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage.interpolation import shift
from reftools.interpretdq import ImageDQ, DQParser
from astroscrappy import detect_cosmics
from scipy.stats import norm

def cosmic_ray_filter_la(self, sigclip=4.5,niter=5,inmask = None, sepmed=False,cleantype='medmask',fsmode='median',readnoise=5,verbose=False, satlevel=65536.0):
    '''
    apply L.A. cosmic ray removal

    Parameters
    ----------
    data : numpy arays
        input data. N.B. Data MUST be in total counts (not counts/sec).
    sigclip : float, optional
        sigma for sigma clipping. The default is 4.5.
    niter : int, optional
        nuber of iteration in sigma clipping. The default is 5.
    inmask : boolean numpy array, optional
        Input bad pixel mask. The default is None.
    sepmed : bool, optional
        Use the separable median filter instead of the full median filter. The default is False.
    cleantype : {‘median’, ‘medmask’, ‘meanmask’, ‘idw’}, optional
        Set which clean algorithm is used:
        ‘median’: An umasked 5x5 median filter
        ‘medmask’: A masked 5x5 median filter
        ‘meanmask’: A masked 5x5 mean filter
        ‘idw’: A masked 5x5 inverse distance weighted interpolation. The default is 'medmask'.
    fsmode : {‘median’, ‘convolve’}, optional
        Method to build the fine structure image:
        ‘median’: Use the median filter in the standard LA Cosmic algorithm
        ‘convolve’: Convolve the image with the psf kernel to calculate the fine structure image. The default is 'median'.
    readnoise : float, optional
        Read noise of the image (electrons). The default is 5.
    verbose : bool, optional
        choose to show prints. The default is False.
    satlevel : float, optional
        data saturation level. The default is 65536.0.

    Returns
    -------
    None.

    '''
    self.cr_mask,self.cr_clean_im=detect_cosmics(self.data,verbose=verbose,sigclip=sigclip,niter=niter,inmask = inmask, sepmed=sepmed,cleantype=cleantype,fsmode=fsmode,readnoise=readnoise,satlevel=satlevel)
    self.cr_mask=self.cr_mask.astype(int)

def cosmic_ray_filter(self,r,key_list=[4096,8192,16384],delta=2,verbose=False,kill=False):
    '''
    apply a median filter to image to filter cosmic rays

    Parameters
    ----------
    data : numpy arays
        input data.
    dqdata : numpy arrays
        input data quality.
    r : float
        minimum distance from center where to not apply the cosmic ray filter.
    inst : str
        name of the instrument.
    key_list : list, optional
        pixels corresponding to these keys in the data quality image will be mask in the
        input data for the popouse of evaluating the median. The default is [4096,8192,16384].
    delta : int, optional
        number of pixels the image will e shifted in +/- x and y befor evaluating the median. The default is 2.
    verbose : bool, optional
        choose to show prints. The default is False.
    kill : bool, optional
        choose to kill bad pixels instead of using the median of the neighbouring pixels. The default is False.

    Returns
    -------
    None.

    '''
    data_temp=self.data.copy()
    list_of_shifted_images=[]
    for x in np.arange(-delta,delta+1):
        for y in np.arange(-delta,delta+1):
            shifted_images=shift(self.data,[y,x],order=0,mode='wrap')
            list_of_shifted_images.append(shifted_images)
    list_of_shifted_images=np.array(list_of_shifted_images)
    if kill==False: mdata=np.median(list_of_shifted_images,axis=0)
    dqparser = DQParser.from_instrument(self.inst)
    acsdq = ImageDQ(self.dqdata, dqparser=dqparser)
    if verbose==True:print(acsdq.parser.tab )
    acsdq.interpret_all(verbose=verbose)

    ylist=[]
    xlist=[]

    for key in key_list:
        for x,y in acsdq.pixlist(origin=0)[key]:
            dx=x-int(data_temp.shape[1]/2)
            dy=y-int(data_temp.shape[0]/2)
            sep=np.sqrt(dx**2+dy**2)
            if sep > r:
                ylist.append(y)
                xlist.append(x)

    ylist=np.array(ylist)
    xlist=np.array(xlist)
    w=(ylist,xlist)

    if len(ylist)!=0 and len(xlist)!=0:
        if kill==False:self.data[w]=mdata[w]
        else: self.data[w]=-1

    del list_of_shifted_images,data_temp,ylist,xlist,w


def dataframe_2D_finer_interpolator(hdf, n_step=0.1,xnew=None,ynew=None,X_index_label='sep',Y_index_label='dmag',Z_columns_label='ratio'):
    '''
    Perform a finer interpolation of the values of a multiindex dataframe accros two indexs

    Parameters
    ----------
    hdf : pandas Multi-index Dataframe
        input dataframe.
    n_step : float, optional
        step for the finer interpolation. The default is 0.1.
    xnew : list, optional
        list of new Xs for the finer interpoplation. If None use min/max of 
        input X obtained from DataFrame
        The default is None.
    ynew : list, optional
        list of new Ys for the finer interpoplation. If None use min/max of 
        input Y obtained from DataFrame
        The default is None.
    X_index_label : str, optional
        label name for input X. The default is 'sep'.
    Y_index_label : str, optional
        label name for input Y. The default is 'dmag'.
    Z_columns_label : str, optional
        label name for input Z on which perfomr the finer interpolation. 
        The default is 'ratio'.

    Returns
    -------
    (new X, Y, Z, X_label, Y_label).

    '''
    X=hdf.index.get_level_values(X_index_label).unique()
    Y=hdf.index.get_level_values(Y_index_label).unique()
    Z=hdf[Z_columns_label].values
    f=interp2d(Y, X, Z)
    
    ############# Build a finer grid in ynew and xnew ######################   
    if np.all(xnew) == None and np.all(ynew) == None:
        xnew=np.arange(min(X),max(X)+1+n_step,n_step)
        ynew=np.arange(min(Y),max(Y)+1+n_step,n_step)
    
    ############# New completeness obtained from the interpolation over the finer grid ################           
    znew = f(ynew, xnew)
    znew[znew>1]=1
    znew[znew<0]=0
    return(xnew,ynew,znew,X_index_label,Y_index_label)

def distances_cube(df,id_label='unq_ids',coords_labels=['ra','dec'],showplot=True,pixelscale=1.0,bins=50,nx=10,ny=5,min_separation=0,max_separation=2,skip_type=None):
    '''
    create a datacube with distances between all entry in dataframe. It needs to be use in conjunction with a dataframe class object

    Parameters
    ----------
    coords_labels: list, optional
        list of labels for sources coordiantes. The default is ['ra','dec']
    showplot : bool, optional
        choose to show final plot. The default is True.
    pixelscale : float, optional
        Camera pixelscale. The default is 1.
    bins : int, optional
        bin size for final histogram. The default is 50.
    nx : int, optional
        x dimension of the figure. The default is 10.
    ny : int, optional
        y dimension of the figure. The default is 5.
    min_separation : float, optional
        minimum separation in arcsec for good targets. The default is 0.
    max_separation : float, optional
        minimum separation in arcsec for Known doubles. The default is 2.
        
    Returns
    -------
    None.

    '''
    getLogger(__name__).info(f'Making the distance cube')
    dist_list=[]
    min_dist=[]
    elno=0
    df['FirstDist']=np.nan
    df['SecondDist']=np.nan
    df['ThirdDist']=np.nan
    df['FirstID']=np.nan
    df['SecondID']=np.nan
    df['ThirdID']=np.nan    
    if skip_type!=None:df_temp=df.loc[df.type!=skip_type].copy()
    else: df_temp=df.copy()

    for ID in df_temp[id_label].unique():
        ID_list=df_temp.loc[df_temp[id_label]!=ID,id_label].values
        coo1=df_temp.loc[df_temp[id_label]==ID,coords_labels[0]].values[0]
        coo2=df_temp.loc[df_temp[id_label]==ID,coords_labels[1]].values[0]

        coo1_all=np.array(df_temp.loc[df_temp[id_label]!=ID,coords_labels[0]].tolist(),dtype=float)
        coo2_all=np.array(df_temp.loc[df_temp[id_label]!=ID,coords_labels[1]].tolist(),dtype=float)
        dist=np.sqrt((coo1_all-coo1)**2+(coo2_all-coo2)**2)
        if np.isnan(dist).any():print(dist[np.isnan(dist)])
        min_sep=Angle(min_separation,u.arcsec).deg
        
        min1=min(dist[dist>=min_sep])
        min2=min(dist[(dist>min1)&(dist>=min_sep)])
        min3=min(dist[(dist>min2)&(dist>min1)&(dist>=min_sep)])

        n1=np.where(np.array(dist)==min1)[0]
        n2=np.where(np.array(dist)==min2)[0]
        n3=np.where(np.array(dist)==min3)[0]

        first_min_dist=np.array(dist)[n1][0]
        second_min_dist=np.array(dist)[n2][0]
        third_min_dist=np.array(dist)[n3][0]

        firstID=ID_list[n1][0]
        secondID=ID_list[n2][0]
        thirdID=ID_list[n3][0]

        if all(x in coords_labels for x in ['ra','dec']):
            df_temp.loc[df_temp[id_label]==ID,'FirstDist']=Angle(first_min_dist,u.deg).arcsec
            df_temp.loc[df_temp[id_label]==ID,'SecondDist']=Angle(second_min_dist,u.deg).arcsec
            df_temp.loc[df_temp[id_label]==ID,'ThirdDist']=Angle(third_min_dist,u.deg).arcsec
        else:
            df_temp.loc[df_temp[id_label]==ID,'FirstDist']=first_min_dist*pixelscale
            df_temp.loc[df_temp[id_label]==ID,'SecondDist']=second_min_dist*pixelscale
            df_temp.loc[df_temp[id_label]==ID,'ThirdDist']=third_min_dist*pixelscale
                    
        df_temp.loc[df_temp[id_label]==ID,'FirstID']=int(firstID)
        df_temp.loc[df_temp[id_label]==ID,'SecondID']=int(secondID)
        df_temp.loc[df_temp[id_label]==ID,'ThirdID']=int(thirdID)

        dist_list.append(list(dist))
        min_dist.append(first_min_dist)

        elno+=1

    #Minimum distance between obj of the original catalog
    if showplot == True:
        fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(nx,ny),squeeze=True)
        plt.subplots_adjust(wspace=0.3)
        axes[0].set_title('Distance DataFrame')
        im0 = axes[0].imshow(dist_list,origin='lower')
        axes[0].set_xlabel('OBJ id')
        axes[0].set_ylabel('OBJ id')

        divider0 = make_axes_locatable(axes[0])
        cax0 = divider0.append_axes("right", size="10%", pad=0.05)
        plt.colorbar(im0, cax=cax0, format="%.2f")

        axes[1].hist(min_dist,bins=bins)
        axes[1].set_title('MinDist Hist')

        plt.show()
    
    # df_temp.loc[df_temp.FirstDist<=max_separation,'type']=3
    if skip_type!=None:out=pd.concat([df.loc[df.type==skip_type],df_temp]).sort_values('unq_ids')
    else: out=df_temp.copy()

    return(out)

def deg2hmsdms(df,Ra_str,Dec_str,New_RA_str=None,New_Dec_str=None):
    new_df=df.copy()
    del df
    target = SkyCoord(ra=np.array(new_df[Ra_str])*u.degree, dec=np.array(new_df[Ra_str])*u.degree, frame='fk5')
    RA_hms=[]
    for elno in range(new_df[Ra_str].count()): 
        RA_hms.append('%i:%i:%.3f'%(target.ra.hms[0][elno],target.ra.hms[1][elno],target.ra.hms[2][elno]))
    DEC_dms=[]
    for elno in range(new_df[Dec_str].count()): 
        DEC_dms.append('%i:%i:%.2f'%(target.dec.signed_dms[0][elno]*target.dec.signed_dms[1][elno],target.dec.signed_dms[2][elno],target.dec.signed_dms[3][elno]))
    if New_RA_str!=None: Ra_str=New_RA_str
    if New_Dec_str!=None: Dec_str=New_Dec_str
    new_df[Ra_str]=RA_hms
    new_df[Dec_str]=DEC_dms
    return(new_df)

def evaluate_multiplicity(ntarget,n_mult,verbose=True,ca=0):
    mf=n_mult/(ntarget)
    emf1=np.sqrt(mf*(1-mf)/ntarget) #vedi sana 2014 e sana 2009
    emf=np.sqrt(emf1**2+(ca/ntarget)**2)
    if verbose==True: print('Systems: %i Binaries: %i MF=%.1f+\-%.1f\n'%(ntarget,n_mult,mf*100,emf*100))
    return(mf,emf)

def exp_func(x, a,b):
    return np.exp(-x*b) + a

def find_max(image4plot,ni,ns,speak=False):
    '''
    Find maxiimum in image data type

    Parameters
    ----------
    image4plot : numpy list of lists
        input image.
    ni : int
        minimum x,y delta to create a box.
    ns : int
        minimum x,y delta to create a box.
    speak : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    x,y max.

    '''
    ni=int(round(ni))
    ns=int(round(ns))
    y_m,x_m=unravel_index(np.nanargmax(image4plot[ni:ns,ni:ns]), image4plot[ni:ns,ni:ns].shape)
    return(x_m+ni,y_m+ni)

def find_centroid(data,x_tile,y_tile,xy_dmax,fwhm,sigma,std):
    daofind = DAOStarFinder(fwhm=fwhm, threshold=sigma*std)
    apertures = CircularAperture([int(x_tile),int(y_tile)], r=xy_dmax)
    deltas=(np.array(data.shape)-1)/2-(np.array(apertures.to_mask(method='center').multiply(data).shape)-1)/2
    sources_tab=daofind(apertures.to_mask(method='center').multiply(data))

    return(sources_tab,deltas)

def find_closer(array,value):
    '''
    find closer value in array to input value

    Parameters
    ----------
    array : list
        input list.
    value : int or float
        input value.

    Returns
    -------
    out: int or float
        closer value in array to input value.
    idx : int
        index of closer value in array to input value.
    '''
    out=array=np.array(array)
    idx=np.unravel_index((np.abs(array - value)).argmin(), array.shape)
    return (out[idx],idx)


def frac_above_thresh(data, thresh):
    """
    thresh can be an integer or an array of thresholds
    """
    if np.size(thresh) == 1:
        thresh = np.array([thresh])
    return np.squeeze(np.sum( (data > thresh[:,None]), axis=-1))/np.float64(data.size)

def gaussian_func(x, a, x0, sigma,c):
    return (a * np.exp(-(x-x0)**2/(2*sigma**2)) + c)

def get_Av_dict(filter_label_list,inst=None,date=None,verbose=False,Av=1,Rv=3.1,band_dict={},path2saveim=None):
    # obsdate = Time(date).mjd
    vegaspec = SourceSpectrum.from_vega()  
    Dict = {}
    
    wav = np.arange(3000, 15000,10) * u.AA
    extinct = CCM89(Rv=Rv)
    ex = ExtinctionCurve(ExtinctionModel1D,points=wav, lookup_table=extinct.extinguish(wav, Av=Av))
    vegaspec_ext = vegaspec*ex
    
    filter_band = SpectralElement.from_filter('johnson_v')#555
    sp_obs = Observation(vegaspec_ext, filter_band)
    sp_obs_before = Observation(vegaspec, filter_band)
    
    sp_stim_before = sp_obs_before.effstim(flux_unit='vegamag', vegaspec=vegaspec)
    sp_stim = sp_obs.effstim(flux_unit='vegamag', vegaspec=vegaspec)
    
    if verbose:
        # print('before dust, V =', np.round(sp_stim_before,4))
        # print('after dust, V =', np.round(sp_stim,4))
        getLogger(__name__).info(f'before dust, V =  {np.round(sp_stim_before,4)}')
        getLogger(__name__).info(f'after dust, V = {np.round(sp_stim,4)}')

        flux_spectrum_norm = vegaspec(wav).to(FLAM, u.spectral_density(wav))
        flux_spectrum_ext = vegaspec_ext(wav).to(FLAM, u.spectral_density(wav))

        plt.semilogy(wav,flux_spectrum_norm,label='Av = 0')
        plt.semilogy(wav,flux_spectrum_ext,label='Av = %s'%Av)
        plt.legend()
        plt.ylabel('Flux [FLAM]')
        plt.xlabel('Wavelength [A]')
        plt.xlim(3000, 15000)
        if path2saveim is not None:
            plt.savefig(path2saveim+'/Vega_spectrum.png')
            plt.close()
        else:
            plt.show()
    
        # Calculate extinction and compare to our chosen value.
        Av_calc = sp_stim - sp_stim_before
        getLogger(__name__).info(f'Av = {np.round(Av_calc, 4)}')
        # print('Av = ', np.round(Av_calc,4))
    
    if any('johnson' in string for string in filter_label_list):
        for filter in filter_label_list:
            obs = Observation(vegaspec, SpectralElement.from_filter(filter))
            obs_ext = Observation(vegaspec_ext, SpectralElement.from_filter(filter))
            if verbose: 
                # print('AV=0 %s'%filter,obs.effstim('vegamag',vegaspec=vegaspec))
                # print('AV=1 %s'%filter,np.round(obs_ext.effstim('vegamag',vegaspec=vegaspec)-obs.effstim('vegamag',vegaspec=vegaspec),4))
                obs_before=obs.effstim('vegamag',vegaspec=vegaspec)
                obs_delta=np.round(obs_ext.effstim('vegamag',vegaspec=vegaspec)-obs.effstim('vegamag',vegaspec=vegaspec),4)
                getLogger(__name__).info(f'AV=0 {filter} {obs_before}')
                getLogger(__name__).info(f'AV=1 {filter} {obs_delta}')

            Dict[filter]=np.round(obs_delta.value,4)
    else:
        for ext in band_dict.keys():
            Dict_temp = {}
            for filter in filter_label_list:
                # if filter in ['F130N','F139M']:
                #     obs = Observation(vegaspec, band('wfc3,ir,%s'%filter.lower()))
                #     obs_ext = Observation(vegaspec_ext, band('wfc3,ir,%s'%filter.lower()))
                # elif filter in ['F336W','F439W','F656N','F814W']:
                #     obs = Observation(vegaspec, band('acs,wfpc2,%s'%filter.lower()))
                #     obs_ext = Observation(vegaspec_ext, band('acs,wfpc2,%s'%filter.lower()))
                # elif filter in ['F110W','F160W']:
                #     obs = Observation(vegaspec, band('nicmos,3,%s'%filter.lower()))
                #     obs_ext = Observation(vegaspec_ext, band('nicmos,3,%s'%filter.lower()))
                # else:
                #     obs = Observation(vegaspec, band(f'acs,wfc1,%s,mjd#{obsdate}'%filter.lower()))
                #     obs_ext = Observation(vegaspec_ext, band(f'acs,wfc1,%s,mjd#{obsdate}'%filter.lower()))
                if inst.lower() == 'acs':
                    band_filter = f"{band_dict[ext].split(',')[0]},{band_dict[ext].split(',')[1]},{filter.lower()},{band_dict[ext].split(',')[2]}"
                else:
                    band_filter = band_dict[ext] + f',{filter.lower()}'

                obs = Observation(vegaspec, band(band_filter))
                obs_ext = Observation(vegaspec_ext, band(band_filter))
                if verbose:
                    # print('Av=0 %s'%filter,obs.effstim('vegamag',vegaspec=vegaspec))
                    # print('Av=1 %s'%filter,np.round(obs_ext.effstim('vegamag',vegaspec=vegaspec)-obs.effstim('vegamag',vegaspec=vegaspec),4))
                    obs_before=obs.effstim('vegamag',vegaspec=vegaspec)
                    obs_delta=np.round(obs_ext.effstim('vegamag',vegaspec=vegaspec)-obs.effstim('vegamag',vegaspec=vegaspec),4)
                    getLogger(__name__).info(f'AV=0 {band_filter} {obs_before}')
                    getLogger(__name__).info(f'AV=1 {band_filter} {obs_delta}')

                Dict_temp[f'm_{filter.lower()}']=np.round(obs_delta.value,4)

            Dict[ext] = Dict_temp

    return(Dict)

def get_Av_mass_and_Teff_from_isochrone(self,iso_df,filter1,filter2,DM,sel_good,show_plot=False,Av_MAX=32):
    q=np.where(np.array(self.filters_list)==filter1)[0][0]
    w=np.where(np.array(self.filters_list)==filter2)[0][0]
    A1=self.Av1_extinction[q]
    A2=self.Av1_extinction[w]
    
    for index,row in tqdm(self.unq_targets_df.loc[sel_good].iterrows()):
        if show_plot: plt.figure(figsize=(5,7))
        xrange=row['m%s'%filter1[1:4]]-row['m%s'%filter2[1:4]]+1 #distance to color=-0.5
        if xrange>=0:
            Color_Av=(A1 - A2)
            K=A1/(A1- A2)
            Npoints = int(xrange/Color_Av)
            Col_vector = np.linspace(0,-xrange,Npoints)
            Mag_vector=K*Col_vector
            
            #orange line of Av trial values:
            x  = row['m%s'%filter1[1:4]]-row['m%s'%filter2[1:4]]+Color_Av+Col_vector#np.linspace(0, -20, 40)
            y2 = row['m%s'%filter1[1:4]]+A1+Mag_vector#np.linspace(0, -20, 40)
            
            #isochrone, blue line
            x_model =iso_df['m%s'%filter1[1:4]]-iso_df['m%s'%filter2[1:4]]
            y_model = iso_df['m%s'%filter1[1:4]]+DM
    
            y1 = np.interp(x,x_model,y_model)
            if show_plot: 
                plt.axis([-1, 2, 24,10])
                plt.plot(self.unq_targets_df.loc[sel_good,'m%s'%filter1[1:4]]-self.unq_targets_df.loc[sel_good,'m%s'%filter2[1:4]],self.unq_targets_df.loc[sel_good,'m%s'%filter1[1:4]], '.',color='gray')
                plt.plot(x_model, y_model, marker='o', mec='none', ms=4, lw=1, label='model')
                plt.plot(x, y2, marker='o', mec='none', ms=4, lw=1, label='reddened')
                plt.plot(x, y1, marker='o', mec='none', ms=4, lw=1, label='interp')
                plt.plot(row['m%s'%filter1[1:4]]-row['m%s'%filter2[1:4]],row['m%s'%filter1[1:4]], 'or')
    
            idx = np.argwhere(np.diff(np.sign(y1 - y2)) != 0)
            if show_plot: plt.plot(x[idx], y1[idx], 'ms', ms=7, label='Nearest data-point method')
    
            xc, yc = interpolated_intercept(x,y1,y2)
            if show_plot: 
                plt.plot(xc, yc, 'co', ms=5, label='Nearest data-point, with linear interpolation')
                plt.legend(frameon=False, fontsize=10, numpoints=1, loc='lower left')
    
            if len(yc) >0:
                Av  = ((row['m%s'%filter1[1:4]]-yc)/(A1)).item()
    
                mag_mod=np.flip(iso_df['m%s'%filter1[1:4]]+DM)
                mass_mod=np.flip(iso_df['mass'])
    
                Mass = np.interp(yc.item(),mag_mod,mass_mod)
    
                if  (Mass >= 0.00101) and (Mass <= 1.395)  and (Av >= 0) and (Av <= Av_MAX):
                    self.unq_targets_df.loc[index,'mass'] = Mass
                    self.unq_targets_df.loc[index,'Av'] = Av
                    if show_plot: display(self.unq_targets_df.loc[index].to_frame().T)
                if show_plot:  
                    plt.show()
                    # break    

def get_monitor_from_coord(x, y):
    '''
    Function to get which monitor the notebook is on

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    monitors = get_monitors()

    for m in reversed(monitors):
        if m.x <= x <= m.width + m.x and m.y <= y <= m.height + m.y:
            return m
    return monitors[0]

def get_screen_px_res(verbose=False):
    '''
    Get screen resolution in pixels

    verbose : bool. Default is False
        Choose to show prints.

    Returns
    -------
    w,h: str.
        width and height.

    '''
    root = Tk()
    
    w = root.winfo_width()
    h = root.winfo_height()
    if verbose: print('Resolution w%s x h%s px'%(w,h))
    return(w,h)

def hmsdms2deg(df,Ra_str,Dec_str,New_RA_str=None,New_Dec_str=None):
    new_df=df.copy()
    del df
    c = SkyCoord(np.array((new_df[Ra_str].values+' '+new_df[Dec_str]).values), unit=(u.hourangle, u.deg),frame='fk5')
    if New_RA_str!=None: Ra_str=New_RA_str.copy()
    if New_Dec_str!=None: Dec_str=New_Dec_str.copy()
    new_df[Ra_str]=c.ra.deg
    new_df[Dec_str]=c.dec.deg
    return(new_df)

def interpND(*args,smooth=0,method='nearest',x_label='x',y_label='y',z_label=None,color_labels=None,showplot=False,radial=True,fx=1450,fy=1000,w_pad=3,h_pad=1,pad=3,npoints=50,nrows=1,surface=True,workers=None,progress=True,horizontal_spacing = 0.1, vertical_spacing = 0.1):
    '''Calculate 2d interpolation for the z axis along x and y variables. 
        Parameters:
            args: x, y, z, where x, y, z, ... are the coordinates of the nodes
            node_list: list of lists of values at the nodes to interpolate. The rotuine will perfom a different interpolation (using the same x and y) for each sublist of node_list.
            x_range,y_range: range of x,y variables in the from [in,end] over which evaluate the interpolation. If not provide will automaticaly take min,max as ranges.
            smooth: smoothnes parametr for Rbf routine
            method: 
                if 'nearest' selected, then use LinearNDInterpolator to interpolate, otherwise use the Rbf routine where method parameters are:
                'multiquadric': sqrt((r/self.epsilon)**2 + 1)
                'inverse': 1.0/sqrt((r/self.epsilon)**2 + 1)
                'gaussian': exp(-(r/self.epsilon)**2)
                'linear': r
                'cubic': r**3
                'quintic': r**5
                'thin_plate': r**2 * log(r)

            x_labe,y_label: labels  for x,y axis
            z_label: list of labels for the z axis
            showplot: if true show plot for the interpolation
            radial: use Rbf insead of interp2d
        Returns:
            interpolations_list: list of interpolatations functions to be called as new_z=interpolations_list(x0,y0)
    '''
    if workers==None: 
        ncpu=cpu_count()
        if ncpu>=3: workers=ncpu-2
        else: workers=1
        print('> Selected Workers:', workers)

    if np.all(z_label==None): z_label=['z']*len(args[0][-1])
    if surface:
        meshed_coords = np.meshgrid(*[np.linspace(np.min(args[0][i]),np.max(args[0][i]),npoints) for i in range(len(args[0][:-1]))])
        new_coords=[meshed_coords[i].ravel() for i in range(len(meshed_coords))]
    else:
        new_coords=np.array([np.linspace(np.min(args[0][i]),np.max(args[0][i]),npoints) for i in range(len(args[0][:-1]))])
    node_list=args[0][-1]
    if showplot: 
        
        ncols= int(round_up(len(node_list)/nrows))
        if len(*args)-1!=1:
            fig = make_subplots(rows=nrows, cols=ncols,
                                specs= [[{"type": "surface"} for i in range(ncols)] for j in range(nrows)],
                                horizontal_spacing = horizontal_spacing, vertical_spacing = vertical_spacing)
        else:
            fig = make_subplots(rows=nrows, cols=ncols,
                                horizontal_spacing = horizontal_spacing, vertical_spacing = vertical_spacing)
    else: 
        fig=None
        ncols=1
    interpolations_list=[]
    elno_list=[i for i in range(len(node_list))]
    row=1
    col=1
    Dict={}

    if progress:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for interpolation,args_reshaped,elno in tqdm(executor.map(interpND_task,elno_list,repeat(args),repeat(method),repeat(smooth))):
                fig,Dict=interpND_plots(elno,args,args_reshaped,node_list,interpolations_list,interpolation,Dict,z_label,x_label,y_label,new_coords,color_labels,row=row,col=col,fig=fig,fx=fx,fy=fy,ncols=ncols,showplot=showplot)
    else:
        if workers==0:
            for elno in elno_list:
                interpolation,args_reshaped,elno = interpND_task(elno,args,method,smooth)
                fig,Dict=interpND_plots(elno,args,args_reshaped,node_list,interpolations_list,interpolation,Dict,z_label,x_label,y_label,new_coords,color_labels,row=row,col=elno+1,fig=fig,fx=fx,fy=fy,ncols=ncols,showplot=showplot)
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for interpolation,args_reshaped,elno in executor.map(interpND_task,elno_list,repeat(args),repeat(method),repeat(smooth)):
                    fig,Dict=interpND_plots(elno,args,args_reshaped,node_list,interpolations_list,interpolation,Dict,z_label,x_label,y_label,new_coords,color_labels,row=row,col=col,fig=fig,fx=fx,fy=fy,ncols=ncols,showplot=showplot)

    if showplot:
        # plt.tight_layout()
        fig.show()
    return(Dict)

def interpND_plots(elno,args,args_reshaped,node_list,interpolations_list,interpolation,Dict,z_label,x_label,y_label,new_coords,color_labels,row=1,col=1,fig=None,fx=7,fy=7,ncols=1,showplot=False):
    interpolations_list.append(interpolation)
    Dict[z_label[elno]]=interpolation
    if showplot:
        if len(*args)-1==3:
            thisdict = {
                          "x": new_coords[0],
                          "y": new_coords[1],
                          "z": new_coords[2]}
                          # "z": interpolation(*new_coords)}
            thisdict2 = {
                          "x": args_reshaped[0].ravel(),
                          "y": args_reshaped[1].ravel(),
                          "z": args_reshaped[2].ravel()}
                          # "z": node_list[elno].ravel()}
            # marker_color1=new_coords[2]
            # marker_color2=args_in[2].ravel()
            marker_color1=interpolation(*new_coords)
            marker_color2=node_list[elno].ravel()
            z_label_ND=z_label[elno]
            label_ND2=z_label[elno]+'_o'
            label_ND1=z_label[elno]+'_i'
            plotND=True
        elif len(*args)-1==2:
            thisdict = {
                          "x": new_coords[0],
                          "y": new_coords[1],
                          "z": interpolation(*new_coords)}
            thisdict2 = {
                          "x": args_reshaped[0].ravel(),
                          "y": args_reshaped[1].ravel(),
                          "z": node_list[elno].ravel()}
            
            marker_color1='lightskyblue'
            marker_color2='black'
            z_label_ND=z_label[elno]
            label_ND2=z_label[elno]+'_o'
            label_ND1=z_label[elno]+'_i'
            plotND=True
        elif len(*args)-1==1:
            thisdict = {
                          "x": new_coords[0],
                          "y": interpolation(*new_coords)}
            thisdict2 = {
                          "x": args_reshaped[0].ravel(),
                          "y": node_list[elno].ravel()}
            
            marker_color1='lightskyblue'
            marker_color2='black'
            z_label_ND=[]
            y_label=z_label[elno]
            label_ND2=z_label[elno]+'_o'
            label_ND1=z_label[elno]+'_i'
            plotND=False
        
        fig=plot_ND(thisdict2,plotND=plotND,row=row,col=col,showplot=False,marker_color=marker_color2,fig=fig,name_label=label_ND2)
        fig=plot_ND(thisdict,plotND=plotND,showplot=False,row=row,col=col,marker_color=marker_color1,fx=fx,fy=fy,fig=fig,x_label=x_label,y_label=y_label,z_label=z_label_ND,name_label=label_ND1)
        col+=1
        if col >ncols:
            row+=1
            col=1
    return(fig,Dict)

def interpND_task(elno,args,method,smooth):
    args_reshaped=[args[0][i] for i in range(len(args[0][:-1]))]
    if method =='nearest': 
        node_list=args[0][-1]
        args2interp=[list(zip(*args_reshaped)), node_list[elno]]
        interpolation=LinearNDInterpolator(*args2interp)
    else:
        args_reshaped.append(args[0][-1][elno])
        interpolation=Rbf(*args_reshaped,smooth=smooth,function=method)
    return(interpolation,args_reshaped,elno)

def interpolated_intercept(x, y1, y2):
    """Find the intercept of two curves, given by the same x data"""

    def intercept(point1, point2, point3, point4):
        """find the intersection between two lines
        the first line is defined by the line between point1 and point2
        the first line is defined by the line between point3 and point4
        each point is an (x,y) tuple.

        So, for example, you can find the intersection between
        intercept((0,0), (1,1), (0,1), (1,0)) = (0.5, 0.5)

        Returns: the intercept, in (x,y) format
        """    

        def line(p1, p2):
            A = (p1[1] - p2[1])
            B = (p2[0] - p1[0])
            C = (p1[0]*p2[1] - p2[0]*p1[1])
            return A, B, -C

        def intersection(L1, L2):
            D  = L1[0] * L2[1] - L1[1] * L2[0]
            Dx = L1[2] * L2[1] - L1[1] * L2[2]
            Dy = L1[0] * L2[2] - L1[2] * L2[0]

            x = Dx / D
            y = Dy / D
            return x,y

        L1 = line([point1[0],point1[1]], [point2[0],point2[1]])
        L2 = line([point3[0],point3[1]], [point4[0],point4[1]])

        R = intersection(L1, L2)

        return R

    idx = np.argwhere(np.diff(np.sign(y1 - y2)) != 0)
    xc, yc = intercept((x[idx], y1[idx]),((x[idx+1], y1[idx+1])), ((x[idx], y2[idx])), ((x[idx+1], y2[idx+1])))
    return xc,yc

def KDE(x,y,xmin,xmax,ymin,ymax,val=100j):
    # Peform the kernel density estimate
    xx, yy = np.mgrid[xmin:xmax:val, ymin:ymax:val]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = stats.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    return(xx,yy,f)

def keys_list_from_dic(dic,search_key):
    '''
    return list of keys in dictionary

    Parameters
    ----------
    dic : dictionary
        input dictionary.
    search_key : str
        key to search for in dictionary.

    Returns
    -------
    All keys containg search_key in name (also partially).

    '''
    return([key for key,val in dic.items() if search_key in key])

def latex_table(rows,alignment=[],caption=''):
    if len(alignment)==0:alignment=["c"]*len(rows[0])
    table = Texttable()
    table.set_cols_align(alignment)
    table.set_deco(Texttable.HEADER | Texttable.VLINES)
    table.add_rows(rows)
    
    print('Tabulate Table:')
    print(tabulate(rows, headers='firstrow'))
    
    # print('\nTexttable Table:')
    # print(table.draw())
    
    # print('\nTabulate Latex:')
    # print(tabulate(rows, headers='firstrow', tablefmt='latex'))
    
    print('\nTexttable Latex:')
    print(draw_latex(table, caption=caption))

def leastsq_lin_fit(x,y,yerr=None,xerr=None,pinit = [0, 0]):
    fit_func = lambda p, x: p[0] * x  + p[1] 
    model=Model(fit_func)
    # if len(yerr)>0: data = scipy.odr.RealData(x, y, sy=yerr)
    # else: 
    data = RealData(x, y, sy=yerr, sx=xerr)
    odr = ODR(data, model, beta0=[0., 1.])
    out = odr.run()

    a=out.beta[0]
    aErr=out.sd_beta[0]
    b=out.beta[1]
    bErr=out.sd_beta[1]

    return(a,aErr,b,bErr)

# def mk_dir(mydir,verbose=False):
#     CHECK_FOLDER = os.path.isdir(mydir)
#
#     # If folder doesn't exist, then create it.
#     if not CHECK_FOLDER:
#         os.makedirs(mydir)
#         if verbose: print("created folder : ", mydir)
#
#     else:
#         if verbose: print(mydir, "folder already exists.")
  

def my_circle_scatter(axes, x_array, y_array, radius=0.5, **kwargs):
    for x, y in zip(x_array, y_array):
        circle = Circle((x,y), radius=radius, **kwargs)
        axes.add_patch(circle)


def match_catalogues(df1, df2, radec_col1, radec_col2, tolerance_mas=np.inf, frame='fk5'):
    """
    Matches two catalogues based on RA and DEC with a given tolerance in milliarcseconds.

    Parameters:
        df1 (pd.DataFrame): First catalogue with RA and DEC columns.
        df2 (pd.DataFrame): Second catalogue with RA and DEC columns.
        raddec_col1 (list): List of names of the RA column in df1 (degrees).
        raddec_col2 (list): List of names of the RA column in df2 (degrees).
        tolerance_mas (float): Tolerance in milliarcseconds for matching. Default: inf
        frame (str): SkyCoord frame. Default: fk5
    Returns:
        pd.DataFrame: DataFrame containing matched rows from both catalogues and their separations in milliarcseconds.
    """

    # Convert RA and DEC to SkyCoord objects
    coords1 = SkyCoord(ra=df1[radec_col1[0]].values * u.deg, dec=df1[radec_col1[1]].values * u.deg, frame=frame)
    coords2 = SkyCoord(ra=df2[radec_col2[0]].values * u.deg, dec=df2[radec_col2[1]].values * u.deg, frame=frame)

    # Define tolerance in milliarcseconds
    tolerance = tolerance_mas * u.mas

    # Find closest matches within the tolerance
    matches = []

    for i, coord1 in enumerate(coords1):
        sep = coord1.separation(coords2).to(u.mas)
        closest_match_index = sep.argmin()
        closest_separation = sep[closest_match_index]
        if closest_separation < tolerance:
            matches.append((i, closest_match_index, closest_separation.value))

    # Create a DataFrame of matches
    matches_df = pd.DataFrame(matches, columns=['Index_df1', 'Index_df2', 'Separation'])

    # Ensure the matched rows are unique
    matches_df.drop_duplicates(inplace=True)

    # Merge matched indices with original DataFrames
    matched_df1 = df1.loc[matches_df['Index_df1']].reset_index(drop=True)
    matched_df2 = df2.loc[matches_df['Index_df2']].reset_index(drop=True)

    # Concatenate the matched DataFrames
    matched_df = pd.concat([matched_df1, matched_df2, matches_df[['Separation']].reset_index(drop=True)], axis=1)
    matched_df.columns = [f"{col}_df1" for col in df1.columns] + [f"{col}_df2" for col in df2.columns] + ['Separation']

    return matched_df

def calc_chunksize(workers, ntarget, factor=4):
    """Calculate chunksize argument for Pool-methods.

    Resembles source-code within `multiprocessing.pool.Pool._map_async`.
    """
    chunksize, extra = divmod(ntarget, workers * factor)
    if extra:
        chunksize += 1
    return int(chunksize)

def parallelization_package(workers,ntarget,chunksize = None,verbose=True, factor=3):
    '''
    Provide basic input to parallelize the rourine

    Parameters
    ----------
    workers : int
        number of worker for the parallelization.
    ntarget : int
        number of targets.
    chunksize : int, optional
        size of each chunk for the parallelization process. The default is None.

    Returns
    -------
    None.

    '''
    ######## Split the workload over different CPUs ##########
    if workers==None: 
        workers=cpu_count()-2
        if workers<1: workers=1

    if chunksize == None:
        chunksize=calc_chunksize(workers, ntarget, factor=factor)
        if chunksize <= 0:
            chunksize = 1
        nchunks = ntarget // chunksize
    else: nchunks = ntarget // chunksize

    if workers > nchunks: workers=nchunks
    if verbose:
        getLogger(__name__).info(f'Max allowable workers {workers}, # of elements {ntarget} , # of chunk {nchunks} approx # of elemtent per chunks {chunksize} (chunksize)')
    return(workers,chunksize,ntarget)

def print_mean_median_and_std_sigmacut(ydata,eydata=[],sigma=3,maxiters=100,pre='',verbose=False,r=3,scientific=False,nonan=False,log=False):
    '''
    print useful statistics about the data

    Parameters
    ----------
    ydata : list
        input data.
    eydata : list, optional
        uncertanties on input data. The default is [].
    sigma : float, optional
        value of sigma cut. The default is 3.
    maxiters : int, optional
        maximum number of iteration for sigma cut to look for convergnece. 
        The default is 100.
    pre : str, optional
        pre append something to the standard output string. The default is ''.
    verbose : bool, optional
        choose to show prints. The default is False.
    r : int, optional
        rounding values for outputs. The default is 3.
    scientific : bool, optional
        choose to show scientific notation. The default is False.
    nonan : bool, optional
        choose to exclude NaN from data. The default is False.

    Returns
    -------
    (mean,median,standard deviation,mask applyed to the data).

    '''
    if log:
        ydata=np.log10(np.array(ydata))
        eydata=[]
    else:
        ydata=np.array(ydata)
        eydata=np.array(eydata)
    if len(eydata)>0:
        w=np.where(~np.isnan(ydata)&~np.isnan(eydata))
        eydata=eydata[w]
        ydata=ydata[w]
    elif nonan:
        w=np.where(~np.isnan(ydata))
        ydata=ydata[w]
    Mask=~sigma_clip(ydata,sigma_lower=sigma, sigma_upper=sigma,maxiters=maxiters,cenfunc='median').mask
    y_sc=np.array(ydata)[Mask]
    y_std_sc=round(np.nanstd(y_sc,ddof=1),r)
    if len(eydata)==0: y_mean_sc=round(np.nanmean(y_sc),r)
    else:
        eydata=np.array(eydata)[Mask]
        y_mean_sc=round(np.average(y_sc,weights=1/eydata),r)
    y_median_sc=round(np.nanmedian(y_sc),r)
    if verbose:
        print('Number of sources at %.1f sigma: %i'%(sigma,len(y_sc)))
        if scientific:print('%s mean %.3e median %.3e, std %.3e\n'%(pre,y_mean_sc,y_median_sc,y_std_sc))
        else:print('%s mean %s median %s, std %s\n'%(pre,y_mean_sc,y_median_sc,y_std_sc))
    return(y_mean_sc,y_median_sc,y_std_sc,Mask)

def plot_ND(args,plotND=True,xerror=None,yerror=None,zerror=None,x_label='x',y_label='y',z_label='z',name_label='var',elno=0,fig=None,color='k',pad=1,w_pad=1,h_pad=1,fx=1000,fy=750,size=3,width=1,row=1,col=1,showplot=False,subplot_titles=['Plot1'],marker_color='black',aspectmode='cube'):
    if showplot: 
        if plotND:
            fig = make_subplots(
                rows=1, cols=1,
                specs= [[{"type": "scatter3d"}]],
                subplot_titles=subplot_titles)
        else:
            fig = make_subplots(
                rows=1, cols=1,
                subplot_titles=subplot_titles)

    error_x=dict(type='data', array=xerror,visible=True)
    error_y=dict(type='data', array=yerror,visible=True)
    error_z=dict(type='data', array=zerror,visible=True)
    if plotND:    
        fig.add_trace(Scatter3d(args, error_x=error_x, error_y=error_y, error_z=error_z,
                                    mode='markers',
                                    marker=dict(size=size,line=dict(width=width),
                                    color=marker_color),
                                    name=name_label),
                                    row=row, col=col)
        fig.update_layout(autosize=True,width=fx,height=fy, margin=dict(l=10,r=10,b=10,t=22),paper_bgcolor="LightSteelBlue")
        fig.update_scenes(xaxis=dict(title_text=x_label),yaxis=dict(title_text=y_label),zaxis=dict(title_text=z_label),row=row,col=col,aspectmode=aspectmode)
    else:
        fig.add_trace(Scatter(args, error_x=error_x, error_y=error_y,
                                    mode='markers',
                                    marker=dict(size=size*3,line=dict(width=width),
                                    color=marker_color),
                                    name=name_label),
                                    row=row, col=col)
        fig.update_layout(autosize=False,width=fx,height=fy, margin=dict(l=10,r=10,b=10,t=10),paper_bgcolor="LightSteelBlue")
        # fig.update_scenes(xaxis=dict(title_text=x_label),yaxis=dict(title_text=y_label),row=row,col=col)
        fig.update_xaxes(title_text=x_label, row=row, col=col)
        fig.update_yaxes(title_text=y_label, row=row, col=col)

    if showplot: fig.show()
    else:return(fig)
    
def power_law_fitting(xdata,ydata,yerr,xerr=None,npoints=1000,showplot=False,verbose=True):
    xdata=np.array(xdata)
    ydata=np.array(ydata)
    yerr=np.array(yerr)   
    powerlaw=lambda x, amp, index: amp * (x**(-index))
    # Define function for calculating a power law
    ##########
    # Fitting the data -- Least Squares Method
    ##########
    
    # Power-law fitting is best done by first converting
    # to a linear equation and then fitting to a straight line.
    #
    #  y = a * x^(-b)
    #  log(y) = log(a) - b*log(x)
    #

    logx = np.log10(xdata)
    logy = np.log10(ydata)
    logyerr = yerr / ydata
    if np.any(xerr!=None): logxerr = xerr / xdata
    else: logxerr=None
    
    a,aErr,b,bErr=leastsq_lin_fit(logx,logy,logyerr,xerr=logxerr,pinit = [1.0, -1.0])

    index = -a
    amp = 10.0**b
    indexErr = aErr
    ampErr = bErr * amp

    ##########
    # Plotting data
    ##########
    xnewdata=np.linspace(min(xdata),max(xdata),npoints)
    
    powerlaw_list=powerlaw(xnewdata, amp, index)
    
    if showplot==True:
        fig,ax=plt.subplots(2,1,figsize=(10,6))
        fig.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.1,hspace=0.5)
        fig.text(0.5, 1, 'Ampli = %.4f +/- %.4f' % (amp, ampErr), horizontalalignment='center',fontsize=15)
        fig.text(0.5, 0.95, 'Index = %.4f +/- %.4f' % (index, indexErr), horizontalalignment='center',fontsize=15)

        ax[0].plot(xnewdata, powerlaw_list)     # Fit
        ax[0].errorbar(xdata, ydata, yerr=yerr, fmt='k.')  # Data
        ax[0].set_title(r'Best Power Law Fit for x$^{-index}$')
        ax[0].set_xlabel('X')
        ax[0].set_ylabel('Y')

        ax[1].loglog(xnewdata, powerlaw_list)
        ax[1].errorbar(xdata, ydata, yerr=yerr, fmt='k.')  # Data
        ax[1].set_xlabel('X (log scale)')
        ax[1].set_ylabel('Y (log scale)')
        # plt.show()
    return(powerlaw_list,xnewdata,amp, ampErr, index, indexErr)

def rotate_point(origin, point, angle,r=2):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    Origin and point should be given as a tuple (,).
    The angle can be given in degree.
    """
    angle=math.radians(angle)
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return(np.round(qx,r),np.round(qy,r))


def round_down(n, decimals=0):
    '''
    round down to the close number

    Parameters
    ----------
    n : float
        input number.
    decimals : int, optional
        number of decimals. The default is 0.

    Returns
    -------
    output rounded number
    '''
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier

def round_up(n, decimals=0):
    '''
    round up to the close number

    Parameters
    ----------
    n : float
        input number.
    decimals : int, optional
        number of decimals. The default is 0.

    Returns
    -------
    output rounded number
    '''
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier

def round2closerint(list_of_numbers,base=1,integer=True):
    '''
    round list to closer integer number

    Parameters
    ----------
    list_of_numbers : list
        input list of numbers.
    base : int, optional
        DESCRIPTION. The default is 1.
    integer : bool, optional
        return integer number. The default is True.

    Returns
    -------
    None.

    '''
    if not isinstance(list_of_numbers,(list,np.ndarray)): list_of_numbers=[list_of_numbers]
    rounded_list=[]
    for elno in list_of_numbers:
        rounded=np.round(elno*base)/base
        if integer==True: rounded=rounded.astype(int)
        rounded_list.append(rounded.tolist())
    return(np.array(rounded_list))

def round2closerhalf(number):
    """Round a number to the closest half integer.
    >>> round_of_rating(1.3)
    1.5
    >>> round_of_rating(2.6)
    2.5
    >>> round_of_rating(3.0)
    3.0
    >>> round_of_rating(4.1)
    4.0"""

    return round(number * 2) / 2

# def PointsInCircum(radius,h=0,k=0,rmin=0,r=2):
def PointsInCircum(radius,h=0,k=0,r=2):
    '''
    Create a point that lie on a specific circumference

    Parameters
    ----------
    radius : float
        radius of circumference.
    h,k : float, optional
        coordiantes of the center of the cirumference. The default is 0.
    # rmin : , optional
    #     DESCRIPTION. The default is 0.
    r : int, optional
        rounding values for outputs. The default is 3.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    # if rmin!=0: 
    #     while radius==0:
    #         radius=random.uniform(rmin,radius)
    theta = random() * 2 * math.pi
    return round(h + math.cos(theta) * radius,r), round(k + math.sin(theta) * radius,r)

def poly_regress(ax,X,y,degree=4,showplot=False,X_label='X',y_label='y',title='',Xn=None):
    '''
    Ploynomial regression

    Parameters
    ----------
    ax : None or matplotlib axis
        axis on which project the plot.
    X : list
        list of x inputs for the polynomial regression.
    y : list
        list of y inputs for the polynomial regression.
    degree : int, optional
        degree of the polynomial regression. The default is 4.
    showplot : bool, optional
        choose to show plots. The default is False.
    X_label : str, optional
        label of the x axis in plot. The default is 'X'.
    y_label : str, optional
        label of the x axis in plot. The default is 'y'.
    title : str, optional
        title lable for plot. The default is ''.
    Xn : float, optional
        target input X position to show on plot. The default is None.

    Returns
    -------
    (Generated polynomial and interaction features,Ordinary least squares Linear Regression).

    '''
    pre_process = PolynomialFeatures(degree=degree)
    # Transform our x input to 1, x and x^2
    X_poly = pre_process.fit_transform(X)
    # Show the transformation on the notebook
    pr_model = LinearRegression()
    # Fit our preprocessed data to the polynomial regression model
    pr_model.fit(X_poly, y)
    # Store our predicted Humidity values in the variable y_new
    y_pred = pr_model.predict(X_poly)

    if showplot==True:
        # Plot our model on our data

        ax.set_title(title)
        ax.set_xlabel(X_label)
        ax.set_ylabel(y_label)
        ax.scatter(X, y, c = "black")
        if Xn!=None:
            yn = pr_model.predict(pre_process.fit_transform([[Xn]]))
            ax.scatter(Xn, yn, c = "red")
        ax.plot(X, y_pred)
    return(pre_process,pr_model)

def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)

def trim_axs(axs, N):
    """
    Reduce *axs* to *N* Axes. All further Axes are removed from the figure.
    """
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap    

def auc(q, m0, std1, std2):
    if q[0] <= m0:std_1=std1
    else: std_1=std2
    if q[1] <= m0: std_2=std1
    else: std_2=std2

    area = norm.cdf(q[1], m0, std_2) - norm.cdf(q[0], m0, std_1)

    return area


def solve(m1, m2, std1, std2):
    a = 1 / (2 * std1 ** 2) - 1 / (2 * std2 ** 2)
    b = m2 / (std2 ** 2) - m1 / (std1 ** 2)
    c = m1 ** 2 / (2 * std1 ** 2) - m2 ** 2 / (2 * std2 ** 2) - np.log(std2 / std1)
    return np.roots([a, b, c])

def flat_list(l):
    flattened_list=[]
    for sublist in l:
        if isinstance(sublist, list):
            for item in sublist:
                flattened_list.append(item)
        else:
            flattened_list.append(sublist)

    return flattened_list