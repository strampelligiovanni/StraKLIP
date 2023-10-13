"""
utilities functions that can be use by or with the photometry class
"""
import sys
sys.path.append('/')
from photometry import Detection,photometry_AP,flux_converter
from ancillary import poly_regress
from tiles import Tile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.ma as ma

from astropy.io import fits
from astropy.stats import sigma_clip,sigma_clipped_stats
from pathlib import Path
from reftools.interpretdq import ImageDQ, DQParser
from scipy.interpolate import interp1d
from photutils.segmentation import SegmentationImage
from stralog import getLogger

def get_rough_sky(data,sigma=3.0,nsigma=2, npixels=5, dilate_size=11,mask_stars=False):
    '''
    Get rough estimate of sky in image

    Parameters
    ----------
    data : numpy ndarray
        target image.
    sigma : float, optional
        value of the sigma cut. The default is 3.0.
    nsigma : float, optional
        The number of standard deviations per pixel above the background for 
        which to consider a pixel as possibly being part of a source. The default is 2.
    npixels : int, optional
        The number of connected pixels, each greater than threshold, that an 
        object must have to be detected. npixels. The default is 5.
    dilate_size : TYPE, optional
        The size of the square array used to dilate the segmentation image. 
        The default is 11.
    mask_stars : bool, optional
        choose to mask stars in image before apply sigma cut. The default is False.

    Returns
    -------
    None.

    '''
    if mask_stars:mask = SegmentationImage.make_source_mask(data, nsigma=nsigma, npixels=npixels, dilate_size=dilate_size)
    else:mask=None
    mean, median, std = sigma_clipped_stats(data, sigma=sigma,mask=mask)
    return(mean, median, std)

def KLIP_throughput(DF_fk,candidate_sep,filter,magbin,dmag,Kmode,verbose=False):
    '''
    evaluate KLIP throughput from fake injections

    Parameters
    ----------
    DF : pandas DataFrame
        DataFrame containing the fake injections.
    candidate_sep : float
        separation between primary and target candidate.
    filter : str
        fiter name.
    magbin : int
        primary magnitude bin index in the fake injection dataframe.
    dmag : int
        companion delta magnitude (contrast) index in the fake injection dataframe.
    Kmode : int
        KLIP mode index in the fake injection dataframe.
    verbose : boool, optional
        choose to show prints. The default is False.

    Returns
    -------
    (throuput,e_throuput).

    '''
    median_dcount=[]
    std_dcount=[]
    sep_list=DF_fk.fk_candidates_df.index.get_level_values('sep').unique()
    for sep in sep_list:
        injected_counts=DF_fk.fk_candidates_df.loc[(filter,magbin,dmag,sep),['counts']].values.ravel().astype(float)
        retrived_counts=DF_fk.fk_candidates_df.loc[(filter,magbin,dmag,sep),['counts_Kmode%s'%(Kmode)]].values.ravel().astype(float)
        retrived_nsigma=DF_fk.fk_candidates_df.loc[(filter,magbin,dmag,sep),['Nsigma_Kmode%s'%(Kmode)]].values.ravel().astype(float)
        filtered_dcounts = sigma_clip(retrived_counts[retrived_nsigma>0]/injected_counts[retrived_nsigma>0], sigma=3, maxiters=10)
        median_masked=np.nanmedian(filtered_dcounts.data[~filtered_dcounts.mask])
        std_masked=np.nanstd(filtered_dcounts.data[~filtered_dcounts.mask])
        if np.isnan(median_masked):
            median_dcount.append(0)
            std_dcount.append(0)    
        else:
            median_dcount.append(median_masked)
            std_dcount.append(std_masked)    

    X=np.array(sep_list).reshape(-1, 1)
    if verbose==True: fig,ax=plt.subplots(1,2,figsize=(10,5))
    else: ax=[None,None]
    pre_process_median,pr_model_median=poly_regress(ax[0],X,median_dcount,degree=3,showplot=verbose,X_label='Separation',y_label='counts',title='',Xn=candidate_sep)
    pre_process_std,pr_model_std=poly_regress(ax[1],X,std_dcount,degree=3,showplot=verbose,X_label='Separation',y_label='counts',title='',Xn=candidate_sep)
    if verbose:plt.show()
    return(pr_model_median.predict(pre_process_median.fit_transform([[candidate_sep]])),pr_model_std.predict(pre_process_std.fit_transform([[candidate_sep]])))

def aperture_photometry_handler(DF,id,filter,data_label='',dq_label='',hdul=None,data=[],dqdata=None,ext=None,x=None,y=None,bpx_list=[],spx_list=[],la_cr_remove=False,cr_radius=1,Python_origin=False,forcedSky=[],exptime=1,zpt=0,ezpt=0,thrpt=1,ethrpt=0,sat_thr=np.inf,radius_a=3,radius1=5,radius2=15,aptype='circular',df_label='mvs_tiles_df',df_ids='mvs_ids',kill_plots=False,noBGsub=False,grow_curves=False,ee_df=None,sigma=3,delta=3,gstep=0.01,p=5,r_in=6,r_min=6,r_max=15,multiply_by_exptime=False,multiply_by_gain=False,multiply_by_PAM=False,ROTA=None,PAV3=None,gain=None):
    '''
    Wrapper for the photometry routines on target tile.

    Parameters
    ----------
    DF : pandas DataFrame
        input DataFrame containing all the tiles.
    id : int
        target id. Can be either the average or the multivisit ID depending on df_ids.
    filter : float
        filter name.
    x,y : float
        Coordiante of the brighter pixel in the tile where to anchor the photometry to.
        Could be either coordinates on a scientific image or a tile depending on input.
    bpx_list: list
        list of values to flag as bad pixels using the data quality image (coud be real bad pixels, hot/warm pixels, cosmic rays etc.)
    spx_list: list
        list of values to flag as saturated pixels using the data quality image 
    input : str, optional
        Choose between {tile, sci}. Help the handler interpret the input positions 
        provided through pos.
        Tile means the input image is a tile image created with the pipeline.
        Sci means the input image is an scientific fits image. 
        The default is tile.
    exptime : float, optional
        exposure time of the image. The default is 1.
    zpt : float, optional
        Zero Point for photometry. The default is 0.
    ezpt : float, optional
        error on the Zero Point to propagate to the magnitude uncertanty. The default is 0.
    hdul : numpy ndarray, optional
        target hdul. If None, look in DF. The default is None.
    Kmode : int, optional
        KLIP mode of the companion detection. If None work on targets. The default is None.
    thrpt : float, optional
        KLIP thorughput for companion photometry. The default is 1.
    ethrpt : float, optional
        error on the KLIP thorughput to propagate on companion photometry. The default is 0.
    radius_a : Int, optional
        radius of circula aperture in arcsecond for aperture photometry. The default is 0.4.
    radius1 : Int, optional
        radius of circula aperture in pixels or inner radius for annulus aperture. The default is 10.
    radius2 : Int or None, optional
        outer radius for annulus aperture. if None use circular aperture. The default is 15.
    aptype : (circular,square,4pixels), optional
       defin the aperture type to use during aperture photometry. 
       The default is 'cirular'.
    df_label : str, optional
        label name of the dataframe. 
        choose between:
            'avg_tiles_df'
            'mvs_tiles_df'
        The default is 'mvs_tiles_df'.
    df_ids : str, optional
        label name of the ID dataframe. 
        choose between:
            'avg_ids'
            'mvs_ids'
        The default is 'mvs_ids'.
    showplot : bool, optional
        choose to show plots. The default is False.
    noBGsub : bool, optional
        choose to apply background subtraction on tile. The default is False.
    sigma : float, optional
        value of sigma clip. The default is 3.
    delta : int, optional
        step to create the square mask in range -delta, x, +delt and -delta, y, +delta. The default is 1.
    multiply_by_exptime:
        to properly evaluate the error you need the images total counts (not divided by exptime). 
        If that the case, this option will multiply the image but the corresponding exptime. The default is False.
    multiply_by_gain:
        to properly evaluate evaluate photometry you need to convert the counts in electrons (multiply by the gain). 
        If that not the case, this option will multiply the image but the corresponding gain. The default is False.
    multiply_by_PAM:
        to properly evaluate photometry accross the FLT you need a not distorted images. 
        If that the case, this option will multiply the image but the corresponding PAM. The default is False.
        
    Returns
    -------
    (counts,e_counts,mag,e_mag).

    '''
    getLogger(__name__).info(f'Performing aperture photometry on tile_ID{id}.fits')
    radius = radius_a
    rows=3
    if not noBGsub:rows+=1
    if grow_curves: rows+=1

    if not kill_plots:
        fig,ax=plt.subplots(1,rows,figsize=(7*rows,5))
        if ROTA==None: ROTA=DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==id,'%s_rota'%filter].values[0]
        if PAV3==None:PAV3=DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==id,'%s_pav3'%filter].values[0]

    else:
        if ROTA==None: ROTA=0
        if PAV3==None:PAV3=0
        fig,ax=[None,[None]*rows]
        
    if isinstance(ee_df,pd.DataFrame): 
        eex=ee_df.loc[ee_df.FILTER==filter].columns[1:].astype(float)
        eey=ee_df.loc[ee_df.FILTER==filter].values[0][1:].astype(float)
        f = interp1d(eex, eey)
        Ei=f(radius_a*DF.pixscale)
    elif isinstance(ee_df,dict):
        ext=ROTA=DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==id,'ext'].values[0].astype(int)
        ee_df0=ee_df[ext]
        eex=ee_df0.loc[ee_df0.FILTER==filter].columns[1:].astype(float)
        eey=ee_df0.loc[ee_df0.FILTER==filter].values[0][1:].astype(float)
        f = interp1d(eex, eey)
        Ei=f(radius_a*DF.pixscale)
    else:
        Ei=1
    
    if not isinstance(data, (list,np.ndarray)): 
        data=hdul[ext].data.copy()
        if multiply_by_exptime: 
            data*=exptime
        if multiply_by_gain: 
            data*=DF.gain
            
        if multiply_by_PAM: 
            path2PAM='%s/PAM'%(DF.path2data)
            phdul=fits.open(path2PAM+'/'+str(DF.PAMdict[ext]+'.fits'))
            try:PAM=phdul[1].data
            except:PAM=phdul[0].data
            data*=PAM

        if x==None: x=DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==id,'x_%s'%filter].values[0]
        if y==None: y=DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==id,'y_%s'%filter].values[0]
        if ext == None: ext=int(DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==id].ext.values[0])
        dqdata=hdul[ext+2].data
        DQDATA=Tile(data=dqdata,x=x,y=y,tile_base=radius2*2+5,inst=DF.inst,Python_origin=Python_origin)
        DQDATA.mk_tile(pad_data=True,legend=False,showplot=False,verbose=False,title='DQ',kill_plots=True,cbar=True)
        dq=DQDATA.data
        bpx,spx=read_dq_from_tile(DF,dq=dq,bpx_list=bpx_list,spx_list=spx_list)

        if not kill_plots:
            DQDATA=Tile(data=dq,x=DQDATA.x_tile,y=DQDATA.y_tile,tile_base=DQDATA.tile_base,inst=DF.inst,Python_origin=Python_origin)
            DQDATA.mk_tile(fig=fig,ax=ax[2],pad_data=False,legend=False,showplot=False,verbose=False,title='shiftedDQ',kill_plots=kill_plots,cbar=True)
    
    else:
        if x==None: x=(DF.tile_base-1)/2
        if y==None: y=(DF.tile_base-1)/2
    
        if isinstance(dqdata, (list,np.ndarray)): 
            DQDATA=Tile(data=dqdata,x=x,y=y,tile_base=DF.tile_base,inst=DF.inst,Python_origin=Python_origin)
            DQDATA.mk_tile(pad_data=False,legend=False,showplot=False,verbose=False,title='shiftedDQ',kill_plots=True,cbar=True)
            dq=DQDATA.data
            bpx,spx=read_dq_from_tile(DF,dq=dq,bpx_list=bpx_list,spx_list=spx_list)
    
            if not kill_plots:
                DQDATA=Tile(data=dq,x=DQDATA.x_tile,y=DQDATA.y_tile,tile_base=DQDATA.tile_base,inst=DF.inst,Python_origin=Python_origin)
                DQDATA.mk_tile(fig=fig,ax=ax[2],pad_data=False,legend=False,showplot=False,verbose=False,title='shiftedDQ',kill_plots=kill_plots,cbar=True)
    
        else:
            dq=[]
            # dq_bkg=[]
            spx=np.nan
            bpx=np.nan

    if np.all(np.isnan(data)):
        data=np.ones((DF.tile_base,DF.tile_base))*np.nan


    if noBGsub:
        if len(forcedSky)>0:
            Sky,eSky,nSky=forcedSky
        else: Sky,eSky,nSky=[0,1,1]
        DATA=Tile(data=data,x=x,y=y,tile_base=radius2*2+5,inst=DF.inst,Python_origin=Python_origin,dqdata=dq)
        DATA.mk_tile(fig=fig,ax=ax[0],la_cr_remove=la_cr_remove,pad_data=True,verbose=False,xy_m=False,legend=False,showplot=False,keep_size=False,xy_dmax=None,cbar=True,title='ID%i ROTA %i PAV3 %i'%(id,ROTA,PAV3),return_tile=False,kill_plots=kill_plots)

        detection_AP=Detection(DATA.data,DATA.x_tile,DATA.y_tile,Sky=Sky,nSky=nSky,eSky=eSky,thrpt=thrpt,ethrpt=ethrpt)
            
    else:
        DATA=Tile(data=data,x=x,y=y,tile_base=radius2*2+5,inst=DF.inst,Python_origin=Python_origin,dqdata=dq)
        DATA.mk_tile(fig=fig,ax=ax[0],la_cr_remove=la_cr_remove,cr_radius=cr_radius,pad_data=True,verbose=False,xy_m=False,legend=False,showplot=False,keep_size=False,xy_dmax=None,cbar=True,title='ID%i ROTA %i PAV3 %i'%(id,ROTA,PAV3),return_tile=False,kill_plots=kill_plots)
        
        if aptype=='4pixels': raise ValueError('aptype = 4pixels not supported for primaries when evaluating Sky')
            # bkg=Detection(data,x,y)
            # photometry_AP.aperture_mask(bkg,aptype='4pixels')
            # photometry_AP.mask_aperture_data(bkg)
            # photometry_AP.aperture_stats(bkg,aperture=bkg.aperture,sigmaclip=True,sigma=sigma,sat_thr=sat_thr,fill=np.nan)
            # Sky=round(np.nanmean(bkg.data_mask_out[bkg.data_mask_out>0]),3)
            # eSky=np.nanstd(bkg.data_mask_out[bkg.data_mask_out>0],ddof=1)
            # nSky=len(bkg.data_mask_out[bkg.data_mask_out>0])
    
            # if not kill_plots:
            #     BG = Tile(data=bkg.data_mask_out, x=DATA.x_tile, y=DATA.y_tile,
            #                     tile_base=bkg.data_mask_out.shape[0], inst=DF.inst, Python_origin=False)
            #     BG.mk_tile(fig=fig, ax=ax[3], pad_data=False, verbose=False, xy_m=False, legend=False, showplot=False,
            #                keep_size=True, xy_dmax=None, cbar=True, title='Sky Area', return_tile=False, kill_plots=kill_plots)
    
            # detection_AP=Detection(DATA.data,DATA.x_tile,DATA.y_tile,Sky=Sky,nSky=nSky,eSky=eSky,thrpt=thrpt,ethrpt=ethrpt,Ei=Ei)
            # Sky,eSky,nSky=[bkg.median,bkg.std,bkg.Nap]          
    
        else:
            # if candidate:
            #     bkg=Detection(data,x,y)
            #     photometry_AP.aperture_mask(bkg,aptype=aptype)
            #     photometry_AP.mask_aperture_data(bkg)
            #     photometry_AP.aperture_stats(bkg,aperture=bkg.aperture,sigmaclip=True,sigma=sigma,sat_thr=sat_thr,fill=np.nan)
            #     Sky=round(np.nanmean(bkg.data_mask_out[bkg.data_mask_out>0]),3)
            #     eSky=np.nanstd(bkg.data_mask_out[bkg.data_mask_out>0],ddof=1)
            #     nSky=len(bkg.data_mask_out[bkg.data_mask_out>0])
        
            #     if not kill_plots:
            #         BG = Tile(data=bkg.data_mask_out, x=DATA.x_tile, y=DATA.y_tile,
            #                         tile_base=bkg.data_mask_out.shape[0], inst=DF.inst, Python_origin=False)
            #         BG.mk_tile(fig=fig, ax=ax[3], pad_data=False, verbose=False, xy_m=False, legend=False, showplot=False,
            #                    keep_size=True, xy_dmax=None, cbar=True, title='Sky Area', return_tile=False, kill_plots=kill_plots)
        
            #     detection_AP=Detection(DATA.data,DATA.x_tile,DATA.y_tile,Sky=Sky,nSky=nSky,eSky=eSky,thrpt=thrpt,ethrpt=ethrpt,Ei=Ei)
            #     Sky,eSky,nSky=[bkg.median,bkg.std,bkg.Nap]   
            
            # else:
            bkg=Detection(DATA.data,DATA.x_tile,DATA.y_tile)
            
            photometry_AP.aperture_mask(bkg,aptype=aptype,radius1=radius1,radius2=radius2,method='center',ap_x=delta,ap_y=delta)
            photometry_AP.mask_aperture_data(bkg)
            photometry_AP.aperture_stats(bkg,aperture=bkg.aperture,sigmaclip=True,sigma=sigma,sat_thr=sat_thr,fill=np.nan)

            if not kill_plots:
                BG = Tile(data=bkg.data_mask_out, x=DATA.x_tile, y=DATA.y_tile,
                            tile_base=bkg.data_mask_out.shape[0], inst=DF.inst, Python_origin=False)
                BG.mk_tile(fig=fig, ax=ax[3], pad_data=False, verbose=False, xy_m=False, legend=False, showplot=False,
                           keep_size=True, xy_dmax=None, cbar=True, title='Sky Area', return_tile=False, kill_plots=kill_plots)
            Sky=bkg.median
            if grow_curves:
                detection_AP=Detection(DATA.data,DATA.x_tile,DATA.y_tile,Sky=Sky,nSky=bkg.Nap,eSky=bkg.std)
                photometry_AP.grow_curves(detection_AP,fig=fig,ax=ax[4],showplot=False,gstep=gstep,p=p,r_in=r_in,r_min=r_min,r_max=r_max)
                grow_corr=detection_AP.grow_corr
                Sky=(1-grow_corr/100)*detection_AP.Sky
    

                
            else:grow_corr=0
            if not kill_plots:
                circle1 = plt.Circle((DATA.x_tile,DATA.y_tile), radius, color='k',fill=False)
                circle2 = plt.Circle((DATA.x_tile,DATA.y_tile), radius1, color='g',fill=False)
                circle3 = plt.Circle((DATA.x_tile,DATA.y_tile), radius2, color='r',fill=False)
                ax[0].add_patch(circle1)
                ax[0].add_patch(circle2)
                ax[0].add_patch(circle3)  
                
            nSky=bkg.Nap
            eSky=bkg.std
            detection_AP=Detection(DATA.data,DATA.x_tile,DATA.y_tile,Sky=Sky,nSky=nSky,eSky=eSky,thrpt=thrpt,ethrpt=ethrpt,grow_corr=grow_corr,Ei=Ei)
            Sky,eSky,nSky=[bkg.median,bkg.std,bkg.Nap]

    photometry_AP.aperture_mask(detection_AP,aptype=aptype,radius1=radius,ap_x=delta,ap_y=delta)
    photometry_AP.mask_aperture_data(detection_AP)
    photometry_AP.aperture_stats(detection_AP,aperture=detection_AP.aperture,sat_thr=sat_thr,fill=np.nan)
    if not kill_plots:
        AP=Tile(data=detection_AP.data_mask_in,x=DATA.x_tile,y=DATA.y_tile,tile_base=detection_AP.data_mask_in.shape[1],inst=DF.inst,raise_errors=False)
        AP.mk_tile(fig=fig,ax=ax[1],showplot=False,keep_size=True,cbar=True,title='Aperture Area',simplenorm='sqrt',return_tile=False,kill_plots=kill_plots)
     
    flux_converter.counts_and_errors(detection_AP)
    if gain==None: gain=DF.gain
    flux_converter.flux2mag(detection_AP,exptime=exptime,zpt=zpt,ezpt=ezpt,gain=gain)
    if not kill_plots:
        plt.show()
        if not noBGsub or len(forcedSky)>0:print('bkg median %.3f, std %.3f, nSky %i'%(Sky,eSky,nSky))
        print('Star counts %e, ecounts %e, Nap %i, Nsigma %.3f, mag %.3f, emag %.3f'%(detection_AP.counts,detection_AP.ecounts,detection_AP.Nap,detection_AP.Nsigma,detection_AP.mag,detection_AP.emag))
        if grow_curves:print('Grow curve correction: %.3f'%(detection_AP.grow_corr))

    del data
    return(detection_AP.counts,detection_AP.ecounts,detection_AP.Nsigma,detection_AP.Nap,detection_AP.mag,detection_AP.emag,spx,bpx,Sky,eSky,nSky,detection_AP.grow_corr)

def KLIP_aperture_photometry_handler(DF,id,filter,data_label='',dq_label='',hdul=None,data=[],dqdata=None,ext=None,x=None,y=None,la_cr_remove=False,Python_origin=False,forcedSky=[],exptime=1,zpt=0,ezpt=0,thrpt=1,ethrpt=0,sat_thr=np.inf,radius_a=3,radius1=5,radius2=15,aptype='circular',df_label='mvs_tiles_df',df_ids='mvs_ids',kill_plots=False,noBGsub=False,grow_curves=False,ee_df=None,sigma=3,delta=3,gstep=0.01,p=5,r_in=6,r_min=6,r_max=15,multiply_by_exptime=False,multiply_by_gain=False,multiply_by_PAM=False,ROTA=None,PAV3=None,gain=None):
    '''
    Wrapper for the photometry routines on target tile.

    Parameters
    ----------
    DF : pandas DataFrame
        input DataFrame containing all the tiles.
    id : int
        target id. Can be either the average or the multivisit ID depending on df_ids.
    filter : float
        filter name.
    pos : list
        Coordiante of the brighter pixel in the tile where to anchor the photometry to.
        Could be either coordinates on a scientific image or a tile depending on input.
    input : str, optional
        Choose between {tile, sci}. Help the handler interpret the input positions 
        provided through pos.
        Tile means the input image is a tile image created with the pipeline.
        Sci means the input image is an scientific fits image. 
        The default is tile.
    exptime : float, optional
        exposure time of the image. The default is 1.
    zpt : float, optional
        Zero Point for photometry. The default is 0.
    ezpt : float, optional
        error on the Zero Point to propagate to the magnitude uncertanty. The default is 0.
    hdul : numpy ndarray, optional
        target hdul. If None, look in DF. The default is None.
    Kmode : int, optional
        KLIP mode of the companion detection. If None work on targets. The default is None.
    thrpt : float, optional
        KLIP thorughput for companion photometry. The default is 1.
    ethrpt : float, optional
        error on the KLIP thorughput to propagate on companion photometry. The default is 0.
    radius_a : Int, optional
        radius of circula aperture in arcsecond for aperture photometry. The default is 0.4.
    radius1 : Int, optional
        radius of circula aperture in pixels or inner radius for annulus aperture. The default is 10.
    radius2 : Int or None, optional
        outer radius for annulus aperture. if None use circular aperture. The default is 15.
    aptype : (circular,square,4pixels), optional
       defin the aperture type to use during aperture photometry. 
       The default is 'cirular'.
    df_label : str, optional
        label name of the dataframe. 
        choose between:
            'avg_tiles_df'
            'mvs_tiles_df'
        The default is 'mvs_tiles_df'.
    df_ids : str, optional
        label name of the ID dataframe. 
        choose between:
            'avg_ids'
            'mvs_ids'
        The default is 'mvs_ids'.
    showplot : bool, optional
        choose to show plots. The default is False.
    noBGsub : bool, optional
        choose to apply background subtraction on tile. The default is False.
    sigma : float, optional
        value of sigma clip. The default is 3.
    delta : int, optional
        step to create the square mask in range -delta, x, +delt and -delta, y, +delta. The default is 1.
    multiply_by_exptime:
        to properly evaluate the error you need the images total counts (not divided by exptime). 
        If that the case, this option will multiply the image but the corresponding exptime. The default is False.
    multiply_by_gain:
        to properly evaluate evaluate photometry you need to convert the counts in electrons (multiply by the gain). 
        If that not the case, this option will multiply the image but the corresponding gain. The default is False.
    multiply_by_PAM:
        to properly evaluate photometry accross the FLT you need a not distorted images. 
        If that the case, this option will multiply the image but the corresponding PAM. The default is False.
        
    Returns
    -------
    (counts,e_counts,mag,e_mag).

    '''
    radius = radius_a
    rows=3

    if not kill_plots:
        fig,ax=plt.subplots(1,rows,figsize=(7*rows,5))
        if ROTA==None: ROTA=DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==id,'%s_rota'%filter].values[0]
        if PAV3==None:PAV3=DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==id,'%s_pav3'%filter].values[0]

    else:
        if ROTA==None: ROTA=0
        if PAV3==None:PAV3=0
        fig,ax=[None,[None]*rows]
        
    x_tile=int(DF.tile_base-1)/2
    y_tile=int(DF.tile_base-1)/2
    if x==None: x=x_tile
    if y==None: y=y_tile

    # dq=[]
    spx=np.nan
    bpx=np.nan
    if not kill_plots:
        DATA=Tile(data=data,x=x,y=y,tile_base=DF.tile_base,inst=DF.inst,Python_origin=Python_origin)
        DATA.mk_tile(fig=fig,ax=ax[0],la_cr_remove=la_cr_remove,pad_data=True,verbose=False,xy_m=False,legend=False,showplot=False,keep_size=False,xy_dmax=None,cbar=True,title='ID%i ROTA %i PAV3 %i'%(id,ROTA,PAV3),return_tile=False,kill_plots=kill_plots)
    
    if noBGsub:
        if DF.skip_photometry:
            DQ_list=list(set(dqdata.ravel()))
            xdata = data.copy()
            mask_x=dqdata.copy()
            for i in [i for i in DQ_list if i not in DF.DQ_values2mask_list]:  mask_x[(mask_x==i)]=0
            mx = ma.masked_array(xdata, mask=mask_x)
            mx.data[mx.mask]=-9999
            mx.data[mx.data<0]=0


            sky_clipped=sigma_clip(mx.data, sigma=sigma)
            Sky= np.nanmedian(sky_clipped[~sky_clipped.mask])
            eSky=np.nanstd(sky_clipped[~sky_clipped.mask])
            nSky=len(sky_clipped[~sky_clipped.mask])
        elif not DF.skip_photometry and len(forcedSky)>0:
            Sky,eSky,nSky=forcedSky
        else: Sky,eSky,nSky=[0,1,1]
    else:
        bkg=Detection(data,x,y)
        photometry_AP.aperture_mask(bkg,aptype=aptype,radius1=radius1,radius2=radius2,method='center',ap_x=delta,ap_y=delta)
        photometry_AP.mask_aperture_data(bkg)
        photometry_AP.aperture_stats(bkg,aperture=bkg.aperture,sigma=sigma,sat_thr=sat_thr,fill=np.nan)
        if not kill_plots:
            BG = Tile(data=bkg.data_mask_out, x=x_tile, y=y_tile,
                            tile_base=bkg.data_mask_out.shape[0], inst=DF.inst, Python_origin=False)
            BG.mk_tile(fig=fig, ax=ax[2], pad_data=False, verbose=False, xy_m=False, legend=False, showplot=False,
                       keep_size=True, xy_dmax=None, cbar=True, title='Sky Area', return_tile=False, kill_plots=kill_plots)

        
        Sky=round(np.nanmean(bkg.data_mask_out[bkg.data_mask_out>0]),3)
        eSky=np.nanstd(bkg.data_mask_out[bkg.data_mask_out>0],ddof=1)
        nSky=len(bkg.data_mask_out[bkg.data_mask_out>0])
        
    detection_AP=Detection(data,x,y,Sky=Sky,nSky=nSky,eSky=eSky,thrpt=thrpt,ethrpt=ethrpt)
    photometry_AP.aperture_mask(detection_AP,aptype=aptype,radius1=radius,ap_x=delta,ap_y=delta)
    photometry_AP.mask_aperture_data(detection_AP)
    photometry_AP.aperture_stats(detection_AP,aperture=detection_AP.aperture,sat_thr=sat_thr,fill=np.nan)
    if not kill_plots:
        AP=Tile(data=detection_AP.data_mask_in,x=x_tile,y=y_tile,tile_base=detection_AP.data_mask_in.shape[1],inst=DF.inst,raise_errors=False)
        AP.mk_tile(fig=fig,ax=ax[1],showplot=False,keep_size=True,cbar=True,title='Aperture Area',simplenorm='sqrt',return_tile=False,kill_plots=kill_plots)
     
    flux_converter.counts_and_errors(detection_AP)
    if gain==None: gain=DF.gain
    flux_converter.flux2mag(detection_AP,exptime=exptime,zpt=zpt,ezpt=ezpt,gain=gain)
    if not kill_plots:
        plt.show()
        if not noBGsub or len(forcedSky)>0:print('bkg median %.3f, std %.3f, nSky %.1f'%(Sky,eSky,nSky))
        print('Star counts %e, ecounts %e, Nap %i, Nsigma %.3f, mag %.3f, emag %.3f'%(detection_AP.counts,detection_AP.ecounts,detection_AP.Nap,detection_AP.Nsigma,detection_AP.mag,detection_AP.emag))
        if grow_curves:print('Grow curve correction: %.3f'%(detection_AP.grow_corr))

    del data
    return(detection_AP.counts,detection_AP.ecounts,detection_AP.Nsigma,detection_AP.Nap,detection_AP.mag,detection_AP.emag,spx,bpx,Sky,eSky,nSky,detection_AP.grow_corr)


def mvs_aperture_photometry(DF,filter,ee_df,zpt_dict,fitsname=None,mvs_ids_list_in=[],data_label='',dq_label='',label='data',bpx_list=[],spx_list=[],la_cr_remove=False,cr_radius=1,radius_in=10,radius1_in=10,radius2_in=15,sat_thr=np.inf,kill_plots=True,grow_curves=True,gstep=0.01,p=30,r_in=4,Python_origin=False,remove_candidate=False,flag='',multiply_by_exptime=False,multiply_by_gain=False,multiply_by_PAM=False,noBGsub=False,forceSky=False):
    getLogger(__name__).info(f'Starting mvs photometry on ids {mvs_ids_list_in}')
    label_dict={'data':1,'crclean_data':4}
    label_KLIP_dict={'data':'','crcleaxn_data':'crclean_'}
    if isinstance(fitsname, str): 
        hdul = fits.open(DF.path2data+'/'+fitsname+DF.fitsext+'.fits',memmap=False)
        data=None
        dqdata=None
    if len(mvs_ids_list_in)==0:mvs_ids_list=DF.mvs_targets_df.loc[DF.mvs_targets_df['fits_%s'%filter]==fitsname].mvs_ids.unique()
    else: mvs_ids_list=mvs_ids_list_in
    
    if len(mvs_ids_list_in)>1:phot=[]
    for mvs_ids in mvs_ids_list:
        if noBGsub:
            if forceSky: forcedSky=DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==mvs_ids,['sky_%s'%filter,'esky_%s'%filter,'nsky_%s'%filter]].values[0].tolist()
            else:forcedSky=[]
        else:
            forcedSky=[]

        if not isinstance(fitsname, str): 
            path2tile='%s/mvs_tiles/%s/tile_ID%i.fits'%(DF.path2data,filter,mvs_ids)
            hdul=None
            DATA=Tile(x=int((DF.tile_base-1)/2),y=int((DF.tile_base-1)/2),tile_base=DF.tile_base,inst=DF.inst)
            DATA.load_tile(path2tile,ext=label_dict[label],verbose=False,return_Datacube=False)
            DQ=Tile(x=int((DF.tile_base-1)/2),y=int((DF.tile_base-1)/2),tile_base=DF.tile_base,inst=DF.inst)
            DQ.load_tile(path2tile,ext='dq',verbose=False,return_Datacube=False)
            if remove_candidate:
                Kmode=DF.avg_candidates_df.loc[DF.avg_candidates_df.avg_ids.isin(DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.mvs_ids==mvs_ids].avg_ids)].mKmode.values[0]
                KLIP=Tile(x=int((DF.tile_base-1)/2),y=int((DF.tile_base-1)/2),tile_base=DF.tile_base,inst=DF.inst)
                KLIP.load_tile(path2tile,ext='%sKmode%i'%(label_KLIP_dict[label],Kmode),verbose=False,return_Datacube=False)
                filtered_data = sigma_clip(KLIP.data[KLIP.data<0])
                KLIP_temp=KLIP.data[KLIP.data<0][filtered_data.mask].copy()
                for elno in range(len(KLIP_temp)):
                    x1=1
                    x2=2
                    y1=1
                    y2=2
                    
                    w=np.where(KLIP_temp[elno]==KLIP.data)
                    if int(w[0])==0:y1=0
                    if int(w[0])==int(DF.tile_base)-1:y2=0
                    if int(w[1])==0:x1=0
                    if int(w[1])==int(DF.tile_base)-1:x2=0
                    KLIP_temp[elno]=np.nanmedian(KLIP.data[int(w[0])-y1:int(w[0])+y2,int(w[1])-x1:int(w[1])+x2])
                KLIP.data[KLIP.data<0][filtered_data.mask]=KLIP_temp
                KLIP.mk_tile(showplot=False,title='Klip',keep_size=True,cbar=True,simplenorm='sqrt',return_tile=False,kill_plots=kill_plots)
                data=DATA.data-KLIP.data
            else: data=DATA.data
            dqdata=DQ.data
        
        radius=radius_in
        radius1=radius1_in
        radius2=radius2_in
        exptime=DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==mvs_ids,'exptime_%s'%filter].values[0]
        ext=DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==mvs_ids,'ext'].values[0]
        if isinstance(zpt_dict,dict):
            zpt=zpt_dict[filter]
        else: 
            zpt=zpt_dict
        try:
            counts,ecounts,Nsigma,Nap,mag,emag,spx,bpx,skym,esky,nSky,grow_corr=aperture_photometry_handler(DF,mvs_ids,filter,bpx_list=bpx_list,spx_list=spx_list,data_label=data_label,dq_label=dq_label,la_cr_remove=la_cr_remove,cr_radius=cr_radius,data=data,dqdata=dqdata,hdul=hdul,zpt=zpt,radius_a=radius,radius1=radius1,radius2=radius2,sat_thr=sat_thr,kill_plots=kill_plots,grow_curves=grow_curves,ee_df=ee_df,gstep=gstep,p=p,r_in=r_in,r_min=radius1,r_max=radius2,Python_origin=Python_origin,ext=ext,exptime=exptime,multiply_by_exptime=multiply_by_exptime,multiply_by_gain=multiply_by_gain,multiply_by_PAM=multiply_by_PAM,noBGsub=noBGsub,forcedSky=forcedSky)
            if len(mvs_ids_list_in)>1: phot.append([int(mvs_ids),float(ext),float(counts),float(ecounts),float(Nap),float(mag),float(emag),float(spx),float(bpx),float(radius),float(radius1),float(radius2),float(skym),float(esky),float(nSky),float(grow_corr),str(flag)])
            else: phot=[int(mvs_ids),float(ext),float(counts),float(ecounts),float(Nap),float(mag),float(emag),float(spx),float(bpx),float(radius),float(radius1),float(radius2),float(skym),float(esky),float(nSky),float(grow_corr),str(flag)]
        except:
            getLogger(__name__).critical(f'Something went wrong during aperture photometry on id {mvs_ids}')
            raise ValueError
    if isinstance(fitsname, str):
        hdul.close()
        del hdul
    return(phot)
    
def avg_aperture_photometry(DF,avg_ids,filter,goodness_phot_label,suffix,skip_flags):
    getLogger(__name__).info(f'Performing average photometry on ID {avg_ids}')
    sel_ids=DF.mvs_targets_df.mvs_ids.isin(DF.crossmatch_ids_df.loc[(DF.crossmatch_ids_df.avg_ids==avg_ids)].mvs_ids)&~DF.mvs_targets_df['%s_flag'%filter].str.contains(skip_flags)
    ydata=DF.mvs_targets_df.loc[sel_ids,['m_%s'%filter]].values#.ravel()
    eydata=DF.mvs_targets_df.loc[sel_ids,['%s_%s%s'%(goodness_phot_label,filter,suffix)]].values#.ravel()

    if not np.all(np.isnan(ydata)):
        ma = np.ma.MaskedArray(ydata, mask=np.isnan(ydata)).ravel()
        ema = np.ma.MaskedArray(eydata, mask=np.isnan(eydata)).ravel()
        yav, we = np.ma.average(ma, weights=1 / ema ** 2, returned=True, axis=0)
    else:
        yav, we = [np.nan,np.nan]

    return([avg_ids,yav,np.sqrt(1/we),DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.avg_ids==avg_ids].mvs_ids),['spx_%s'%filter]].mean().values[0],DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.avg_ids==avg_ids].mvs_ids),['bpx_%s'%filter]].mean().values[0]])

def read_dq_from_tile(DF,path2tile=None,dq=None,bpx_list=[],spx_list=[]):
    if not isinstance(dq,(np.ndarray,list)):
        DQ=Tile(x=int((DF.tile_base-1)/2),y=int((DF.tile_base-1)/2),tile_base=DF.tile_base,inst=DF.inst)
        DQ.load_tile(path2tile,ext='dq',verbose=False,return_Datacube=False)
        dq=DQ.data
    bpx=np.sum([np.sum([x==i for x in dq.ravel()]) for i in bpx_list])
    spx=np.sum([np.sum([x==i for x in dq.ravel()]) for i in spx_list])
    return(bpx,spx)