"""
utilities functions that are use by the pipline
"""
import sys,os
sys.path.append('/')
# from pipeline_config import path2data

from ancillary import print_mean_median_and_std_sigmacut,PointsInCircum,rotate_point,round2closerint,parallelization_package
from utils_tile import allign_images,perform_PSF_subtraction
from tiles import Tile
from fake_star import Fake_Star
from photometry import Detection,flux_converter
from utils_dataframe import fk_writing
from utils_false_positives import FP_analysis,get_roc_curve
from utils_dataframe import create_empty_df
from utils_photometry import mvs_aperture_photometry,read_dq_from_tile,avg_aperture_photometry,aperture_photometry_handler,KLIP_aperture_photometry_handler,KLIP_throughput,photometry_AP
from straklip import config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import contextlib
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
import numpy.ma as ma
from astropy.io import fits
from pathlib import Path,PurePath
from itertools import repeat
from tqdm import tqdm
# from glob import glob
from IPython.display import display
from scipy.spatial import distance_matrix
from random import sample,randint,uniform,choice
from sklearn import metrics
from matplotlib.colors import PowerNorm,ListedColormap
from collections import Counter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from stralog import getLogger

def check4duplicants(DF,filter,mvs_ids_list,showduplicants=False):
    '''
    check for duplicated tiles in the tiles dataframe

    filter : str
        target firter for the update
    mvs_ids_list : list, optional
        list of ids from the multivisit dataframe to test.

    Returns
    -------
    None.

    '''
    # for filter in DF.filters_list:
    for cell in np.sort(DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(mvs_ids_list)]['%s_cell'%filter].unique()):
        skip_flags=['rejected','known_double']
        list_of_target4test=[]
        ids_list=DF.mvs_targets_df.loc[(DF.mvs_targets_df['%s_cell'%filter]==cell)&~DF.mvs_targets_df['%s_flag'%filter].isin([skip_flags])].mvs_ids.unique()

        for id in ids_list:
            if not DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==id,'%s_flag'%filter].str.contains('rejected').values[0]:
                path2tile = '%s/mvs_tiles/%s/tile_ID%i.fits' % (DF.path2out, filter, id)

                target=Tile(x=(DF.tilebase-1)/2,y=(DF.tilebase-1)/2,tile_base=DF.tilebase,delta=0,inst=DF.inst,Python_origin=True)
                data=target.load_tile(path2tile)
                list_of_target4test.append(data)


        var_list=np.zeros((len(list_of_target4test),len(list_of_target4test)))
        for i in range(len(list_of_target4test)):
            for j in range(i,len(list_of_target4test)):
                var_list[i][j]=int(np.all(list_of_target4test[i]==list_of_target4test[j]))

        ###### test for duplicates in the references
        d =  Counter(np.where(np.array(var_list)==1)[0])
        dup=np.array([k for k, v in d.items() if v > 1])
        if len(dup)>0:
            if showduplicants:
                print('Duplicants in %s %s'%(filter,cell))
                fig,ax=plt.subplots(figsize=(10,10))
                cmap = ListedColormap(['k', 'w'])
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                im=ax.imshow(var_list,origin='lower',interpolation='none',cmap=cmap)
                fig.colorbar(im,
                        cax=cax,
                        cmap=cmap,
                        orientation='vertical')#,
                plt.show()

            for el in dup:
                w=np.where(var_list[el]==1)[0]
                k=w[w==el][0]
                q=w[w!=el][0]
                min=np.min([np.min(list_of_target4test[k]),np.min(list_of_target4test[q])])
                max=np.max([np.max(list_of_target4test[k]),np.max(list_of_target4test[q])])
                if showduplicants:
                    fig,ax=plt.subplots(1,2,figsize=(10,5))
                    ax[0].imshow(list_of_target4test[k],origin='lower',norm=PowerNorm(0.2),vmin=min,vmax=max)
                    ax[1].imshow(list_of_target4test[q],origin='lower',norm=PowerNorm(0.2),vmin=min,vmax=max)
                    plt.show()
                    display(DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.mvs_ids.isin([np.array(ids_list)[k]])|DF.crossmatch_ids_df.mvs_ids.isin([np.array(ids_list)[q]])])
                    display(DF.avg_targets_df.loc[(DF.avg_targets_df.avg_ids.isin(DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.mvs_ids.isin([np.array(ids_list)[k]])].avg_ids.unique()))|(DF.avg_targets_df.avg_ids.isin(DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.mvs_ids.isin([np.array(ids_list)[q]])].avg_ids.unique()))])
                    display(DF.mvs_targets_df.loc[(DF.mvs_targets_df.mvs_ids.isin([np.array(ids_list)[k]]))|(DF.mvs_tiles_df.mvs_ids.isin([np.array(ids_list)[q]]))])
                ids_skip=(DF.mvs_targets_df.loc[(DF.mvs_targets_df.mvs_ids.isin([np.array(ids_list)[k]]))|(DF.mvs_tiles_df.mvs_ids.isin([np.array(ids_list)[q]]))].mvs_ids.unique())
                # os.remove('%s/%s/%s/%s/mvs_tiles/%s/tile_ID%i.fits'%(path2data,DF.project,DF.target,DF.inst,filter,id))
                DF.crossmatch_ids_df=DF.crossmatch_ids_df.loc[~DF.crossmatch_ids_df.mvs_ids.isin(ids_skip)]
    DF.avg_targets_df=DF.avg_targets_df.loc[DF.avg_targets_df.avg_ids.isin(DF.crossmatch_ids_df.avg_ids.unique())]
    DF.mvs_targets_df=DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(DF.crossmatch_ids_df.mvs_ids.unique())]
    print('All duplicants killed')

def task_perform_KLIP_PSF_subtraction_on_tiles(DF,filter,cell,mvs_ids_list,label_dict,hdul_dict,KLIP_label_dict,skip_flags,label,kmodes_list,overwrite):
    if len(mvs_ids_list)==0:
        ids_list=DF.mvs_targets_df.loc[(DF.mvs_targets_df['cell_%s'%filter]==cell)&~DF.mvs_targets_df['flag_%s'%filter].isin([skip_flags])].mvs_ids.unique()
    else:
        ids_list=mvs_ids_list

    psf_ids_list=DF.mvs_targets_df.loc[DF.mvs_targets_df['flag_%s'%filter].str.contains('psf')&(DF.mvs_targets_df['cell_%s'%filter]==cell)].mvs_ids.unique()
    for id in ids_list:
        if not DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==id,'flag_%s'%filter].str.contains('rejected').values[0]:
            path2tile='%s/mvs_tiles/%s/tile_ID%i.fits'%(DF.path2out,filter,id)
            getLogger(__name__).info(f'Working on tile: {path2tile}.')

            if not overwrite:
                try:
                    with fits.open(path2tile) as hdul:
                        for Kmode in kmodes_list : hdul['KMODE%i'%Kmode]
                    go4PSFsub=False
                except:
                    go4PSFsub=True
            else: go4PSFsub=True
            if go4PSFsub:
                DATA=Tile(x=(DF.tilebase-1)/2,y=(DF.tilebase-1)/2,tile_base=DF.tilebase,delta=0,inst=DF.inst,Python_origin=True)
                Datacube=DATA.load_tile(path2tile,ext=label_dict[label],verbose=False,return_Datacube=True,hdul_max=hdul_dict[label],mode='update',raise_errors=True)
                DQ=Tile(x=int((DF.tilebase-1)/2),y=int((DF.tilebase-1)/2),tile_base=DF.tilebase,inst=DF.inst)
                DQ.load_tile(path2tile,ext='dq',verbose=False,return_Datacube=False)
                DQ_list=list(set(DQ.data.ravel()))

                x = DATA.data.copy()
                mask_x=DQ.data.copy()
                for i in [i for i in DQ_list if i not in DF.dq2mask]:
                    mask_x[(mask_x==i)]=0
                mx = ma.masked_array(x, mask=mask_x)
                mx.data[mx.mask]=-9999
                mx.data[mx.data<0]=0
                targ_tiles=pd.DataFrame(data=[[np.array(mx.data)]],columns=['data'])
                elno=0
                if not targ_tiles.data.isna().all():
                    ref_tiles=pd.DataFrame(data=[[[0]]],columns=['data'])
                    avg_ids=DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.mvs_ids==id].avg_ids.unique()[0]
                    for refid in DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(psf_ids_list)&~(DF.mvs_targets_df.mvs_ids.isin(DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.avg_ids==avg_ids].mvs_ids))].mvs_ids.unique():
                        path2ref = '%s/mvs_tiles/%s/tile_ID%i.fits' % (DF.path2out, filter, id)
                        REF=Tile(x=(DF.tilebase-1)/2,y=(DF.tilebase-1)/2,tile_base=DF.tilebase,delta=0,inst=DF.inst,Python_origin=True)
                        REF.load_tile(path2ref,ext=label_dict[label],verbose=False,hdul_max=hdul_dict[label])
                        DQREF=Tile(x=int((DF.tilebase-1)/2),y=int((DF.tilebase-1)/2),tile_base=DF.tilebase,inst=DF.inst)
                        DQREF.load_tile(path2ref,ext='dq',verbose=False,return_Datacube=False)

                        xref=REF.data.copy()
                        mask_ref=DQREF.data.copy()
                        for i in [i for i in DQ_list if i not in DF.dq2mask]:  mask_ref[(mask_ref==i)]=0
                        # mask_ref[(mask_ref!=4096)&(mask_ref!=8192)]=0
                        mref = ma.masked_array(xref, mask=mask_ref)
                        mref.data[mref.mask]=-9999
                        ref_tiles.loc[elno]=[np.array(mref.data)]
                        if len(mref[mref<=-9999])>10:pass
                        else:
                            mref.data[mref.data<0]=0
                            ref_tiles.loc[elno]=[np.array(mref.data)]
                            elno+=1

                    residuals,psf_models=perform_PSF_subtraction(targ_tiles['data'],ref_tiles['data'],kmodes_list=kmodes_list)
                    return_Datacube=True
                    for Kmode in residuals.columns: # loop over the residuals of the different kmodes_list
                        KLIP=Tile(data=residuals[Kmode].tolist()[0],x=(DF.tilebase-1)/2,y=(DF.tilebase-1)/2,tile_base=DF.tilebase,delta=0,inst=DF.inst,Python_origin=True)
                        KLIP.mk_tile(pad_data=False,legend=False,showplot=False,verbose=False,title='Kmode%i'%Kmode,kill_plots=True)
                        Datacube=KLIP.append_tile(path2tile,Datacube=Datacube,verbose=False,name='%s%i'%(KLIP_label_dict[label],Kmode),return_Datacube=return_Datacube,write=False)
                    for model in psf_models.columns:
                        if model == psf_models.columns[-1]:return_Datacube=False
                        PSF=Tile(data=psf_models[model].tolist()[0],x=(DF.tilebase-1)/2,y=(DF.tilebase-1)/2,tile_base=DF.tilebase,delta=0,inst=DF.inst,Python_origin=True)
                        PSF.mk_tile(pad_data=False,legend=False,showplot=False,verbose=False,title='Model%s'%model,kill_plots=True)
                        Datacube=PSF.append_tile(path2tile,Datacube=Datacube,verbose=False,name='Model%s'%model,return_Datacube=return_Datacube,write=False)


def perform_KLIP_PSF_subtraction_on_tiles(DF,filter,label,workers=None,parallel_runs=True,mvs_ids_list=[],kmodes_list=[],skip_flags=['rejected','known_double'],overwrite=False,chunksize = None):
    '''
    Perform PSF subtraction on targets tiles


    Parameters
    ----------
    filter : TYPE
        DESCRIPTION.
    label : TYPE
        DESCRIPTION.
    workers : TYPE, optional
        DESCRIPTION. The default is None.
    parallel_runs : TYPE, optional
        DESCRIPTION. The default is True.
    mvs_ids_list : TYPE, optional
        list of ids from the multivisit dataframe to test. The default is [].
    kmodes_list : TYPE, optional
        list of KLIP modes to use for the PSF subtraction. The default is [].
    skip_flags : TYPE, optional
        list of flagged targets to skip during PSF subtraction. The default is ['rejected','known_double']. The default is ['rejected','known_double'].
    num_of_chunks : TYPE, optional
        DESCRIPTION. The default is None.
    overwrite : bool, optional
        if True, overwrite existing residuals.
        if False overwrite them ONLY if one or more Kmodes are missing. Defaulte False.
    chunksize : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    '''
    label_dict={'data':1,'crclean_data':4}
    hdul_dict={'data':3,'crclean_data':4}
    KLIP_label_dict={'data':'Kmode','crclean_data':'crclean_Kmode'}
    if label not in ['data','crclean_data']:
        raise ValueError('Chose either data or crclean_data label for subtraction')

    if len(mvs_ids_list)==0: cell_list=DF.mvs_targets_df['cell_%s'%filter].unique()
    else: cell_list=np.sort(DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(mvs_ids_list), 'cell_%s'%filter].unique())
    cell_list=cell_list[~np.isnan(cell_list)]

    if parallel_runs:
        workers,chunksize,ntarget=parallelization_package(workers,len(cell_list),chunksize = chunksize)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for _ in executor.map(task_perform_KLIP_PSF_subtraction_on_tiles,repeat(DF),repeat(filter),cell_list,repeat(mvs_ids_list),repeat(label_dict),repeat(hdul_dict),repeat(KLIP_label_dict),repeat(skip_flags),repeat(label),repeat(kmodes_list),repeat(overwrite),chunksize=chunksize):
                pass

    else:
        for cell in cell_list:
            task_perform_KLIP_PSF_subtraction_on_tiles(DF,filter,cell,mvs_ids_list,label_dict,hdul_dict,KLIP_label_dict,skip_flags,label,kmodes_list,overwrite)

def perform_KLIP_PSF_subtraction_on_fakes(DF,filter,target,psf_list,pos,kmodes_list,npsfs,showplot,exptime,aptype,delta,noBGsub):
    '''
    Taks perfomed in the perform_kKLIP_PSF_subtraction_on_fakes_tiles

    Parameters
    ----------
    filter : str
        filter name.
    magbin : int
        magnitude bins to analyze.
    dmag_list : list
        delta magnitude list to analyze.
    sep_list : list
        separation list to analyze.
    fk_ids_list : list
        fake ids list to analyze.
    psf_list : list
        fake psf ids list to analyze.
    kmodes_list : list
        Kmode list to analyze.
    npsfs : int
        number of psf stars to use in the PSF subtraction.
    aptype : (circular,square,4pixels), optional
        defin the aperture type to use during aperture photometry.
        The default is '4pixels'.
    delta : int, optional
        step to create the square mask in range -delta, x, +delt and -delta, y, +delta. The default is 1.

    Returns
    -------
    None.

    '''
    y,x=round2closerint(pos)
    references_list=sample(psf_list,npsfs)
    targ_tiles=pd.Series([target],name='%s_data'%filter)
    ref_tiles=pd.Series(references_list,name='%s_data'%filter)
    residuals,_=perform_PSF_subtraction(targ_tiles,ref_tiles,kmodes_list=kmodes_list,no_PSF_models=True)
    zpt=DF.header_df.loc['Delta%s'%filter,'Values']
    kill_plots=True
    out_list=[]
    for Kmode in residuals.columns:
        if not np.all(np.isnan(residuals[Kmode].values[0])):
            if showplot:
                print('>. KLIPmode %i'%Kmode)
                kill_plots=False
            counts,ecounts,Nsigma,Nap,mag,emag,spx,bpx,Sky,eSky,nSky,grow_corr=KLIP_aperture_photometry_handler(DF,0,filter,x=x,y=y,data=residuals[Kmode].values[0],zpt=zpt,ezpt=0,aptype=aptype,noBGsub=noBGsub,sigma=3,kill_plots=kill_plots,Python_origin=True,delta=delta,exptime=exptime,gain=1)
            out_list.extend([Nsigma,counts,ecounts,mag])
        else:
            out_list.extend([np.nan,np.nan,np.nan,np.nan])
    return(out_list)

def task_mvs_targets_infos(DF,avg_id,skip_filters,aptype,verbose,noBGsub,sigma,kill_plots,label,delta,sat_thr):
    '''
    parallelized task for the update_mvs_targets.

    Parameters
    ----------
    avg_id : int, optional
        id from the average dataframe to test.
    skip_filters : list, optional
        list of filters to skip. The default is ''.
    aptype : (circular,square,4pixels), optional
       defin the aperture type to use during aperture photometry.
      verbose : bool, optional
        choose to show print and plots. The default is False.
    noBGsub : bool
        choose to skip sky suntraction from tile.
    sigma : float
        value of the sigma clip.

    Returns
    -------
    None.

    '''
    label_dict={'data':1,'crclean_data':4}

    if verbose: print('Verbose mode: ',verbose)
    if verbose:
        display(DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.avg_ids==avg_id])
        display(DF.avg_targets_df.loc[DF.avg_targets_df.avg_ids==avg_id])
        display(DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.avg_ids==avg_id].mvs_ids)])

    candidate_df=create_empty_df(['filter','mvs_ids'],['counts','ecounts','nsky','Nsigma','mag','emag','flag','ROTA','PA_V3','std','sep'],multy_index=True,levels=[DF.filters_list,DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.avg_ids==avg_id].mvs_ids])
    for filter in DF.filters_list:
        if filter not in skip_filters:
            mvs_ids=DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.avg_ids==avg_id].mvs_ids.unique()
            for id in mvs_ids:
                if not DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==id,'%s_flag'%filter].str.contains('rejected').values[0]:
                    x,y=[int((DF.tilebase-1)/2),int((DF.tilebase-1)/2)]
                    path2tile='%s/%s/%s/%s/mvs_tiles/%s/tile_ID%i.fits'%(path2data,DF.project,DF.target,DF.inst,filter,id)
                    DATA=Tile(x=int((DF.tilebase-1)/2),y=int((DF.tilebase-1)/2),tile_base=DF.tilebase,inst=DF.inst)
                    DATA.load_tile(path2tile,ext=label_dict[label],verbose=False,return_Datacube=False)
                    DQ=Tile(x=int((DF.tilebase-1)/2),y=int((DF.tilebase-1)/2),tile_base=DF.tilebase,inst=DF.inst)
                    DQ.load_tile(path2tile,ext=3,verbose=False,return_Datacube=False)

                    forcedSky=DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==id,['sky%s'%filter,'esky%s'%filter,'nsky%s'%filter]].values[0]
                    exptime=DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==id,'exptime_%s'%filter].values[0]
                    counts,ecounts,Nsigma,Nap,mag,emag,spx,bpx,Sky,eSky,nSky,grow_corr=KLIP_aperture_photometry_handler(DF,id,filter,x=x,y=y,data=DATA.data,dqdata=DQ.data,aptype=aptype,noBGsub=True,forcedSky=forcedSky,sigma=sigma,kill_plots=True,Python_origin=True,delta=delta,sat_thr=sat_thr,exptime=exptime)#,multiply_by_exptime=multiply_by_exptime,multiply_by_gain=multiply_by_gain,multiply_by_PAM=multiply_by_PAM)
                    candidate_df.loc[(filter,id),['counts','ecounts','nsky','Nsigma','mag','emag']]=[counts,ecounts,Nap,Nsigma,mag,emag]
    if verbose:
        plt.show()
    return(candidate_df)

def update_mvs_targets(DF,pipe_cfg,filter):
    '''
    Update the multi-visits targets dataframe with PA_V3 and exposure time infos.

    Parameters
    ----------
    Returns
    -------
    None.

    '''
    # for filter in DF.filters_list:
    for filename,row in DF.mvs_targets_df.groupby('fits_%s'%(filter)):
        if isinstance(filename, str):
            hdul = fits.open(pipe_cfg.paths['data']+'/'+filename+DF.fitsext+'.fits')
            PA_V3=hdul[0].header['PA_V3']
            EXPTIME=hdul[0].header['EXPTIME']
            rot_angle=hdul[1].header['ORIENTAT']

            DF.mvs_targets_df.loc[(DF.mvs_targets_df['fits_%s'%(filter)]==filename),['%s_PA_V3'%filter,'%s_ROTA'%filter,'exptime_%s'%filter]]=[round(float(PA_V3),3),round(rot_angle,3),round(EXPTIME,3)]
            hdul.close()

def task_mvs_candidates_infos(DF,avg_id,d,skip_filters,aptype,verbose,noBGsub,sigma,DF_fk,label,kill_plots,delta,radius,sat_thr,mkd,mad):
    '''
    parallelized task for the update_mvs_candidates.

    Parameters
    ----------
    avg_id : int, optional
        id from the average dataframe to test.
    d : float, optional
        maximum distances between candidate's detections to accept it. The default is 1.5.
    skip_filters : list, optional
        list of filters to skip. The default is ''.
    aptype : (circular,square,4pixels), optional
       defin the aperture type to use during aperture photometry.
    verbose : bool, optional
        choose to show print and plots. The default is False.
    noBGsub : bool
        choose to skip sky suntraction from tile.
    sigma : float
        value of the sigma clip.
    DF_fk : dataframe class
        dataframe class containing the fake dataframe.
    minimum_Kmode_detections: int, optional
        minimum filter detections per KLIPmode to accept a candidate
    mad: int, optional
        minimum arecsecond distance from center to accept a candidate

    Returns
    -------
    None.

    '''
    KLIP_label_dict={'data':'Kmode','crclean_data':'crclean_Kmode'}
    minimum_px_distance_from_center=mad/DF.pixelscale
    origin=(int((DF.tilebase-1)/2),int((DF.tilebase-1)/2)) #these are the postons of the peak in each tile
    if not kill_plots:
        if verbose: kill_plots=False
        else: kill_plots=True

    if verbose:
        print('Verbose mode: ',verbose)
        display(DF.mvs_targets_df.loc[DF.crossmatch_ids_df.avg_ids==avg_id])
    filters_list=[filter for filter in DF.filters_list if filter not in ['F658N']]
    temporary_candidate_df=create_empty_df(['Kmode','filter','mvs_ids'],['x_tile','y_tile','x_rot','y_rot','pdc','ROTA','PA_V3','Nsigma','flag','counts','ecounts','mag','emag'],multy_index=True,levels=[DF.kmodes_list,filters_list,DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.avg_ids==avg_id].mvs_ids])
    sub_ids=DF.crossmatch_ids_df.loc[(DF.crossmatch_ids_df.avg_ids==avg_id)].mvs_ids.unique()
    for Kmode in DF.kmodes_list:
        for filter in DF.filters_list:
            if filter not in skip_filters:
                zpt=DF.header_df.loc['Delta%s'%filter,'Values']
                ezpt=DF.header_df.loc['eDelta%s'%filter,'Values']
                elno=0
                if verbose: fig,ax=plt.subplots(1,len(sub_ids),figsize=(5*len(sub_ids),5))
                else: fig,ax=[None,None]


                for mvs_ids in sub_ids:
                    if not DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==mvs_ids,'%s_flag'%filter].str.contains('rejected').values[0]:
                        PA_V3=DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==mvs_ids,'%s_PA_V3'%filter].values[0]
                        ROTA=DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==mvs_ids,'%s_ROTA'%filter].values[0]
                        path2tile='%s/%s/%s/%s/mvs_tiles/%s/tile_ID%i.fits'%(path2data,DF.project,DF.target,DF.inst,filter,mvs_ids)
                        KDATA=Tile(x=int((DF.tilebase-1)/2),y=int((DF.tilebase-1)/2),tile_base=DF.tilebase,inst=DF.inst)
                        KDATA.load_tile(path2tile,ext='%s%s'%(KLIP_label_dict[label],Kmode),verbose=False,return_Datacube=False,raise_errors=False)
                        if np.all(ax)==None:
                            axis=None
                        else:
                            if len(sub_ids)==1: axis=ax
                            else:axis=ax[elno]
                        if not np.all(np.isnan(KDATA.data)):
                            KDATA.mk_tile(fig=fig,ax=axis,pad_data=False,verbose=False,xy_m=True,legend=False,showplot=False,keep_size=True,xy_dmax=None,title='%s ID %i ROTA %s'%(filter,mvs_ids,ROTA),kill_plots=True)
                            pdc=np.round(np.sqrt(abs(KDATA.x_m - (DF.tilebase-1)/2)**2+abs(KDATA.y_m - (DF.tilebase-1)/2)**2),3)
                            elno+=1
                            if KDATA.x_m < DF.tilebase and KDATA.y_m < DF.tilebase and KDATA.x_m >= 0 and  KDATA.y_m >= 0 and (pdc >=minimum_px_distance_from_center):
                                point=[KDATA.x_m,KDATA.y_m]
                                angle=float(ROTA)
                                Xrot,Yrot=rotate_point(point=point, angle=angle, origin=origin,r=3) # these are the rotated coordinate of the peack on the common referance frame
                                x,y=[KDATA.x_m, KDATA.y_m]
                                dist=np.round(np.sqrt((int((DF.tilebase-1)/2)-x)**2+(int((DF.tilebase-1)/2)-y)**2),3)
                                exptime=DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==mvs_ids,'exptime_%s'%filter].values[0]
                                if not np.all(np.isnan(KDATA.data)) and Xrot>=0 and Xrot<=DF.tilebase and Yrot>=0 and Yrot<=DF.tilebase:
                                    if not kill_plots:print('> Kmode %s, %s, mvs_id %s'%(Kmode, filter,mvs_ids))
                                    counts,ecounts,Nsigma,Nap,mag,emag,spx,bpx,Sky,eSky,nSky,grow_corr=KLIP_aperture_photometry_handler(DF,mvs_ids,filter,x=x,y=y,data=KDATA.data,zpt=zpt,ezpt=ezpt,aptype=aptype,noBGsub=False,sigma=sigma,kill_plots=kill_plots,Python_origin=True,delta=delta,radius_a=radius,sat_thr=sat_thr,exptime=exptime)#,multiply_by_exptime=multiply_by_exptime,multiply_by_gain=multiply_by_gain,multiply_by_PAM=multiply_by_PAM)
                                    if Nsigma>0 :#and dist>= minimum_dist:
                                        temporary_candidate_df.loc[(Kmode,filter,mvs_ids),['counts','ecounts','Nsigma','mag','emag']]=[counts,ecounts,Nsigma,mag,emag]
                                        temporary_candidate_df.loc[(Kmode,filter,mvs_ids),['sep']]=dist
                                        temporary_candidate_df.loc[(Kmode,filter,mvs_ids),['x_tile','y_tile','x_rot','y_rot','pdc','ROTA','PA_V3','flag']]=[KDATA.x_m,KDATA.y_m,Xrot,Yrot,pdc,ROTA,PA_V3,'good_candidate']
                                    else:
                                        temporary_candidate_df.loc[(Kmode,filter,mvs_ids),['x_tile','y_tile','x_rot','y_rot','pdc','ROTA','PA_V3','Nsigma','flag']]=[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,'rejected']
                                        plt.close()
                                else:
                                    temporary_candidate_df.loc[(Kmode,filter,mvs_ids),['x_tile','y_tile','x_rot','y_rot','pdc','ROTA','PA_V3','Nsigma','flag']]=[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,'rejected']
                                    plt.close()
                            else:
                                temporary_candidate_df.loc[(Kmode,filter,mvs_ids),['x_tile','y_tile','x_rot','y_rot','pdc','ROTA','PA_V3','Nsigma','flag']]=[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,'rejected']
                                plt.close()
                        else:
                            temporary_candidate_df.loc[(Kmode,filter,mvs_ids),['x_tile','y_tile','x_rot','y_rot','pdc','ROTA','PA_V3','Nsigma','flag']]=[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,'rejected']
                            plt.close()
                    else:
                        temporary_candidate_df.loc[(Kmode,filter,mvs_ids),['x_tile','y_tile','x_rot','y_rot','ROTA','pdc','PA_V3','Nsigma','flag']]=[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,'rejected']
                        plt.close()
    Nsigma_score=[]
    Kmode_idx_list=[]
    # filter_idx_list=[]
    for Kmode in DF.kmodes_list:
        if (temporary_candidate_df.loc[(Kmode),'flag']!='rejected').astype(int).sum(axis=0)>=2:
            pos=temporary_candidate_df.loc[temporary_candidate_df.flag!='rejected'].loc[(Kmode),['y_rot','x_rot']].values
            pair_dist = np.round(distance_matrix(pos, pos).astype(float),3)
            num=pos.shape[0]
            pair_dist[np.r_[:num], np.r_[:num]] = np.nan

            n_detections=np.array([temporary_candidate_df.loc[(Kmode,filter),'flag'].str.contains('candidate').astype(int).sum() for filter in DF.filters_list if filter not in skip_filters])
            expected_n_detections=np.array([DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.avg_ids==avg_id].mvs_ids)&(DF.mvs_targets_df['%s_flag'%filter]!='rejected')].mvs_ids.nunique() for filter in DF.filters_list if filter not in skip_filters])
            temp_nn=[]
            for elno in range(len(expected_n_detections)):
                temp_nn.extend([expected_n_detections[elno]]*expected_n_detections[elno])
            temp_filters_list=temporary_candidate_df.loc[temporary_candidate_df.flag!='rejected'].loc[Kmode].index.get_level_values('filter').values
            if sum(n_detections) == sum(expected_n_detections):
                distance_matrix_df=pd.DataFrame(data=pair_dist,columns=temp_filters_list,index=temp_filters_list)
                sel_flag=(temporary_candidate_df.loc[(Kmode)].flag!='rejected')
                distance_matrix_df['mvs_ids']=temporary_candidate_df.loc[(Kmode)].loc[sel_flag].index.get_level_values('mvs_ids').values
                columns=distance_matrix_df.columns[:-1].unique()
                minimum_filters_detections=np.max([int((len(distance_matrix_df[columns].columns))/2)+1,2])
                # minimum_visits_detections=np.max([int(distance_matrix_df.mvs_ids.nunique())/2+1,2])
                for filter in distance_matrix_df.index.get_level_values(0).unique():
                    try:
                        selected_distance_matrix_df=(((distance_matrix_df.loc[filter,columns].to_frame().T<=1.5)&(distance_matrix_df.loc[filter,columns].to_frame().T>=0))|(distance_matrix_df.loc[filter,columns].to_frame().T.isna()))
                        filter_id_sel_list=(selected_distance_matrix_df.astype(int).sum(axis=1).values>=minimum_filters_detections)
                        sel_mvs_ids=temporary_candidate_df.index.get_level_values('mvs_ids').isin(distance_matrix_df.loc[filter].to_frame().T.mvs_ids.values[filter_id_sel_list])
                        if verbose:
                            print('> Printing Kmode %i distance_matrix_df'%Kmode)
                            display(distance_matrix_df.loc[filter,columns].to_frame().T)
                    except:
                        selected_distance_matrix_df=(((distance_matrix_df.loc[filter,columns]<=1.5)&(distance_matrix_df.loc[filter,columns]>=0))|(distance_matrix_df.loc[filter,columns].isna()))
                        filter_id_sel_list=(selected_distance_matrix_df.astype(int).sum(axis=1).values>=minimum_filters_detections)
                        sel_mvs_ids=temporary_candidate_df.index.get_level_values('mvs_ids').isin(distance_matrix_df.loc[filter].mvs_ids.values[filter_id_sel_list])
                        if verbose:
                            print('> Printing Kmode %i distance_matrix_df'%Kmode)
                            display(distance_matrix_df.loc[filter,columns])

                    temporary_candidate_df.loc[(Kmode,filter,~sel_mvs_ids)]=np.nan
                    temporary_candidate_df.loc[(Kmode,filter,~sel_mvs_ids),'flag']='rejected'

                if np.any(temporary_candidate_df.loc[(Kmode),'flag'].values!='rejected'):
                    Nsigma_score.append(np.nanmean(temporary_candidate_df.loc[(Kmode),'Nsigma'].values))
                    Kmode_idx_list.append(True)
                    if verbose:
                        print('> Selected candidate df:')
                        display(temporary_candidate_df.loc[Kmode])
                else:
                    Nsigma_score.append(0)
                    Kmode_idx_list.append(False)

            else:
                temporary_candidate_df.loc[Kmode]=np.nan
                temporary_candidate_df.loc[Kmode,'flag']='rejected'
                Nsigma_score.append(0)
                Kmode_idx_list.append(False)
        else:
            temporary_candidate_df.loc[Kmode]=np.nan
            temporary_candidate_df.loc[Kmode,'flag']='rejected'
            Nsigma_score.append(0)
            Kmode_idx_list.append(False)
            # filter_idx_list.append([])

    Nsigma_score=np.array(Nsigma_score)
    Nsigma_sel_score=np.array(Nsigma_score)[Kmode_idx_list]
    minimum_Kmode_detections=mkd
    if len(Nsigma_sel_score[Nsigma_sel_score>0])>=minimum_Kmode_detections:
        q=np.where(Nsigma_score==np.nanmax(Nsigma_sel_score))[0][-1]
        Kmode_final=DF.kmodes_list[q]
        candidate_df=temporary_candidate_df.loc[(Kmode_final)]
        candidate_df.loc[candidate_df.flag=='rejected',['x_tile','y_tile','x_rot','y_rot','ROTA','PA_V3','Nsigma','counts','ecounts','mag','emag']]=np.nan

        if verbose:
            print('EUREKA!!!!! We have a Candidate')
            print('Kmode selected # %s'%Kmode_final)
            display(candidate_df)
    else:
        Kmode_final=None
        candidate_df=None
        if verbose:
            print('BOOMER!!!!! No Candidate found')
    # sys.exit()
    return(Kmode_final,candidate_df)

def update_header_photometry(DF,suffix='',skip_filters='',aptype='4pixels',verbose=False,workers=None,noBGsub=True,sigma=2.5,min_mag_list=[],max_mag_list=[],DF_fk=None,parallel_runs=True,chunksize = None,kill_plots=True,label='data',delta=3,sat_thr=np.inf):
    '''
    update the photometry entry in the header

    Parameters
    ----------
    suffix: str, optional
        suffix to append to mag label. For example, if original photometry is present in the catalog, it canbe use with suffix='_o'.
        Default is ''.
    skip_filters : list, optional
        list of filters to skip. The default is ''.
    aptype : (circular,square,4pixels), optional
       defin the aperture type to use during aperture photometry.
       The default is '4pixels'.
    verbose : bool, optional
        choose to show print and plots. The default is False.
    workers : int, optional
        number of workers to split the work accross multiple CPUs. The default is 3.
    noBGsub : bool
        choose to skip sky subtraction from tile.
    sigma : float
        value of the sigma clip.
    min_mag_list: list
        list of magnitudes (one for filter) to us as upper cut for suitable stars selection to evaluate the delta for 4p aperture photometry
    mac_mag_list: list
        list of magnitudes (one for filter) to us as lower cut for suitable stars selection to evaluate the delta for 4p aperture photometry
    DF_fk : dataframe class
        dataframe class containing the fake dataframe. If None look into DF. The default is None.
    parallel_runs: bool, optional
        Choose to to split the workload over different CPUs. The default is True
    num_of_chunks : int, optional
        number of chunk to split the targets. The default is None.
    chunksize : int, optional
        size of each chunk for the parallelization process. The default is None.

    Returns
    -------
    None.

    '''
    if __name__ == 'utils_straklip':
        if DF_fk==None: DF_fk=DF
        avg_ids=DF.avg_targets_df.avg_ids.unique()
        ############################################################ ZPT ##############################################################
        print('Working on the zeropoints')
        if parallel_runs:
            workers,chunksize,ntarget=parallelization_package(workers,len(avg_ids),chunksize = chunksize)
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for candidate_df in tqdm(executor.map(task_mvs_targets_infos,repeat(DF),avg_ids,repeat(skip_filters),repeat(aptype),repeat(verbose),repeat(noBGsub),repeat(sigma),repeat(kill_plots),repeat(label),repeat(delta),repeat(sat_thr),chunksize=chunksize)):
                    for filter in candidate_df.index.get_level_values('filter'):
                        DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(candidate_df.index.get_level_values('mvs_ids').unique()),'counts%s_ap'%filter]=candidate_df.loc[(filter),'counts'].values
                        DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(candidate_df.index.get_level_values('mvs_ids').unique()),'ecounts%s_ap'%filter]=candidate_df.loc[(filter),'ecounts'].values
                        DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(candidate_df.index.get_level_values('mvs_ids').unique()),'nsky%s_ap'%filter]=candidate_df.loc[(filter),'nsky'].values
                        DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(candidate_df.index.get_level_values('mvs_ids').unique()),'m%s_ap'%filter]=candidate_df.loc[(filter),'mag'].values
                        DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(candidate_df.index.get_level_values('mvs_ids').unique()),'e%s_ap'%filter]=candidate_df.loc[(filter),'emag'].values
        else:
            for id in tqdm(avg_ids):
                candidate_df =task_mvs_targets_infos(DF,id,skip_filters,aptype,verbose,noBGsub,sigma,kill_plots,label,delta,sat_thr)
                for filter in candidate_df.index.get_level_values('filter'):
                    DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(candidate_df.index.get_level_values('mvs_ids').unique()),'counts%s_ap'%filter]=candidate_df.loc[(filter),'counts'].values
                    DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(candidate_df.index.get_level_values('mvs_ids').unique()),'ecounts%s_ap'%filter]=candidate_df.loc[(filter),'ecounts'].values
                    DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(candidate_df.index.get_level_values('mvs_ids').unique()),'nsky%s_ap'%filter]=candidate_df.loc[(filter),'nsky'].values
                    DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(candidate_df.index.get_level_values('mvs_ids').unique()),'m%s_ap'%filter]=candidate_df.loc[(filter),'mag'].values
                    DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(candidate_df.index.get_level_values('mvs_ids').unique()),'e%s_ap'%filter]=candidate_df.loc[(filter),'emag'].values


        elno = 0
        for filter in DF.filters_list:
            # sat_sel = (DF.mvs_targets_df['spx%s'%filter] <= 0)
            type_sel = (DF.mvs_targets_df['%s_flag'%filter].str.contains('psf'))
            # emag_sel = (DF.mvs_targets_df['e%s'%filter] < 0.01)
            if len(min_mag_list) == 0:
                min_mag = np.nanmin(DF.mvs_targets_df['m%s%s'%(filter,suffix)].values)
            else:
                min_mag = min_mag_list[elno]
            if len(max_mag_list) == 0:
                max_mag = np.nanmax(DF.mvs_targets_df['m%s%s'%(filter,suffix)].values)
            else:
                max_mag = max_mag_list[elno]
            mag_sel = (DF.mvs_targets_df.loc[type_sel,'m%s%s'%(filter,suffix)] >= min_mag) & (DF.mvs_targets_df.loc[type_sel,'m%s%s'%(filter,suffix)] < max_mag)
            dmags=DF.mvs_targets_df.loc[mag_sel&type_sel,'m%s%s'%(filter,suffix)].values-DF.mvs_targets_df.loc[mag_sel&type_sel,'m%s_ap'%filter].values
            dmags=dmags.astype(float)
            dmag_mean_sc,dmag_median_sc,dmag_std_sc,Mask=print_mean_median_and_std_sigmacut(dmags,pre='%s '%filter,verbose=False,sigma=sigma,nonan=True)
            DF.header_df.loc['Delta%s'%filter,'Values']=dmag_median_sc
            DF.header_df.loc['eDelta%s'%filter,'Values']=dmag_std_sc
            DF.mvs_targets_df['m%s_ap'%filter]+=dmag_median_sc
            DF.mvs_targets_df['e%s_ap'%filter]=np.sqrt(DF.mvs_targets_df['e%s_ap'%filter].values.astype(float)**2+dmag_std_sc**2)
            elno+=1
        display(DF.header_df)

def update_candidates_with_detection(DF,candidate_df,Kmode_final,verbose):
    for filter in candidate_df.index.get_level_values('filter').unique():
        for mvs_ids in candidate_df.loc[(filter)].index.get_level_values('mvs_ids').unique():
            if candidate_df.loc[candidate_df.index.get_level_values('mvs_ids')==mvs_ids].loc[(filter),'flag'].values[0]!='rejected':
                # display(candidate_df.loc[candidate_df.index.get_level_values('mvs_ids')==mvs_ids])
                try:
                    # display(candidate_df.loc[candidate_df.index.get_level_values('mvs_ids')==mvs_ids].loc[(filter)])
                    DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,'x%s_tile'%filter]=candidate_df.loc[candidate_df.index.get_level_values('mvs_ids')==mvs_ids].loc[(filter),'x_tile'].values[0]
                    DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,'y%s_tile'%filter]=candidate_df.loc[candidate_df.index.get_level_values('mvs_ids')==mvs_ids].loc[(filter),'y_tile'].values[0]
                    DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,'x%s_rot'%filter]=candidate_df.loc[candidate_df.index.get_level_values('mvs_ids')==mvs_ids].loc[(filter),'x_rot'].values[0]
                    DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,'y%s_rot'%filter]=candidate_df.loc[candidate_df.index.get_level_values('mvs_ids')==mvs_ids].loc[(filter),'y_rot'].values[0]

                    DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,'counts%s'%filter]=candidate_df.loc[candidate_df.index.get_level_values('mvs_ids')==mvs_ids].loc[(filter),'counts'].values[0]
                    DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,'ecounts%s'%filter]=candidate_df.loc[candidate_df.index.get_level_values('mvs_ids')==mvs_ids].loc[(filter),'ecounts'].values[0]
                    DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,'m%s'%filter]=candidate_df.loc[candidate_df.index.get_level_values('mvs_ids')==mvs_ids].loc[(filter),'mag'].values[0]
                    DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,'e%s'%filter]=candidate_df.loc[candidate_df.index.get_level_values('mvs_ids')==mvs_ids].loc[(filter),'emag'].values[0]
                    # DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,'Nsigma%s'%filter]=candidate_df.loc[candidate_df.index.get_level_values('mvs_ids')==mvs_ids].loc[(filter),'Nsigma'].values

                    DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,'%s_ROTA'%filter]=candidate_df.loc[candidate_df.index.get_level_values('mvs_ids')==mvs_ids].loc[(filter),'ROTA'].values[0]
                    DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,'%s_PA_V3'%filter]=candidate_df.loc[candidate_df.index.get_level_values('mvs_ids')==mvs_ids].loc[(filter),'PA_V3'].values[0]
                    # DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,'%s_std'%filter]=candidate_df.loc[candidate_df.index.get_level_values('mvs_ids')==mvs_ids].loc[(filter),'std'].values[0]
                    DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,'%s_Nsigma'%filter]=candidate_df.loc[candidate_df.index.get_level_values('mvs_ids')==mvs_ids].loc[(filter),'Nsigma'].values[0]
                    DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,'%s_Kmode'%filter]=Kmode_final
                    DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,'%s_flag'%filter]=candidate_df.loc[candidate_df.index.get_level_values('mvs_ids')==mvs_ids].loc[(filter),'flag'].values[0]
                    DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,'%s_sep'%filter]=candidate_df.loc[candidate_df.index.get_level_values('mvs_ids')==mvs_ids].loc[(filter),'sep'].values[0]
                except:
                    display(candidate_df)
                    raise ValueError('Problematic mvs_id %i. please check'%mvs_ids)

        counts=candidate_df.loc[filter,['counts']].values.T[0]
        ecounts=candidate_df.loc[filter,['ecounts']].values.T[0]
        counts=counts.astype(float)
        ecounts=ecounts.astype(float)
        Mask=~(np.isnan(counts))
        counts=counts[Mask]
        ecounts=ecounts[Mask]
        if len(counts)>=1:
            if len(counts)>1:
                c,ec=np.average(counts,weights=1/ecounts**2,axis=0,returned=True)
            else:pass
                # c,ec=[counts,ecounts]
            mags=candidate_df.loc[filter,['mag']].values.T[0]
            emags=candidate_df.loc[filter,['emag']].values.T[0]
            mags=mags.astype(float)
            emags=emags.astype(float)
            mags=mags[Mask]
            emags=emags[Mask]
            if len(mags)>1:
                m,w=np.average(mags,weights=1/emags**2,axis=0,returned=True)
                em=1/np.sqrt(w)
            else:m,em=[mags,emags]
            DF.avg_candidates_df.loc[DF.avg_candidates_df.avg_ids.isin(DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.mvs_ids.isin(candidate_df.index.get_level_values('mvs_ids').unique())].avg_ids),'m%s'%filter]=np.round(m,3)
            DF.avg_candidates_df.loc[DF.avg_candidates_df.avg_ids.isin(DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.mvs_ids.isin(candidate_df.index.get_level_values('mvs_ids').unique())].avg_ids),'e%s'%filter]=np.round(em,3)
            DF.avg_candidates_df.loc[DF.avg_candidates_df.avg_ids.isin(DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.mvs_ids.isin(candidate_df.index.get_level_values('mvs_ids').unique())].avg_ids),'N%s'%filter]=len(emags)
    if verbose:
        display(DF.avg_candidates_df.loc[DF.avg_candidates_df.avg_ids.isin(DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.mvs_ids.isin(candidate_df.index.get_level_values('mvs_ids').unique())].avg_ids)])
        display(DF.mvs_candidates_df.loc[(DF.mvs_candidates_df.mvs_ids.isin(candidate_df.index.get_level_values('mvs_ids').unique()))])

def update_candidates(DF,avg_ids_list=[],suffix='',d=1.,skip_filters='F658N',aptype='4pixels',verbose=False,workers=None,noBGsub=False,sigma=2.5,min_mag_list=[],max_mag_list=[],DF_fk=None,parallel_runs=True,update_header=True,chunksize = None,label='data',kill_plots=True,delta=3,radius=3,sat_thr=np.inf,mkd=2,mad=0.1):
    '''
    update the multivisits candidates dataframe with candidates infos

    Parameters
    ----------
    avg_ids_list : list, optional
        list of ids from the average dataframe to test. The default is [].
    suffix: str, optional
        suffix to append to mag label. For example, if original photometry is present in the catalog, it canbe use with suffix='_o'.
        Default is ''.
    d : float, optional
        maximum distances between candidate's detections to accept it. The default is 1.5.
    skip_filters : list, optional
        list of filters to skip. The default is ''.
    aptype : (circular,square,4pixels), optional
       defin the aperture type to use during aperture photometry.
       The default is '4pixels'.
    verbose : bool, optional
        choose to show print and plots. The default is False.
    workers : int, optional
        number of workers to split the work accross multiple CPUs. The default is 3.
    noBGsub : bool
        choose to skip sky subtraction from tile.
    sigma : float
        value of the sigma clip.
    min_mag_list: list
        list of magnitudes (one for filter) to us as upper cut for suitable stars selection to evaluate the delta for 4p aperture photometry
    max_mag_list: list
        list of magnitudes (one for filter) to us as lower cut for suitable stars selection to evaluate the delta for 4p aperture photometry
    DF_fk : dataframe class
        dataframe class containing the fake dataframe. If None look into DF. The default is None.
    parallel_runs: bool, optional
        Choose to to split the workload over different CPUs. The default is True
    update_header:bool, optional
        choose update the photometry entry in the header
    num_of_chunks : int, optional
        number of chunk to split the targets. The default is None.
    chunksize : int, optional
        size of each chunk for the parallelization process. The default is None.
    mkd: int, optional
        minimum filter detections per KLIPmode to accept a candidate
    mad: int, optional
        minimum arecsecond distance from center to accept a candidate

    Returns
    -------
    None.

    '''
    DF.header_df.loc['mad','Values']=mad
    DF.header_df.loc['mkd','Values']=mkd

    if update_header:
        update_header_photometry(DF,suffix=suffix,skip_filters=skip_filters,aptype=aptype,verbose=False,workers=workers,sigma=sigma,min_mag_list=min_mag_list,max_mag_list=max_mag_list,DF_fk=DF_fk,parallel_runs=parallel_runs,chunksize = chunksize, label=label,delta=delta,sat_thr=sat_thr)

    else:
        if verbose: display(DF.header_df)

    if parallel_runs:
        print('Working on the candidates')
        workers,chunksize,ntarget=parallelization_package(workers,len(avg_ids_list),chunksize = chunksize)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for Kmode_final,candidate_df in tqdm(executor.map(task_mvs_candidates_infos,repeat(DF),avg_ids_list,repeat(d),repeat(skip_filters),repeat(aptype),repeat(verbose),repeat(False),repeat(3),repeat(DF_fk),repeat(label),repeat(kill_plots),repeat(delta),repeat(radius),repeat(sat_thr),repeat(mkd),repeat(mad),chunksize=chunksize)):
                if Kmode_final!=None: update_candidates_with_detection(DF,candidate_df,Kmode_final,verbose)

    else:
        for avg_id in tqdm(avg_ids_list):
            try:
                Kmode_final,candidate_df=task_mvs_candidates_infos(DF,avg_id,d,skip_filters,aptype,verbose,False,3,DF_fk,label,kill_plots,delta,radius,sat_thr,mkd,mad)
                if Kmode_final!=None: update_candidates_with_detection(DF,candidate_df,Kmode_final,verbose)
            except:
                raise ValueError('Something wrong with avg_id %s, Please check'%avg_id)

    print('> Pruning the mvs_candidate_df:')
    mvs_bad_ids_list=[]
    for avg_ids in tqdm(DF.avg_candidates_df.avg_ids):
        mvs_ids_list=DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.avg_ids==avg_ids].mvs_ids.unique()

        sel_mvs_ids=(DF.mvs_candidates_df.mvs_ids.isin(mvs_ids_list))
        sel_flags=(('good_candidate' == DF.mvs_candidates_df[['%s_flag'%i for i in DF.filters_list]]).astype(int).sum(axis=1)<1)

        mvs_bad_ids=DF.mvs_candidates_df.loc[sel_mvs_ids&sel_flags].mvs_ids.unique()
        mvs_bad_ids_list.extend(mvs_bad_ids)

    DF.mvs_candidates_df=DF.mvs_candidates_df.loc[~DF.mvs_candidates_df.mvs_ids.isin(mvs_bad_ids_list)].reset_index(drop=True)
    print('> Finishing up the candidate dfs:')
    selected_avg_ids=DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.mvs_ids.isin(DF.mvs_candidates_df.mvs_ids)].avg_ids.unique()
    DF.avg_candidates_df=DF.avg_candidates_df.loc[DF.avg_candidates_df.avg_ids.isin(selected_avg_ids)].reset_index(drop=True)
    for avg_ids in tqdm(DF.avg_candidates_df.avg_ids):
        mvs_ids_list=DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.avg_ids==avg_ids].mvs_ids.unique()
        DF.avg_candidates_df.loc[DF.avg_candidates_df.avg_ids==avg_ids,'mKmode']=np.nanmedian(DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids.isin(mvs_ids_list),['%s_Kmode'%filter for filter in DF.filters_list]].values)
        DF.avg_candidates_df.loc[DF.avg_candidates_df.avg_ids==avg_ids,'sep']=np.nanmean(DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids.isin(mvs_ids_list),['%s_sep'%filter for filter in DF.filters_list]].values)

        for filter in DF.filters_list:
            if filter not in skip_filters and not DF.avg_candidates_df.loc[DF.avg_candidates_df.avg_ids==avg_ids,'m%s'%filter].isna().values[0]:
                DF.avg_candidates_df.loc[DF.avg_candidates_df.avg_ids==avg_ids,'Nsigma%s'%filter]=np.nanmedian(DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids.isin(mvs_ids_list),'%s_Nsigma'%filter].values)
                try:
                    DF.avg_candidates_df.loc[(DF.avg_candidates_df.avg_ids==avg_ids),'MagBin%s'%filter]=DF.avg_targets_df.loc[(DF.avg_targets_df.avg_ids==avg_ids),'m%s%s'%(filter,suffix)].values[0].astype(int)
                except:
                    DF.avg_candidates_df[(DF.avg_candidates_df.avg_ids==avg_ids),'MagBin%s'%filter]=np.nan
    display(DF.avg_candidates_df)

def update_candidates_photometry(DF,avg_ids_list=[],label='data',aptype='4pixels',verbose=False,noBGsub=False,sigma=2.5,DF_fk=None,kill_plots=True,delta=3,skip_filters=[],sat_thr=np.inf,suffix=''):
    KLIP_label_dict={'data':'Kmode','crclean_data':'crclean_Kmode'}
    if DF_fk==None:DF_fk=DF
    if len(avg_ids_list)==0:
        avg_ids_list=DF.avg_candidates_df.avg_ids.unique()
    print('Updating the candidates photometry. Loading a total of %i targets'%len(avg_ids_list))


    for avg_ids in tqdm(avg_ids_list):
        mvs_ids_list=DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids.isin(DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.avg_ids==avg_ids].mvs_ids.values)].mvs_ids.unique()
        if verbose:
            print('> Before:')
            display(DF.avg_candidates_df.loc[DF.avg_candidates_df.avg_ids==avg_ids])
            display(DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids.isin(mvs_ids_list)])
        zelno=0
        for filter in DF.filters_list:
            if filter not in skip_filters:
                zpt=DF.header_df.loc['Delta%s'%filter,'Values']
                ezpt=DF.header_df.loc['eDelta%s'%filter,'Values']
                zelno+=1
                for mvs_ids in mvs_ids_list:
                    if DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,'%s_flag'%filter].values[0]!='rejected':
                        # print(DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,'%s_flag'%filter].values[0])
                        magbin=DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==mvs_ids,'m%s%s'%(filter,suffix)].values[0].astype(int)
                        sep=DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,'%s_sep'%filter].values[0]
                        mag=DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,'m%s'%filter].values[0]
                        x,y=DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,['x%s_tile'%filter,'y%s_tile'%filter]].values[0]
                        Kmode=DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,'%s_Kmode'%filter].astype(int).values[0]
                        exptime=DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==mvs_ids,'exptime_%s'%filter].values[0]
                        if not np.isnan(mag) and mag-magbin>=0 and mag-magbin<=DF_fk.fk_candidates_df.index.get_level_values('dmag').astype(int).max() and int(magbin)>=DF_fk.fk_candidates_df.index.get_level_values('magbin').astype(int).min() and int(magbin)<=DF_fk.fk_candidates_df.index.get_level_values('magbin').astype(int).max() and int(sep)>=DF_fk.fk_candidates_df.index.get_level_values('sep').astype(int).min() and int(sep)<=DF_fk.fk_candidates_df.index.get_level_values('sep').astype(int).max():
                            thrpt,ethrpt=KLIP_throughput(DF_fk,sep,filter,int(magbin),int(mag-magbin),Kmode,verbose=False)
                            path2tile='%s/%s/%s/%s/mvs_tiles/%s/tile_ID%i.fits'%(path2data,DF.project,DF.target,DF.inst,filter,mvs_ids)
                            KDATA=Tile(x=x,y=y,tile_base=DF.tilebase,inst=DF.inst)
                            KDATA.load_tile(path2tile,ext='%s%s'%(KLIP_label_dict[label],Kmode),verbose=False,return_Datacube=False)

                            if not np.all(np.isnan(KDATA.data)): counts,ecounts,Nsigma,Nap,mag,emag,spx,bpx,Sky,eSky,nSky,grow_corr=KLIP_aperture_photometry_handler(DF,mvs_ids,filter,x=x,y=y,data=KDATA.data,zpt=zpt,ezpt=ezpt,aptype=aptype,noBGsub=False,sigma=sigma,kill_plots=True,Python_origin=True,delta=delta,sat_thr=sat_thr,exptime=exptime,thrpt=thrpt[0],ethrpt=ethrpt[0])#(DF,mvs_ids,filter,x=x,y=y,data=KDATA.data,zpt=zpt,ezpt=ezpt,aptype=aptype,noBGsub=False,sigma=sigma,kill_plots=True,Python_origin=True,delta=delta,sat_thr=sat_thr,exptime=exptime,thrpt=thrpt[0],ethrpt=ethrpt[0],candidate=True)
                            else: counts,ecounts,mag,emag=[np.nan,np.nan,np.nan,np.nan,np.nan]
                            DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,['counts%s'%filter,'ecounts%s'%filter,'Nsigma%s'%filter,'m%s'%filter,'e%s'%filter]]=[counts,ecounts,Nsigma,mag,emag]
                        else:
                            columns=DF.mvs_candidates_df.columns[DF.mvs_candidates_df.columns.str.contains(filter)]
                            DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,columns]=np.nan
                            DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,'%s_flag'%filter]='rejected'
                    else:
                        columns=DF.mvs_candidates_df.columns[DF.mvs_candidates_df.columns.str.contains(filter)]
                        DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,columns]=np.nan
                        DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,'%s_flag'%filter]='rejected'

                counts=DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids.isin(mvs_ids_list),'counts%s'%filter].values
                counts=counts.astype(float)
                ecounts=DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids.isin(mvs_ids_list),'ecounts%s'%filter].values
                ecounts=ecounts.astype(float)
                Mask=~(np.isnan(counts))

                if len(counts[Mask])>0:
                    _,_,_,Mask=print_mean_median_and_std_sigmacut(counts,verbose=False,sigma=sigma)
                    counts=counts[Mask]
                    ecounts=ecounts[Mask]
                    c,ec=np.average(counts,weights=1/ecounts**2,axis=0,returned=True)

                    mags=DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids.isin(mvs_ids_list),'m%s'%filter].values
                    emags=DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids.isin(mvs_ids_list),'e%s'%filter].values
                    mags=mags.astype(float)
                    emags=emags.astype(float)

                    mags=mags[Mask]
                    emags=emags[Mask]
                    m,w=np.average(mags,weights=1/emags**2,axis=0,returned=True)
                    em=1/np.sqrt(w)
                else:
                    m,em=[np.nan,np.nan]
                sep=np.nanmean(DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids.isin(mvs_ids_list),['%s_sep'%filter for filter in DF.filters_list]].values)
                DF.avg_candidates_df.loc[DF.avg_candidates_df.avg_ids==avg_ids,'m%s'%filter]=np.round(m,3)
                DF.avg_candidates_df.loc[DF.avg_candidates_df.avg_ids==avg_ids,'e%s'%filter]=np.round(em,3)
                DF.avg_candidates_df.loc[DF.avg_candidates_df.avg_ids==avg_ids,'sep']=np.round(sep,3)
                dmag=DF.avg_candidates_df.loc[DF.avg_candidates_df.avg_ids==avg_ids,'m%s'%filter].values[0]-DF.avg_targets_df.loc[DF.avg_targets_df.avg_ids==avg_ids,'m%s'%filter].values[0]
                if dmag<0:
                    columns=DF.avg_candidates_df.columns[DF.avg_candidates_df.columns.str.contains(filter)]
                    DF.avg_candidates_df.loc[DF.avg_candidates_df.avg_ids==avg_ids,columns]=np.nan
        if verbose:
            print('> After:')
            display(DF.avg_candidates_df.loc[DF.avg_candidates_df.avg_ids==avg_ids])
            display(DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids.isin(mvs_ids_list)])

    bad_mvs_ids=[]
    bad_avg_ids=[]
    for avg_ids in DF.avg_candidates_df.avg_ids.unique():
        mvs_ids_list=DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.avg_ids.isin([avg_ids])].mvs_ids.unique()
        if DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids.isin(mvs_ids_list),['%s_flag'%(filter) for filter in DF.filters_list]].apply(lambda x: x.str.contains('rejected',case=False)).all(axis=1).all(axis=0):
            bad_mvs_ids.extend(mvs_ids_list)
            bad_avg_ids.append(avg_ids)
    DF.mvs_candidates_df=DF.mvs_candidates_df.loc[~DF.mvs_candidates_df.mvs_ids.isin(bad_mvs_ids)].reset_index(drop=True)
    DF.avg_candidates_df=DF.avg_candidates_df.loc[~DF.avg_candidates_df.avg_ids.isin(bad_avg_ids)].reset_index(drop=True)


###########################################
# Target tiles dataframe related routines #
###########################################
def task_mvs_tiles(DF,fitsname,ids_list,filter,use_xy_SN,use_xy_m,use_xy_cen,xy_shift_list,xy_dmax,legend,overwrite,verbose,Python_origin,cr_remove,la_cr_remove,cr_radius,multiply_by_exptime,multiply_by_gain,multiply_by_PAM,redo):
    '''
    parallelized task for the update_mvs_tiles.
    '''
    # out=[]
    path2fits=DF.path2data+'/'+fitsname+DF.fitsext+'.fits'
    hdul = fits.open(path2fits)
    if len(hdul)>=4:
        # phot=[]
        for id in ids_list:
            path2tile='%s/mvs_tiles/%s/tile_ID%i.fits'%(DF.path2out,filter,id)
            if redo or not os.path.exists(path2tile):
                type_flag=DF.avg_targets_df.loc[DF.avg_targets_df.avg_ids==DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.mvs_ids==id].avg_ids.unique()[0]].type.values[0]
                x=DF.mvs_targets_df.loc[(DF.mvs_targets_df['fits_%s'%(filter)]==fitsname)&(DF.mvs_targets_df.mvs_ids==id),'x_%s'%filter].values[0]
                y=DF.mvs_targets_df.loc[(DF.mvs_targets_df['fits_%s'%(filter)]==fitsname)&(DF.mvs_targets_df.mvs_ids==id),'y_%s'%filter].values[0]
                ext=int(DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==id].ext.values[0])
                if x % 0.5==0: x-=0.001
                if y % 0.5==0: y-=0.001

                if not np.isnan(x) and not np.isnan(y):
                    flag='good_target'
                    if overwrite or not os.path.isfile(path2tile) :
                        getLogger(__name__).info(f'Making tile {path2tile}')
                        exptime=DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==id,'exptime_%s'%filter].values[0]
                        SCI=hdul[ext].data.copy()
                        if multiply_by_exptime:
                            SCI*=exptime

                        if multiply_by_gain:
                            SCI*=DF.gain

                        if multiply_by_PAM:
                            path2PAM='%s/PAM'%(DF.path2data)
                            phdul=fits.open(path2PAM+'/'+str(DF.PAMdict[ext]+'.fits'))
                            try:PAM=phdul[1].data
                            except:PAM=phdul[0].data
                            SCI*=PAM

                        ERR=hdul[ext+1].data
                        DQ=hdul[ext+2].data

                        if type_flag==2 and (use_xy_cen==True or use_xy_SN==True):
                            flag='unresolved_double'
                            xy_m=True
                            xy_cen=False
                        else:
                            xy_cen=use_xy_cen
                            xy_m=use_xy_m

                        if xy_cen:
                            if cr_remove or la_cr_remove:
                                DQDATA=Tile(data=DQ,x=x,y=y,tile_base=DF.tilebase,delta=6,inst=DF.inst,Python_origin=Python_origin)
                                DQDATA.mk_tile(pad_data=True,legend=legend,showplot=False,verbose=verbose,title='shiftedDQ',kill_plots=True,cbar=True)

                                IDATA=Tile(data=SCI,x=x,y=y,tile_base=DF.tilebase,delta=6,dqdata=DQDATA.data,inst=DF.inst,Python_origin=Python_origin)
                                IDATA.mk_tile(xy_m=True,pad_data=True,legend=legend,showplot=False,verbose=verbose,title='CRcleanSCI',kill_plots=True,cr_remove=cr_remove, la_cr_remove=la_cr_remove,cr_radius=cr_radius,cbar=True)
                            else:

                                IDATA=Tile(data=SCI,x=x,y=y,tile_base=DF.tilebase,delta=6,inst=DF.inst,Python_origin=Python_origin)
                                IDATA.mk_tile(pad_data=True,legend=False,showplot=False,verbose=False,xy_cen=True,xy_dmax=xy_dmax,title='OrigSCI',kill_plots=True,cbar=True)

                            deltax=IDATA.x_cen-(IDATA.tile_base-1)/2
                            deltay=IDATA.y_cen-(IDATA.tile_base-1)/2

                        elif xy_m:
                            if cr_remove or la_cr_remove:
                                DQDATA=Tile(data=DQ,x=x,y=y,tile_base=DF.tilebase,delta=6,inst=DF.inst,Python_origin=Python_origin)
                                DQDATA.mk_tile(pad_data=True,legend=legend,showplot=False,verbose=verbose,title='shiftedDQ',kill_plots=True,cbar=True)
                                IDATA=Tile(data=SCI,x=x,y=y,tile_base=DF.tilebase,delta=6,dqdata=DQDATA.data,inst=DF.inst,Python_origin=Python_origin)
                                IDATA.mk_tile(xy_m=True,pad_data=True,legend=legend,showplot=False,verbose=verbose,title='CRcleanSCI',kill_plots=True,cr_remove=cr_remove, la_cr_remove=la_cr_remove,cr_radius=cr_radius,cbar=True)
                            else:
                                IDATA=Tile(data=SCI,x=x,y=y,tile_base=DF.tilebase,delta=6,inst=DF.inst,Python_origin=Python_origin)
                                IDATA.mk_tile(pad_data=True,legend=False,showplot=False,verbose=False,xy_m=True,xy_dmax=xy_dmax,title='OrigSCI',kill_plots=True,cbar=True)

                            deltax=IDATA.x_m-(IDATA.tile_base-1)/2
                            deltay=IDATA.y_m-(IDATA.tile_base-1)/2

                        else:
                            if cr_remove or la_cr_remove:
                                DQDATA=Tile(data=DQ,x=x,y=y,tile_base=DF.tilebase,delta=6,inst=DF.inst,Python_origin=Python_origin)
                                DQDATA.mk_tile(pad_data=True,legend=legend,showplot=False,verbose=verbose,title='shiftedDQ',kill_plots=True,cbar=True)

                                IDATA=Tile(data=SCI,x=x,y=y,tile_base=DF.tilebase,delta=6,dqdata=DQDATA.data,inst=DF.inst,Python_origin=Python_origin)
                                IDATA.mk_tile(xy_m=True,pad_data=True,legend=legend,showplot=False,verbose=verbose,title='CRcleanSCI',kill_plots=True,cr_remove=cr_remove, la_cr_remove=la_cr_remove,cr_radius=cr_radius,cbar=True)
                            else:

                                IDATA=Tile(data=SCI,x=x,y=y,tile_base=DF.tilebase,delta=6,inst=DF.inst,Python_origin=Python_origin)
                                IDATA.mk_tile(pad_data=True,legend=False,showplot=False,verbose=False,title='OrigSCI',kill_plots=True,cbar=True)
                            deltax=0
                            deltay=0


                        if len(xy_shift_list)>0:
                            deltax+=xy_shift_list[0]
                            deltay+=xy_shift_list[1]

                        #making the tile datacube and save it
                        DF.mvs_targets_df.loc[(DF.mvs_targets_df['fits_%s'%(filter)]==fitsname)&(DF.mvs_targets_df.mvs_ids==id),['x_%s'%filter,'y_%s'%filter]]=[x+deltax,y+deltay]
                        x=DF.mvs_targets_df.loc[(DF.mvs_targets_df['fits_%s'%(filter)]==fitsname)&(DF.mvs_targets_df.mvs_ids==id),'x_%s'%filter].values[0]#-0.001
                        y=DF.mvs_targets_df.loc[(DF.mvs_targets_df['fits_%s'%(filter)]==fitsname)&(DF.mvs_targets_df.mvs_ids==id),'y_%s'%filter].values[0]#-0.001

                        DATA=Tile(data=SCI,x=x,y=y,tile_base=DF.tilebase,delta=6,inst=DF.inst,Python_origin=Python_origin)
                        DATA.mk_tile(pad_data=True,legend=legend,showplot=False,verbose=verbose,title='shiftedSCI',kill_plots=True,cbar=True)
                        Datacube=DATA.append_tile(path2tile,Datacube=None,verbose=False,name='SCI',return_Datacube=True)

                        EDATA=Tile(data=ERR,x=x,y=y,tile_base=DF.tilebase,delta=6,inst=DF.inst,Python_origin=Python_origin)
                        EDATA.mk_tile(pad_data=True,legend=legend,showplot=False,verbose=verbose,title='shiftedERR',kill_plots=True,cbar=True)
                        Datacube=EDATA.append_tile(path2tile,Datacube=Datacube,verbose=False,name='ERR',return_Datacube=True)

                        return_Datacube=False
                        if cr_remove or la_cr_remove: return_Datacube=True
                        DQDATA=Tile(data=DQ,x=x,y=y,tile_base=DF.tilebase,delta=6,inst=DF.inst,Python_origin=Python_origin)
                        DQDATA.mk_tile(pad_data=True,legend=legend,showplot=False,verbose=verbose,title='shiftedDQ',kill_plots=True,cbar=True)
                        Datacube=DQDATA.append_tile(path2tile,Datacube=Datacube,verbose=False,name='DQ',return_Datacube=return_Datacube)

                        if cr_remove or la_cr_remove:
                            CRDATA=Tile(data=SCI,x=x,y=y,tile_base=DF.tilebase,delta=6,dqdata=DQDATA.data,inst=DF.inst,Python_origin=Python_origin)
                            CRDATA.mk_tile(pad_data=True,legend=legend,showplot=False,verbose=verbose,title='shiftedCRcleanSCI',kill_plots=True,cr_remove=cr_remove, la_cr_remove=la_cr_remove,cr_radius=cr_radius,cbar=True)
                            Datacube=CRDATA.append_tile(path2tile,Datacube=Datacube,verbose=False,name='CRcelanSCI',return_Datacube=False)

                        # phot.append(mvs_aperture_photometry(DF,filter,ee_df,zpt,fitsname=fitsname,mvs_ids_list_in=[id],bpx_list=bpx_list,spx_list=spx_list,la_cr_remove=la_cr_remove,cr_radius=cr_radius,radius_in=radius_in,radius1_in=radius1_in,radius2_in=radius2_in,sat_thr=sat_thr,kill_plots=kill_plots,grow_curves=grow_curves,r_in=r_in,p=p,gstep=gstep,flag=flag,multiply_by_exptime=multiply_by_exptime,multiply_by_gain=multiply_by_gain,multiply_by_PAM=multiply_by_PAM))
            else:
                getLogger(__name__).info(f'Tile {path2tile} already exist and redo = {redo}. Skipping.')
    #             sel = (DF.mvs_targets_df.mvs_ids == id)
    #             phot.append(DF.mvs_targets_df.loc[sel, ['ext', 'counts_%s' % filter, 'ecounts_%s' % filter, 'nap_%s' % filter,
    #                       'm_%s' % filter, 'e_%s' % filter, 'spx_%s' % filter,
    #                       'bpx_%s' % filter, 'r_%s' % filter, 'rsky1_%s' % filter, 'rsky2_%s' % filter,
    #                       'sky_%s' % filter, 'esky_%s' % filter, 'nsky_%s' % filter,
    #                       'grow_corr_%s' % filter, 'flag_%s'%filter]].tolist())
    #     else:
    #         phot.append([id,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,'rejected'])
    #
    # else:
    #     phot=[]
    #     for id in ids_list: phot.append([id,ext,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,'rejected'])
    # if not kill_plots:
    #     print('> m%s e%s spx%s bpx%s %s_r %s_rsky1 %s_rsky2'%(filter,filter,filter,filter,filter,filter,filter))
    #     print('> ', phot)
    # return(phot)

def task_mvs_photometry(DF,fitsname,ids_list,filter,ee_df,zpt,la_cr_remove,cr_radius,multiply_by_exptime,multiply_by_gain,multiply_by_PAM,redo,bpx_list,spx_list,radius_in,radius1_in,radius2_in,sat_thr,kill_plots,grow_curves,r_in,p,gstep):
    '''
    parallelized task for the update_mvs_tiles.
    '''
    # out=[]
    path2fits=DF.path2data+'/'+fitsname+DF.fitsext+'.fits'
    hdul = fits.open(path2fits)
    if len(hdul)>=4:
        phot=[]
        for id in ids_list:
            type_flag = DF.avg_targets_df.loc[DF.avg_targets_df.avg_ids == DF.crossmatch_ids_df.loc[
                DF.crossmatch_ids_df.mvs_ids == id].avg_ids.unique()[0]].type.values[0]
            flag = 'good_target'
            if type_flag == 2:
                flag = 'unresolved_double'
            if redo:
                phot.append(mvs_aperture_photometry(DF,filter,ee_df,zpt,fitsname=fitsname,mvs_ids_list_in=[id],bpx_list=bpx_list,spx_list=spx_list,la_cr_remove=la_cr_remove,cr_radius=cr_radius,radius_in=radius_in,radius1_in=radius1_in,radius2_in=radius2_in,sat_thr=sat_thr,kill_plots=kill_plots,grow_curves=grow_curves,r_in=r_in,p=p,gstep=gstep,flag=flag,multiply_by_exptime=multiply_by_exptime,multiply_by_gain=multiply_by_gain,multiply_by_PAM=multiply_by_PAM))
            else:
                getLogger(__name__).info(f'Redo = {redo}. Skipping.')
                sel = (DF.mvs_targets_df.mvs_ids == id)
                phot.append(DF.mvs_targets_df.loc[sel, ['ext', 'counts_%s' % filter, 'ecounts_%s' % filter, 'nap_%s' % filter,
                          'm_%s' % filter, 'e_%s' % filter, 'spx_%s' % filter,
                          'bpx_%s' % filter, 'r_%s' % filter, 'rsky1_%s' % filter, 'rsky2_%s' % filter,
                          'sky_%s' % filter, 'esky_%s' % filter, 'nsky_%s' % filter,
                          'grow_corr_%s' % filter, 'flag_%s'%filter]].tolist())
        else:
            phot.append([id,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,'rejected'])
    #
    else:
        phot=[]
        for id in ids_list: phot.append([id,ext,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,'rejected'])
    if not kill_plots:
        print('> m%s e%s spx%s bpx%s %s_r %s_rsky1 %s_rsky2'%(filter,filter,filter,filter,filter,filter,filter))
        print('> ', phot)
    return(phot)

def mk_mvs_tiles(DF,filter,mvs_ids_test_list=[],overwrite=True,xy_SN=True,xy_m=False,xy_cen=False,xy_shift_list=[],xy_dmax=3,bpx_list=[],spx_list=[],legend=False,showplot=False,showplot_final=False,verbose=False,workers=None,Python_origin=True,parallel_runs=True,cr_remove=False,la_cr_remove=False,cr_radius=3,kill_plots=False,chunksize = None,ee_df=None,zpt=0,radius_in=10,radius1_in=10,radius2_in=15,sat_thr=np.inf,grow_curves=True,r_in=1,p=100,gstep=0.1,multiply_by_exptime=False,multiply_by_gain=False,multiply_by_PAM=False,skip_photometry=False,redo=False):
    '''
    update the multi-visits tile dataframe with the tiles for each source

    '''
    if __name__ == 'utils_straklip':
        getLogger(__name__).info(f'Working on filter {filter}')
        config.make_paths(config=None, paths=DF.path2out+'/mvs_tiles/%s'%filter)
        fits_dict={}
        skip_IDs_list_list=[]

        if len(mvs_ids_test_list)!=0:
            fitsname_test_list=DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(mvs_ids_test_list)&~DF.mvs_targets_df.mvs_ids.isin(skip_IDs_list_list),'fits_%s'%(filter)].tolist()
            for index,row in DF.mvs_targets_df.loc[DF.mvs_targets_df['fits_%s'%(filter)].isin(fitsname_test_list)&DF.mvs_targets_df.mvs_ids.isin(mvs_ids_test_list)&~DF.mvs_targets_df.mvs_ids.isin(skip_IDs_list_list)].groupby('fits_%s'%(filter)):
                fits_dict[index]=row.mvs_ids.tolist()
        else:
            fitsname_test_list=DF.mvs_targets_df.loc[~DF.mvs_targets_df.mvs_ids.isin(skip_IDs_list_list),'fits_%s'%(filter)].unique().tolist()
            for index,row in DF.mvs_targets_df.loc[DF.mvs_targets_df['fits_%s'%(filter)].isin(fitsname_test_list)&~DF.mvs_targets_df.mvs_ids.isin(skip_IDs_list_list)].groupby('fits_%s'%(filter)):
                fits_dict[index]=row.mvs_ids.tolist()

        fits_dict = (dict(sorted(fits_dict.items(), key=lambda item: item[1])))
        ids_list_of_lists = list(fits_dict.values())
        fitsname_list = list(fits_dict.keys())

        if parallel_runs:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                workers, chunksize, ntarget = parallelization_package(workers, len(fitsname_list), chunksize=chunksize)
                getLogger(__name__).info(f'Loading a total of {ntarget} images')

                for _ in executor.map(task_mvs_tiles, repeat(DF), fitsname_list, ids_list_of_lists, repeat(filter),
                                      repeat(xy_SN), repeat(xy_m), repeat(xy_cen), repeat(xy_shift_list),
                                      repeat(xy_dmax), repeat(legend),
                                      repeat(overwrite), repeat(verbose), repeat(Python_origin),
                                      repeat(cr_remove), repeat(la_cr_remove), repeat(cr_radius),
                                      repeat(multiply_by_exptime), repeat(multiply_by_gain),
                                      repeat(multiply_by_PAM), repeat(redo),
                                      chunksize=chunksize):
                    pass
                    # phot=np.array(phot)
                    # for elno in range(len(phot)):
                    #     sel=(DF.mvs_targets_df.mvs_ids==phot[elno,0].astype(float))&(DF.mvs_targets_df.ext==phot[elno,1].astype(float))
                    #     DF.mvs_targets_df.loc[
                    #         sel, ['counts_%s' % filter, 'ecounts_%s' % filter, 'nap_%s' % filter,
                    #       'm_%s' % filter, 'e_%s' % filter, 'spx_%s' % filter,
                    #       'bpx_%s' % filter, 'r_%s' % filter, 'rsky1_%s' % filter, 'rsky2_%s' % filter,
                    #       'sky_%s' % filter, 'esky_%s' % filter, 'nsky_%s' % filter,
                    #       'grow_corr_%s' % filter]] = phot[elno, 2:-1].astype(float)
                    #     DF.mvs_targets_df.loc[sel,['%s_flag'%filter]]=phot[elno,-1:]

        else:
            for elno in range(len(fitsname_list)):
                if len(xy_shift_list)>0:
                    w=np.where(np.array(fitsname_list)[elno]==np.array(fitsname_test_list))[0][0]
                    use_xy_shift_list=xy_shift_list[w]
                else:use_xy_shift_list=[]
                # phot=
                task_mvs_tiles(DF,fitsname_list[elno],ids_list_of_lists[elno],filter,xy_SN,xy_m,xy_cen,
                               use_xy_shift_list,xy_dmax,legend,overwrite,verbose,
                               Python_origin,cr_remove,la_cr_remove,cr_radius,
                               multiply_by_exptime,multiply_by_gain,multiply_by_PAM,redo)
                # phot=np.array(phot)
                # for elno in range(len(phot)):
                #     sel=(DF.mvs_targets_df.mvs_ids==phot[elno,0].astype(float))&(DF.mvs_targets_df.ext==phot[elno,1].astype(float))
                #     DF.mvs_targets_df.loc[
                #         sel, ['counts_%s' % filter, 'ecounts_%s' % filter, 'nap_%s' % filter,
                #               'm_%s' % filter, 'e_%s' % filter, 'spx_%s' % filter,
                #               'bpx_%s' % filter, '%s_r' % filter, '%s_rsky1' % filter, '%s_rsky2' % filter,
                #               'sky_%s' % filter, 'esky_%s' % filter, 'nsky_%s' % filter,
                #               'grow_corr%s' % filter]] = phot[elno, 2:-1].astype(float)
                #     DF.mvs_targets_df.loc[sel,['%s_flag'%filter]]=phot[elno,-1:]

def mk_mvs_photometry(DF,filter,mvs_ids_test_list=[],overwrite=True,xy_SN=True,xy_m=False,xy_cen=False,xy_shift_list=[],xy_dmax=3,bpx_list=[],spx_list=[],legend=False,showplot=False,showplot_final=False,verbose=False,workers=None,Python_origin=True,parallel_runs=True,cr_remove=False,la_cr_remove=False,cr_radius=3,kill_plots=False,chunksize = None,ee_df=None,zpt=0,radius_in=10,radius1_in=10,radius2_in=15,sat_thr=np.inf,grow_curves=True,r_in=1,p=100,gstep=0.1,multiply_by_exptime=False,multiply_by_gain=False,multiply_by_PAM=False,skip_photometry=False,redo=False):
    '''
    update the multi-visits dataframe with the tile photometry for each source

    '''
    if __name__ == 'utils_straklip':
        getLogger(__name__).info(f'Working on filter {filter}')
        # config.make_paths(config=None, paths=DF.path2out + '/mvs_tiles/%s' % filter)
        fits_dict = {}
        skip_IDs_list_list = []

        if len(mvs_ids_test_list) != 0:
            fitsname_test_list = DF.mvs_targets_df.loc[
                DF.mvs_targets_df.mvs_ids.isin(mvs_ids_test_list) & ~DF.mvs_targets_df.mvs_ids.isin(
                    skip_IDs_list_list), 'fits_%s' % (filter)].tolist()
            for index, row in DF.mvs_targets_df.loc[
                DF.mvs_targets_df['fits_%s' % (filter)].isin(fitsname_test_list) & DF.mvs_targets_df.mvs_ids.isin(
                        mvs_ids_test_list) & ~DF.mvs_targets_df.mvs_ids.isin(skip_IDs_list_list)].groupby(
                    'fits_%s' % (filter)):
                fits_dict[index] = row.mvs_ids.tolist()
        else:
            fitsname_test_list = DF.mvs_targets_df.loc[
                ~DF.mvs_targets_df.mvs_ids.isin(skip_IDs_list_list), 'fits_%s' % (filter)].unique().tolist()
            for index, row in DF.mvs_targets_df.loc[
                DF.mvs_targets_df['fits_%s' % (filter)].isin(fitsname_test_list) & ~DF.mvs_targets_df.mvs_ids.isin(
                        skip_IDs_list_list)].groupby('fits_%s' % (filter)):
                fits_dict[index] = row.mvs_ids.tolist()

        fits_dict = (dict(sorted(fits_dict.items(), key=lambda item: item[1])))
        ids_list_of_lists = list(fits_dict.values())
        fitsname_list = list(fits_dict.keys())

        if parallel_runs:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                workers, chunksize, ntarget = parallelization_package(workers, len(fitsname_list), chunksize=chunksize)
                getLogger(__name__).info(f'Loading a total of {ntarget} images')

                for _ in executor.map(task_mvs_photometry, repeat(DF), fitsname_list, ids_list_of_lists, repeat(filter),
                                      repeat(xy_SN), repeat(xy_m), repeat(xy_cen), repeat(xy_shift_list),
                                      repeat(xy_dmax), repeat(legend),
                                      repeat(overwrite), repeat(verbose), repeat(Python_origin),
                                      repeat(cr_remove), repeat(la_cr_remove), repeat(cr_radius),
                                      repeat(multiply_by_exptime), repeat(multiply_by_gain),
                                      repeat(multiply_by_PAM), repeat(redo),
                                      chunksize=chunksize):
                    phot=np.array(phot)
                    for elno in range(len(phot)):
                        sel=(DF.mvs_targets_df.mvs_ids==phot[elno,0].astype(float))&(DF.mvs_targets_df.ext==phot[elno,1].astype(float))
                        DF.mvs_targets_df.loc[
                            sel, ['counts_%s' % filter, 'ecounts_%s' % filter, 'nap_%s' % filter,
                          'm_%s' % filter, 'e_%s' % filter, 'spx_%s' % filter,
                          'bpx_%s' % filter, 'r_%s' % filter, 'rsky1_%s' % filter, 'rsky2_%s' % filter,
                          'sky_%s' % filter, 'esky_%s' % filter, 'nsky_%s' % filter,
                          'grow_corr_%s' % filter]] = phot[elno, 2:-1].astype(float)
                        DF.mvs_targets_df.loc[sel,['%s_flag'%filter]]=phot[elno,-1:]

        else:
            for elno in range(len(fitsname_list)):
                if len(xy_shift_list) > 0:
                    w = np.where(np.array(fitsname_list)[elno] == np.array(fitsname_test_list))[0][0]
                    use_xy_shift_list = xy_shift_list[w]
                else:
                    use_xy_shift_list = []
                # phot=
                task_mvs_photometry(DF, fitsname_list[elno], ids_list_of_lists[elno], filter, xy_SN, xy_m, xy_cen,
                               use_xy_shift_list, xy_dmax, legend, overwrite, verbose,
                               Python_origin, cr_remove, la_cr_remove, cr_radius,
                               multiply_by_exptime, multiply_by_gain, multiply_by_PAM, redo)
                phot=np.array(phot)
                for elno in range(len(phot)):
                    sel=(DF.mvs_targets_df.mvs_ids==phot[elno,0].astype(float))&(DF.mvs_targets_df.ext==phot[elno,1].astype(float))
                    DF.mvs_targets_df.loc[
                        sel, ['counts_%s' % filter, 'ecounts_%s' % filter, 'nap_%s' % filter,
                              'm_%s' % filter, 'e_%s' % filter, 'spx_%s' % filter,
                              'bpx_%s' % filter, '%s_r' % filter, '%s_rsky1' % filter, '%s_rsky2' % filter,
                              'sky_%s' % filter, 'esky_%s' % filter, 'nsky_%s' % filter,
                              'grow_corr%s' % filter]] = phot[elno, 2:-1].astype(float)
                    DF.mvs_targets_df.loc[sel,['%s_flag'%filter]]=phot[elno,-1:]

def task_median_target_infos(DF,id,filter,overwrite,column_name,zfactor,alignment_box,legend,showplot,verbose,method,cr_remove,la_cr_remove,kill,kill_plots,suffix,goodness_phot_label,skip_flag):
    '''
    Taks perfomed in the update_median_targets_tile.
    Note: all multivisits targets with FILTER_flag == 'rejected' are considered NaN
    when arranging the median tile. If all are 'rejected', the final median tile for
    that FILTER will be all NaN.

    Parameters
    ----------
    id : int
        identification number for the dataframe.
    filter : str
        filter name.
    overwrite: bool, optional
        if False, skip IDs that already have tiles in directory
    column_name : str
        column name.
    zfactor : int, optional
        zoom factor to apply to re/debin each image.
    alignment_box : bool, optional
        half base of the square box centered at the center of the tile employed by the allignment process to allign the images.
        The box is constructed as alignment_box-x,x+alignment_box and alignment_box-y,y+alignment_box, where x,y are the
        coordinate of the center of the tile.
    legend : bool, optional
        choose to show legends in plots. The default is False.
    showplot : bool, optional
        choose to show plots. The default is False.
    verbose : bool, optional
        choose to show prints. The default is False.
    method : str, optional
        choose between median image or mean image. The default is median.
    cr_remove : bool, optional
        choose to apply cosmic ray removal. The default is False.
    la_cr_remove : bool, optional
        choose to apply L.A. cosmic ray removal. The default is False.
    kill : bool, optional
        choose to kill bad pixels instead of using the median of the neighbouring pixels. The default is False.
    kill_plots:
        choose to kill all plots created. The default is False.

    Returns
    -------
    None.

    '''
    if not DF.skip_photometry: phot=avg_aperture_photometry(DF,id,filter,goodness_phot_label,suffix,skip_flag)
    else: phot=[id,np.nan,np.nan,np.nan,np.nan]
    target_images=[]
    mvs_ids_list=DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.avg_ids==id].mvs_ids.unique()
    path2tile='%s/%s/%s/%s/median_tiles/%s/tile_ID%i.fits'%(path2data,DF.project,DF.target,DF.inst,filter,id)
    PAV_3s=[]
    ROTAs=[]
    if overwrite or not os.path.isfile(path2tile):
        for mvs_ids in mvs_ids_list:
            try:
                if not DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==mvs_ids,'%s_flag'%filter].str.contains(skip_flag).values[0]:
                    sel_ids=(DF.mvs_targets_df.mvs_ids==mvs_ids)
                    DATA=Tile(x=(DF.tilebase-1)/2,y=(DF.tilebase-1)/2,tile_base=DF.tilebase,delta=0,inst=DF.inst,Python_origin=True)
                    DATA.load_tile('%s/%s/%s/%s/mvs_tiles/%s/tile_ID%i.fits'%(path2data,DF.project,DF.target,DF.inst,filter,mvs_ids),ext=1,raise_errors=False)
                    if not np.all(np.isnan(DATA.data)):
                        target_images.append(DATA.data)
                        PAV_3s.append(DF.mvs_targets_df.loc[sel_ids,'%s_PA_V3'%filter].values[0])
                        ROTAs.append(DF.mvs_targets_df.loc[sel_ids,'%s_ROTA'%filter].values[0])
            except:
                print('>',mvs_ids_list,ROTAs,PAV_3s)
                sys.exit()

        if len(target_images)>0:
            target_images=np.array(target_images)
            return_Datacube=False
            try:
                target_tile,shift_list=allign_images(target_images,ROTAs,PAV_3s,filter,legend=legend,showplot=showplot,verbose=False,zfactor=zfactor,alignment_box=alignment_box,title='%s Median Target'%(filter),method=method,tile_base=DF.tilebase,kill=kill,kill_plots=kill_plots)
                if cr_remove or la_cr_remove: return_Datacube=True
                Datacube=target_tile.append_tile(path2tile,Datacube=None,verbose=False,name='SCI',return_Datacube=return_Datacube)
            except:
                print(mvs_ids_list,len(target_images),ROTAs,PAV_3s)
                sys.exit()
            if cr_remove or la_cr_remove:
                crtarget_images=[]
                try:
                    for mvs_ids in mvs_ids_list:
                        if not DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==mvs_ids,'%s_flag'%filter].str.contains(skip_flag).values[0] :
                            CRDATA=Tile(x=(DF.tilebase-1)/2,y=(DF.tilebase-1)/2,tile_base=DF.tilebase,delta=0,inst=DF.inst,Python_origin=True)
                            CRDATA.load_tile('%s/%s/%s/%s/mvs_tiles/%s/tile_ID%i.fits'%(path2data,DF.project,DF.target,DF.inst,filter,mvs_ids),ext=4)
                            crtarget_images.append(CRDATA.data)
                    if len(crtarget_images)>0:
                        crtarget_images=np.array(crtarget_images)
                        crtarget_tile,shift_list=allign_images(crtarget_images,ROTAs,PAV_3s,filter,legend=legend,showplot=showplot,verbose=False,zfactor=zfactor,alignment_box=alignment_box,title='%s CR clean Median Target'%(filter),method=method,tile_base=DF.tilebase,kill=kill,kill_plots=kill_plots)
                        crtarget_tile.append_tile(path2tile,Datacube=Datacube,verbose=False,name='CRcleanSCI',return_Datacube=False)
                except:
                    print(mvs_ids_list,len(crtarget_images),ROTAs,PAV_3s)
                    sys.exit()
            if return_Datacube: Datacube.close()
    if not showplot:plt.close('all')
    else:
        try:os.remove(path2tile)
        except:pass
    return(phot)


def mk_median_tiles_and_photometry(DF,filter,overwrite=True,avg_ids_list=[],column_name='data',workers=None,zfactor=10,alignment_box=3,legend=False,showplot=False,verbose=False,parallel_runs=True,method='median',cr_remove=False,la_cr_remove=False,chunksize = None,kill=False,kill_plots=False,suffix='',goodness_phot_label='e', skip_flag='rejected'):
    '''
    Update the median targets dataframe tile.

    Parameters
    ----------
    filter : str
        filter name.
    overwrite: bool, optional
        if False, skip IDs that already have tiles in directory. Default is True.
    avg_ids_list : list
        list of average ids. The default is [].
    column_name : list, optional
        list of column names. The default is 'data'.
    workers : int, optional
        number of workers for parallelization. The default is None.
    zfactor : int, optional
        zoom factor to apply to re/debin each image.
        The default is 10.
    alignment_box : bool, optional
        half base of the square box centered at the center of the tile employed by the allignment process to allign the images.
        The box is constructed as alignment_box-x,x+alignment_box and alignment_box-y,y+alignment_box, where x,y are the
        coordinate of the center of the tile.
        The default is 2.
    legend : bool, optional
        choose to show legends in plots. The default is False.
    showplot : bool, optional
        choose to show plots. The default is False.
    verbose : bool, optional
        choose to show prints. The default is False.
    parallel_runs: bool, optional
        Choose to to split the workload over different CPUs. The default is True
    method : str, optional
        choose between median image or mean image. The default is median.
    num_of_chunks : int, optional
        number of chunk to split the targets. The default is None.
    chunksize : int, optional
        size of each chunk for the parallelization process. The default is None.
    kill : bool, optional
        choose to kill bad pixels instead of using the median of the neighbouring pixels. The default is False.
    kill_plots:
        choose to kill all plots created. The default is False.

    Returns
    -------
    None.

    '''
    if __name__ == 'utils_straklip':
        if len(avg_ids_list)==0: avg_ids_list=DF.avg_targets_df.avg_ids.unique()
        if verbose:
            print('Loading a total of %i images'%len(avg_ids_list))
            print('Working in %s: '%filter)
        mk_dir('%s/%s/%s/%s/median_tiles/%s'%(path2data,DF.project,DF.target,DF.inst,filter),verbose=verbose)
        if parallel_runs:
            workers,chunksize,ntarget=parallelization_package(workers,len(avg_ids_list),chunksize = chunksize)
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for phot_out in tqdm(executor.map(task_median_target_infos,repeat(DF),avg_ids_list,repeat(filter),repeat(overwrite),repeat(column_name),repeat(zfactor),repeat(alignment_box),repeat(legend),repeat(showplot),repeat(verbose),repeat(method),repeat(cr_remove),repeat(la_cr_remove),repeat(kill),repeat(kill_plots),repeat(suffix),repeat(goodness_phot_label),repeat(skip_flag),chunksize=chunksize)):
                     phot_out=np.array(phot_out)
                     DF.avg_targets_df.loc[DF.avg_targets_df.avg_ids==phot_out[0],['m%s'%filter,'e%s'%filter,'spx%s'%filter,'bpx%s'%filter]]=phot_out[1:]

        else:
            for id in tqdm(avg_ids_list):
                phot_out=task_median_target_infos(DF,id,filter,overwrite,column_name,zfactor,alignment_box,legend,showplot,verbose,method,cr_remove,la_cr_remove,kill,kill_plots,suffix,goodness_phot_label,skip_flag)
                phot_out=np.array(phot_out)
                DF.avg_targets_df.loc[DF.avg_targets_df.avg_ids==phot_out[0],['m%s'%filter,'e%s'%filter,'spx%s'%filter,'bpx%s'%filter]]=phot_out[1:]
##############################################
# Candidate tiles dataframe related routines #
##############################################

def task_median_candidate_infos(DF,id,filter,column_name,zfactor,alignment_box,label):
    '''
    Taks perfomed in the update_median_candidate_tile

    Parameters
    ----------
    id : int
        identification number for the dataframe.
    filter : str
        filter name.
    column_name : str
        column name.
    zfactor : int, optional
        zoom factor to apply to re/debin each image.
    alignment_box : bool, optional
        half base of the square box centered at the center of the tile employed by the allignment process to allign the images.
        The box is constructed as alignment_box-x,x+alignment_box and alignment_box-y,y+alignment_box, where x,y are the
        coordinate of the center of the tile.

    Returns
    -------
    None.

    '''
    label_dict={'data':1,'crclean_data':4}
    hdul_dict={'data':1,'crclean_data':2}
    KLIP_label_dict={'data':'Kmode','crclean_data':'crclean_Kmode'}
    path2tile='%s/%s/%s/%s/median_tiles/%s/tile_ID%i.fits'%(path2data,DF.project,DF.target,DF.inst,filter,id)

    mvs_ids_list=DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids.isin(DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.avg_ids==id].mvs_ids.unique())].mvs_ids.unique()
    # DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.avg_ids==id].mvs_ids.values
    target_images=[]
    candidate_images=[]

    sel_ids=DF.mvs_targets_df.mvs_ids.isin(DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.avg_ids==id].mvs_ids.unique())
    sel_flag=(DF.mvs_targets_df['%s_flag'%filter]!='rejected')
    PAV_3s=DF.mvs_targets_df.loc[sel_ids&sel_flag,'%s_PA_V3'%filter].values
    ROTAs=DF.mvs_targets_df.loc[sel_ids&sel_flag,'%s_ROTA'%filter].values

    IMAGE=Tile(x=(DF.tilebase-1)/2,y=(DF.tilebase-1)/2,tile_base=DF.tilebase,delta=0,inst=DF.inst,Python_origin=True)
    Datacube=IMAGE.load_tile(path2tile,return_Datacube=True,hdul_max=hdul_dict[label],verbose=False,mode='update',raise_errors=False)
    # if not DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.avg_ids==id].mvs_ids),'%s_flag'%filter].str.contains('rejected').all():
    for mvs_ids in mvs_ids_list:
        if DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,'%s_flag'%filter].values[0]!='rejected':
            if not DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids.isin(DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.avg_ids==id].mvs_ids)].empty:
                Kmode=int(DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,'%s_Kmode'%filter].values[0])
            else:
                Kmode=int(max(DF.kmodes_list))
            IMAGE=Tile(x=(DF.tilebase-1)/2,y=(DF.tilebase-1)/2,tile_base=DF.tilebase,delta=0,inst=DF.inst,Python_origin=True)
            IMAGE.load_tile('%s/%s/%s/%s/mvs_tiles/%s/tile_ID%i.fits'%(path2data,DF.project,DF.target,DF.inst,filter,mvs_ids),ext=label_dict[label],raise_errors=False)
            image=np.array(IMAGE.data)
            if not np.all(np.isnan(image)):
                target_images.append(image)
                CANDIDATE=Tile(x=(DF.tilebase-1)/2,y=(DF.tilebase-1)/2,tile_base=DF.tilebase,delta=0,inst=DF.inst,Python_origin=True)
                CANDIDATE.load_tile('%s/%s/%s/%s/mvs_tiles/%s/tile_ID%i.fits'%(path2data,DF.project,DF.target,DF.inst,filter,mvs_ids),ext='%s%s'%(KLIP_label_dict[label],Kmode),raise_errors=False)
                candidate_images.append(CANDIDATE.data)
    if len(candidate_images)>0:
        target_tile,shift_list=allign_images(target_images,ROTAs,PAV_3s,filter,zfactor=zfactor,alignment_box=alignment_box,tile_base=DF.tilebase)
        candidate_tile,shift_list=allign_images(candidate_images,ROTAs,PAV_3s,filter,shift_list=shift_list,zfactor=zfactor,alignment_box=alignment_box,title=column_name,tile_base=DF.tilebase)
        if Datacube!=None:
            candidate_tile.append_tile(path2tile,Datacube=Datacube,verbose=False,name='%s'%(KLIP_label_dict[label]),return_Datacube=False,write=False)

def update_median_candidates_tile(DF,avg_ids_list,column_name='Kmode',workers=None,zfactor=10,alignment_box=3,parallel_runs=True,chunksize = None,label='data',kill_plots=True,skip_filters=[]):
    '''
    Update the median candidate dataframe tile.

    Parameters
    ----------
    avg_ids_list : list
        list of average ids.
    column_name : list, optional
        list of column names. The default is 'data'.
    workers : int, optional
        number of workers for parallelization. The default is None.
    zfactor : int, optional
        zoom factor to apply to re/debin each image.
        The default is 10.
    alignment_box : bool, optional
        half base of the square box centered at the center of the tile employed by the allignment process to allign the images.
        The box is constructed as alignment_box-x,x+alignment_box and alignment_box-y,y+alignment_box, where x,y are the
        coordinate of the center of the tile.
        The default is 2.
    parallel_runs: bool, optional
        Choose to to split the workload over different CPUs. The default is True
    num_of_chunks : int, optional
        number of chunk to split the targets. The default is None.
    chunksize : int, optional
        size of each chunk for the parallelization process. The default is None.

    Returns
    -------
    None.

    '''
    if __name__ == 'utils_straklip':
        print('Loading a total of %i images'%len(avg_ids_list))
        for filter in DF.filters_list:
            if filter not in skip_filters:
                print('Working in %s: '%filter)
                if parallel_runs:
                    workers,chunksize,ntarget=parallelization_package(workers,len(avg_ids_list),chunksize = chunksize)
                    with ProcessPoolExecutor(max_workers=workers) as executor:
                        for _ in tqdm(executor.map(task_median_candidate_infos,repeat(DF),avg_ids_list,repeat(filter),repeat(column_name),repeat(zfactor),repeat(alignment_box),repeat(label),chunksize=chunksize)):
                            pass
                else:
                    for id in tqdm(avg_ids_list):
                        # print(id)
                        task_median_candidate_infos(DF,id,filter,column_name,zfactor,alignment_box,label)


def update_photometry_after_KLIP_subtraction(DF,ee_df=None,label='data',Python_origin=True,kill_plots=True,suffix='',goodness_phot_label='e',skip_flag='rejected'):
    sat_thr=DF.header_df.loc['sat_thr','Values']
    radius_in=DF.header_df.loc['radius_in','Values']

    for q in range(len(DF.filters_list)):
        print('> Working on filter: ',DF.filters_list[q])
        for avg_ids in tqdm(DF.avg_candidates_df.avg_ids.unique()):
            for mvs_ids in DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids.isin(DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.avg_ids==avg_ids].mvs_ids)].mvs_ids.unique():
                filter=DF.filters_list[q]
                try:
                    if not DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==mvs_ids,'counts%s'%filter].isna().values[0] and not DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,'counts%s'%filter].isna().values[0]:
                        phot=mvs_aperture_photometry(DF,filter,ee_df,DF.ZPT_list[q],mvs_ids_list_in=[mvs_ids],kill_plots=kill_plots,Python_origin=Python_origin,sat_thr=sat_thr,radius_in=radius_in,remove_candidate=True,label=label,noBGsub=True,forceSky=True)
                        phot=np.array(phot)
                        for elno in range(len(phot)):
                            sel=(DF.mvs_targets_df.mvs_ids==phot[elno,0].astype(float))&(DF.mvs_targets_df.ext==phot[elno,1].astype(float))
                            DF.mvs_targets_df.loc[sel,['counts%s'%filter,'ecounts%s'%filter,'Nap%s'%filter,'m%s'%filter,'e%s'%filter,'spx%s'%filter,'bpx%s'%filter,'%s_r'%filter,'%s_rsky1'%filter,'%s_rsky2'%filter,'sky%s'%filter,'esky%s'%filter,'nsky%s'%filter,'grow_corr%s'%filter]]=phot[elno,2:-1].astype(float)
                            DF.mvs_targets_df.loc[sel,['%s_flag'%filter]]=phot[elno,-1:]
                except:
                    print(avg_ids,mvs_ids)
                    display(DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids])
                    sys.exit()
            phot_out=avg_aperture_photometry(DF,avg_ids,filter,goodness_phot_label,suffix,skip_flag)
            phot_out=np.array(phot_out)
            DF.avg_targets_df.loc[DF.avg_targets_df.avg_ids==phot_out[0],['m%s'%filter,'e%s'%filter,'spx%s'%filter,'bpx%s'%filter]]=phot_out[1:]
            DF.avg_candidates_df.loc[DF.avg_candidates_df.avg_ids==phot_out[0],'MagBin%s'%filter]=phot_out[1].astype(int)

#############################################
# Fake Injection dataframe related routines #
#############################################



def mk_completeness_from_fakes(DF,filter,Nvisit_range=None,magbin_list=None,AUC_lim=0.75,FP_lim=0.001,DF_fk=None,chunksize = None,skip_filters=[],parallel_runs=False,workers=None):
    '''
    Make completeness maps from fake injections.

    Parameters
    ----------
    Nvisit_range : list, or int
        If a list, make a completness map for each of these numbers of visitis.
            If None, take it from catalogue. The default is None.
    AUC_lim : float, optional
        minimum AUC to consider for detection. The default is 0.75.
    FP_lim : float, optional
        minimum false posive % accepted for detection. The default is 0.001.
    workers : int, optional
        number of workers to split the work accross multiple CPUs. The default is 3.
    DF_fk : pandas DataFrame, optional
        fake injection dataframe. If None, look in DF. The default is None.
    num_of_chunks : int, optional
        number of chunk to split the targets. The default is None.
    chunksize : int, optional
        size of each chunk for the parallelization process. The default is None.

    Returns
    -------
    None.

    '''
    if __name__ == 'utils_straklip':
        print('> ',filter)
        if magbin_list==None: magbin_list=DF.fk_candidates_df.loc[filter].index.get_level_values('magbin').unique()
        dmag_list=DF.fk_candidates_df.loc[filter].index.get_level_values('dmag').unique()
        sep_list=DF.fk_candidates_df.loc[filter].index.get_level_values('sep').unique()
        values_list=[]

        if parallel_runs:
            for Nvisit in Nvisit_range:
                workers,chunksize,ntarget=parallelization_package(workers,len(magbin_list),chunksize = chunksize)
                print('Testing nvisit %i. Working on magbin:'%Nvisit)
                FP_sel=FP_lim**(1/(Nvisit*len(DF.filters_list)))
                if DF_fk==None: DF_fk=DF
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    for out_list in tqdm(executor.map(task_completeness_from_fakes_infos,repeat(DF),repeat(filter),repeat(Nvisit),magbin_list,repeat(dmag_list),repeat(sep_list),repeat(FP_sel),repeat(AUC_lim),repeat(skip_filters),chunksize=chunksize)):
                        values_list.extend(out_list)

        else:
            for Nvisit in Nvisit_range:
                print('Testing nvisit %i. Working on magbin:'%Nvisit)
                FP_sel=FP_lim**(1/(Nvisit*len(DF.filters_list)))
                if DF_fk==None: DF_fk=DF
                for magbin in tqdm(magbin_list):
                    out_list=task_completeness_from_fakes_infos(DF,filter,Nvisit,magbin,dmag_list,sep_list,FP_sel,AUC_lim,skip_filters)
                    values_list.extend(out_list)

        print('Writing entry in dataframe:')
        for elno in tqdm(range(len(values_list))):
            DF.fk_completeness_df.loc[(filter,values_list[elno][0],values_list[elno][1],values_list[elno][2],values_list[elno][3]),['ratio_Kmode%s'%values_list[elno][-1]]]=values_list[elno][4]

def task_completeness_from_fakes_infos(DF,filter,Nvisit,magbin,dmag_list,sep_list,FP_sel,AUC_lim,skip_filters):
    '''
    parallelized task for the mk_completeness_from_fakes.

    Parameters
    ----------
    Nvisit : int
        number of visit index in the fake injection dataframe.
    magbin : int
        primary magnitude bin index in the fake injection dataframe.
    dmag_list : int
        companion delta magnitude (contrast) index in the fake injection dataframe.
    sep_list : float
        separation index between primary and companon in the fake injection dataframe.
    AUC_lim : float, optional
        minimum AUC to consider for detection. The default is 0.75.
    FP_lim : float, optional
        minimum false posive % accepted for detection. The default is 0.001.

    Returns
    -------
    None.

    '''
    out_list=[]
    for dmag in dmag_list:
        for sep in sep_list:
            for Kmode in DF.kmodes_list:
                if filter not in skip_filters:
                    TPnsigma_inj_list=DF.fk_candidates_df.loc[(filter,magbin,dmag,sep),['Nsigma_Kmode%i'%(Kmode)]].values.ravel()
                    FPnsigma_list=DF.fk_targets_df.loc[(filter,magbin,dmag,sep),['Nsigma_Kmode%i'%(Kmode)]].values.ravel()

                    X,Y,th=get_roc_curve(FPnsigma_list,TPnsigma_inj_list,nbins=10000)
                    X=np.insert(X,0,0)
                    Y=np.insert(Y,0,0)

                    w=min(np.where(abs(X-FP_sel)==min(abs(X-FP_sel)))[0])
                    AUC=metrics.auc(X, Y)
                    Ratio_list=len(TPnsigma_inj_list[TPnsigma_inj_list>=th[w]])/len(TPnsigma_inj_list)
                    # bins=np.arange(np.min(np.min(X),np.min(Y)),np.max(np.max(X),np.max(Y)),20)
                if AUC >= AUC_lim:
                    Ratio_median=round(np.nanmedian(Ratio_list),3)
                else:
                    Ratio_median=0
                out_list.append([Nvisit,magbin,dmag,sep,Ratio_median,Kmode])
                if np.any(np.isnan([Nvisit,magbin,dmag,sep,Ratio_median,Kmode])):
                    print(AUC,AUC_lim)
                    print([Nvisit,magbin,dmag,sep,Ratio_median,Kmode])
                    sys.exit()

    # sys.exit()
    return(out_list)

def task_fake_reference_infos(DF,elno,psf,magmin,magmax,zpt,exptime,bkg_list,ebkg_list,inner_shift):
    '''
    Taks perfomed in the update_fakes_df for reference stars

    Parameters
    ----------
    elno : int
        counter for the psf.
    psf : numpy array
        image of the PSF star.
    magmin : int
        minimum magnitude limit.
    magmax : int
        maximum magnitude limit.
    zpt : float
        zero point for photometry.
    exptime : float
        exposure time for photmetry.
    bg_list : list
        list of background values to sample.
    ebg_list : list
        list of uncertanties on the background values to sample.

    Returns
    -------
    None.

    '''
    idx = randint(0, len(bkg_list)-1)
    bkg=bkg_list[idx]
    ebkg=ebkg_list[idx]
    m3=round(uniform(magmin,magmax+1),2)
    shift3=[round(uniform(-inner_shift,inner_shift),2),round(uniform(-inner_shift,inner_shift),2)]
    c3=10**(-(m3-zpt)/2.5)
    PSF=Fake_Star(psf,c3*exptime,shift=shift3,Sky=bkg*exptime,eSky=ebkg*exptime)
    im3=np.array(PSF.star.tolist())/exptime
    if psf.shape[1]%2==0: x_cen=int((psf.shape[1])/2)
    else:  x_cen=int((psf.shape[1]-1)/2)

    if psf.shape[0]%2==0: y_cen=int((psf.shape[0])/2)
    else:  y_cen=int((psf.shape[0]-1)/2)


    DATA3=Tile(data=im3,x=x_cen,y=y_cen,tile_base=DF.tilebase,inst=DF.inst)
    DATA3.mk_tile(pad_data=False,verbose=False,xy_m=False,legend=False,showplot=False,keep_size=True,kill_plots=True,cbar=True)
    return(DATA3.data.tolist())

def update_fakes_df(DF,filter,parallel_runs=False,workers=None,NPSFstars=300,NPSFsample=30,inner_shift=0.25,path2psfdir=None,psf_filename='',showplot=False,aptype='4pixels',delta=1,use_median_sky=True,suffix=''):
    '''
    Update the fake dataframe.

    Parameters
    ----------
    # zpt_list_in : TYPE, optional
    #     zero point input list for photometry. The default is [].
    # exptime_list_in : TYPE, optional
    #     exposure time input list for photmetry. The default is [].
    workers : int, optional
        number of workers for parallelization. The default is None.
    NPSFstars : int, optional
        number of PSF to create. The default is 300.
    NPSFsample : int, optional
        number of PSF to select to creat the sample for the PSF libray. The default is 30.
    inner_shift : float, optional
        subpixel shift around satr injection coordinate. The default is 0.45.
    path2psfdir: str, optional
         default dir for psfs. If None, use default path. The default is None
    aptype : (circular,square,4pixels), optional
        defin the aperture type to use during aperture photometry.
        The default is '4pixels'.
    delta : int, optional
        step to create the square mask in range -delta, x, +delt and -delta, y, +delta. The default is 1.

    Returns
    -------
    None.

    '''

    if __name__ == 'utils_straklip':
        elno=np.where(filter in DF.filters_list)[0][0]
        if path2psfdir == None:
            x='mvs_tiles/%s'%filter
            # path2psfdir=PurePath(path2data/DF.project/DF.target/DF.inst/x)
            path2psfdir=PurePath(path2data+'/%s/%s/%s/%s'%(DF.project,DF.target,DF.inst,x))

            psf=[]
        else:
            hdul = fits.open(path2psfdir+psf_filename)
            psf=hdul[0].data
            hdul.close()

        print(path2psfdir)
        print('Collecting background values in %s imags'%filter)
        bkg_list=[]
        ebkg_list=[]
        nbkg_list=[]

        mvs_psf_ids_list=DF.mvs_targets_df.loc[DF.mvs_targets_df['%s_flag'%filter]=='good_psf'].mvs_ids.unique()
        if use_median_sky:
            for fitsfile in tqdm(DF.mvs_targets_df['%s_flc'%filter].unique()):
                fits_image_filename='%s\%s\%s\%s\%s'%(str(DF.path2fitsdir),DF.project,DF.target,DF.inst,fitsfile+'.fits')
                hdul = fits.open(fits_image_filename)
                exptime=hdul[0].header['exptime']
                for ext in [1,4]:
                    data = hdul[ext].data
                    data[data<0]=0
                    if not DF.header_df.loc['multiply_by_exptime','Values']:
                        data/=exptime
                    if DF.header_df.loc['multiply_by_gain','Values']:
                        data*=DF.gain

                    if DF.header_df.loc['multiply_by_PAM','Values']:
                        path2PAM='%s/%s/%s/%s/PAM'%(DF.path2fitsdir,DF.project,DF.target,DF.inst)
                        phdul=fits.open(path2PAM+'/'+str(DF.PAMdict[0][ext]+'.fits'))
                        try:PAM=phdul[1].data
                        except:PAM=phdul[0].data
                        data*=PAM

                sigma_clip = SigmaClip(sigma=3.)
                bkg_estimator = MedianBackground()
                bkg = Background2D(data, (10, 10), filter_size=(3, 3),
                        sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
                bkg_list.append(bkg.background_median)
                ebkg_list.append(bkg.background_rms_median)

            bkg_list=np.array(bkg_list)
            ebkg_list=np.array(ebkg_list)
            nbkg_list=np.array(nbkg_list)
            bgk_sel=~np.isnan(bkg_list)
            bkg_list=bkg_list[bgk_sel]
            ebkg_list=ebkg_list[bgk_sel]
        else:
            bkg_list=DF.mvs_targets_df['sky%s'%filter].values.astype(float)/DF.mvs_targets_df['exptime_%s'%filter].values.astype(float)
            ebkg_list=DF.mvs_targets_df['esky%s'%filter].values.astype(float)/DF.mvs_targets_df['exptime_%s'%filter].values.astype(float)
            nbkg_list=DF.mvs_targets_df['nsky%s'%filter].values.astype(float)

            bgk_sel=~np.isnan(bkg_list)
            bkg_list=bkg_list[bgk_sel]
            ebkg_list=ebkg_list[bgk_sel]
            nbkg_list=nbkg_list[bgk_sel]

        magmin=DF.mvs_targets_df.loc[(DF.mvs_targets_df['%s_flag'%filter]=='good_psf'),'m%s%s'%(filter,suffix)].min()#&(DF.mvs_targets_df['m%s%s'%(filter,suffix)]>0)
        magmax=DF.mvs_targets_df.loc[(DF.mvs_targets_df['%s_flag'%filter]=='good_psf'),'m%s%s'%(filter,suffix)].max()#&(DF.mvs_targets_df['m%s%s'%(filter,suffix)]>0)
        zpt=DF.ZPT_list[elno]
        multiply_by_exptime=DF.header_df.loc['multiply_by_exptime','Values']
        fk_ids_list=[i for i in range(0,NPSFstars)]
        print('Working on %s, generating the fake PSF library stars'%filter)
        psf_list=[]
        psf_ids_list=[]
        for fk_ids in tqdm(fk_ids_list):
            ID=int(choice(mvs_psf_ids_list))
            exptime=DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==ID,'exptime_%s'%filter].values[0]
            if len(psf) == 0:
                with fits.open(str(path2psfdir)+'/tile_ID%s.fits'%ID,memmap=False) as hdul:
                    if multiply_by_exptime:target_psf=hdul[-1].data/exptime
                    else:target_psf=hdul[-1].data
                    psf_ids_list.append(ID)
            else:
                target_psf=psf.copy()

            psf_list.append(task_fake_reference_infos(DF,fk_ids,target_psf,magmin,magmax,zpt,exptime,bkg_list,ebkg_list,inner_shift))

        magbin_list=DF.fk_targets_df.loc[filter].index.get_level_values('magbin').unique().astype(float)
        dmag_list=DF.fk_targets_df.loc[filter].index.get_level_values('dmag').unique().astype(float)
        sep_list=DF.fk_targets_df.loc[filter].index.get_level_values('sep').unique().astype(float)
        columns_no_injection=['x','y','m','exptime']+np.array([['Nsigma_Kmode%s'%(Kmode),'counts_Kmode%s'%(Kmode),'noise_Kmode%s'%(Kmode),'m_Kmode%s'%(Kmode)] for Kmode in DF.kmodes_list]).ravel().tolist()
        columns=['x','y','counts','m','exptime']+np.array([['Nsigma_Kmode%s'%(Kmode),'counts_Kmode%s'%(Kmode),'noise_Kmode%s'%(Kmode),'m_Kmode%s'%(Kmode)] for Kmode in DF.kmodes_list]).ravel().tolist()

        print('Working on %s, generating fake binaries for each magbin'%(filter))
        for magbin in tqdm(magbin_list):
            for dmag in dmag_list:
                if parallel_runs:
                    workers,chunksize,ntarget=parallelization_package(workers,len(sep_list),verbose=False)
                    with ProcessPoolExecutor(max_workers=workers) as executor:
                        for out,out_no_injection in executor.map(task_fake_infos,repeat(DF),repeat(magbin),repeat(dmag),sep_list,repeat(filter),repeat(zpt),repeat(psf),repeat(psf_list),repeat(psf_ids_list),repeat(NPSFsample),repeat(bkg_list),repeat(ebkg_list),repeat(nbkg_list),repeat(inner_shift),repeat(path2psfdir),repeat(multiply_by_exptime),repeat(showplot),repeat(aptype),repeat(delta),chunksize=chunksize):
                            fk_writing(DF,filter,out_no_injection,'fk_targets_df',columns_no_injection)
                            fk_writing(DF,filter,out,'fk_candidates_df',columns)
                            del out,out_no_injection
                else:
                    for sep in sep_list:
                        out,out_no_injection=task_fake_infos(DF,magbin,dmag,sep,filter,zpt,psf,psf_list,psf_ids_list,NPSFsample,bkg_list,ebkg_list,nbkg_list,inner_shift,path2psfdir,multiply_by_exptime,showplot,aptype,delta)
                        fk_writing(DF,filter,out_no_injection,'fk_targets_df',columns_no_injection)
                        fk_writing(DF,filter,out,'fk_candidates_df',columns)
                        del out,out_no_injection
        del psf_list,psf_ids_list,bkg_list,ebkg_list,psf

def psf_scale(psfdata):
    psfdata[psfdata<0]=0
    # psfdata+=(1-np.sum(psfdata))/(psfdata.shape[1]*psfdata.shape[0])
    psfdata/=np.sum(psfdata)
    return(psfdata)
        
def task_fake_infos(DF,magbin,dmag,sep,filter,zpt,psf,psf_list,psf_ids_list,npsfs,bkg_list,ebkg_list,nbkg_list,inner_shift,path2psfdir,multiply_by_exptime,showplot,aptype,delta):
    '''
    Taks perfomed in the update_fakes_df for target stars

    Parameters
    ----------
    magbin : int
        magnitude bins to analyze.
    filter : str
        filter name.
    psf : numpy array
        image of the PSF star.
    zpt : float
        zero point for photometry.
    exptime : float
        exposure time for photmetry.
    bkg_list : list
        list of background values to sample.
    ebkg_list : list
        list of uncertanties on the background values to sample.
    aptype : (circular,square,4pixels), optional
        defin the aperture type to use during aperture photometry. 
        The default is '4pixels'.
    delta : int, optional
        step to create the square mask in range -delta, x, +delt and -delta, y, +delta. The default is 1.

    Returns
    -------
    None.

    '''
 
    out_list=[]
    out_no_injection_list=[]
    
    for elno in DF.fk_targets_df.index.get_level_values('fk_ids').unique():
        idx = randint(0, len(bkg_list)-1)
        ID=psf_ids_list[idx]
        bkg=bkg_list[idx]
        ebkg=ebkg_list[idx]

        exptime=DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==ID,'exptime_%s'%filter].values[0]
        if len(psf) == 0: 
            with fits.open(str(path2psfdir)+'/tile_ID%s.fits'%ID,memmap=False) as hdul:  
                if multiply_by_exptime:target_psf=hdul[-1].data/exptime
                else:target_psf=hdul[-1].data
        else: 
            target_psf=psf.copy()

        m1=round(float(magbin),2)#uniform(magbin,magbin+1)
        shift1=[round(uniform(-inner_shift,inner_shift),2),round(uniform(-inner_shift,inner_shift),2)]
        c1=10**(-(m1-zpt)/2.5)
        
        contrast=round(float(dmag),2)#round(uniform(dmag,dmag+1),2)
        m2=m1+contrast
        c2=10**(-(m2-zpt)/2.5)
        
        r=sep#round(uniform(sep,sep+1),2)
        x,y=PointsInCircum(r)
        shift2=[round(uniform(-inner_shift,inner_shift),2),round(uniform(-inner_shift,inner_shift),2)]
        if round2closerint([x+int((DF.tilebase-1)/2)+shift2[1]])[0] < 0 or round2closerint([y+int((DF.tilebase-1)/2)+shift2[0]])[0] < 0 or round2closerint([x+int((DF.tilebase-1)/2)+shift2[1]])[0] >= DF.tilebase or round2closerint([y+int((DF.tilebase-1)/2)+shift2[0]])[0] >= DF.tilebase:
            cc=0
            while round2closerint([x+int((DF.tilebase-1)/2)+shift2[1]])[0] < 0 or round2closerint([y+int((DF.tilebase-1)/2)+shift2[0]])[0] < 0 or round2closerint([x+int((DF.tilebase-1)/2)+shift2[1]])[0] >= DF.tilebase or round2closerint([y+int((DF.tilebase-1)/2)+shift2[0]])[0] >= DF.tilebase:
                x,y=PointsInCircum(r)
                if cc >=1000: raise ValueError('Positions %i,%i (generated by a separation of %.2f), are outside the tile box 0,%i. Please choose smaller separations'%(round2closerint([x+int((DF.tilebase-1)/2)+shift2[1]])[0],round2closerint([y+int((DF.tilebase-1)/2)+shift2[0]])[0],r,DF.tilebase-1))
                cc+=1

        if target_psf.shape[1]%2==0: x_cen=int((target_psf.shape[1])/2)
        else:  x_cen=int((target_psf.shape[1]-1)/2)
        
        if target_psf.shape[0]%2==0: y_cen=int((target_psf.shape[0])/2)
        else:  y_cen=int((target_psf.shape[0]-1)/2)
        target_psf=psf_scale(target_psf)
        if showplot:
            try:
                radius_in=DF.header_df.loc['radius_in','Values']
                radius1=DF.header_df.loc['radius1_in','Values']
                radius2=DF.header_df.loc['radius2_in','Values']
            except:
                radius_in=5
                radius1=10
                radius2=15
                
            gain=DF.header_df.loc['gain','Values']

            print('> PSF, ID%i:'%ID)
            aperture_photometry_handler(DF,0,filter,x=int((DF.tilebase-1)/2),y=int((DF.tilebase-1)/2),data=target_psf,zpt=0,ezpt=0,aptype='circular',noBGsub=True,sigma=3,kill_plots=False,Python_origin=True,delta=3,exptime=1,radius1=radius1,radius2=radius2,gain=gain)
        
        S0=Fake_Star(target_psf,c1*exptime,shift=shift1,Sky=bkg*exptime,eSky=ebkg*exptime)
        im0=np.array(S0.star.tolist())
        S1=Fake_Star(target_psf,c1*exptime,shift=shift1,Sky=bkg*exptime,eSky=ebkg*exptime,PNoise=False)
        S2=Fake_Star(target_psf,c2*exptime,shift=shift2,PNoise=False)
        im2=np.array(S2.star.tolist())
        pos=[y,x]
        S1.combine(S2.star,pos)
        im12=np.array(S1.binary.tolist())

        DATA0=Tile(data=im0,x=x_cen,y=y_cen,tile_base=DF.tilebase,inst=DF.inst)
        DATA0.mk_tile(pad_data=False,verbose=False,xy_m=False,legend=False,showplot=False,keep_size=False,kill_plots=True,cbar=True,title='S0 ,m %i'%(m1))

        DATA12=Tile(data=im12,x=x_cen,y=y_cen,tile_base=DF.tilebase,inst=DF.inst)
        DATA12.mk_tile(pad_data=False,verbose=False,xy_m=False,legend=False,showplot=False,keep_size=False,kill_plots=True,cbar=True,title='Bin ,magbin %i,dmag %i, sep %.1f'%(magbin,dmag,sep))
        pos=[y+int((DF.tilebase-1)/2)+shift2[0],x+int((DF.tilebase-1)/2)+shift2[1]]
        y,x=round2closerint(pos)
        
        dt=Detection(im2,x_cen,y_cen)
        photometry_AP.aperture_mask(dt,aptype=aptype,ap_x=delta,ap_y=delta)
        photometry_AP.mask_aperture_data(dt)
        photometry_AP.aperture_stats(dt,aperture=dt.aperture,sigma=3)
        flux_converter.counts_and_errors(dt)
        
        if showplot:

            print('> Isolated primary with Sky:'.upper())
            aperture_photometry_handler(DF,0,filter,x=x_cen,y=y_cen,data=im0,zpt=zpt,ezpt=0,aptype='circular',noBGsub=False,sigma=3,kill_plots=False,Python_origin=True,exptime=exptime,radius_a=radius_in,radius1=radius1,radius2=radius2,gain=gain)
            print('> Isolated companion without Sky:'.upper())
            print('> Input counts: %e'%(c2*exptime))
            aperture_photometry_handler(DF,0,filter,x=x_cen,y=y_cen,data=im2,zpt=zpt,ezpt=0,aptype='circular',noBGsub=True,sigma=3,kill_plots=False,Python_origin=True,exptime=exptime,radius_a=radius_in,radius1=radius1,radius2=radius2,gain=gain)
            print('> %s aperture on the same target:'%aptype)
            aperture_photometry_handler(DF,0,filter,x=x_cen,y=y_cen,data=im2,zpt=zpt,ezpt=0,aptype=aptype,noBGsub=True,sigma=3,kill_plots=False,Python_origin=True,exptime=exptime,radius_a=radius_in,radius1=radius1,radius2=radius2,gain=gain,delta=delta)
            print('> Expected counts: %e'%dt.counts)

            print('> Binary with the two component combined:'.upper())
            print('> Centered on the primary:')
            aperture_photometry_handler(DF,0,filter,x=x_cen,y=y_cen,data=im12,zpt=zpt,ezpt=0,aptype='circular',noBGsub=False,sigma=3,kill_plots=False,Python_origin=True,exptime=exptime,radius_a=radius_in,radius1=radius1,radius2=radius2,gain=gain)
            print('> Centered on the companion:')

        out_0_no_injection=[magbin,dmag,sep,elno,int((DF.tilebase-1)/2)+shift1[1],int((DF.tilebase-1)/2)+shift1[0],m1,exptime]
        out_0=[magbin,dmag,sep,elno,x+int((DF.tilebase-1)/2)+shift2[1],y+int((DF.tilebase-1)/2)+shift2[0],dt.counts,m2,exptime]
        try:
            if showplot:
                print('>. PSF subtraction with no injection')
            out_no_injection=perform_KLIP_PSF_subtraction_on_fakes(DF,filter,DATA0.data,psf_list,pos,DF.kmodes_list,npsfs,showplot,exptime,aptype,delta,noBGsub=False)
            if showplot:
                print('>. PSF subtraction with injection')
            out=perform_KLIP_PSF_subtraction_on_fakes(DF,filter,DATA12.data,psf_list,pos,DF.kmodes_list,npsfs,showplot,exptime,aptype,delta,noBGsub=False)
        except:
            raise ValueError([magbin,dmag,sep])
        out_0_no_injection.extend(out_no_injection)
        out_0.extend(out)        
        out_no_injection_list.append(out_0_no_injection)
        out_list.append(out_0)
        del S0,S1,S2,DATA0,DATA12   
    return(out_list,out_no_injection_list)
############################################
# False Positive Analysis related routines #
############################################

def perform_FP_analysis(DF,avg_ids_list=[],AUC_lim=0.5,showplot=False,verbose=False,DF_fk=None,skip_filters=[],nbins=10,suffix=''):
    '''
    Perform False Positive anlysis using the fake injection dataframe and update the catalogs

    Parameters
    ----------
    avg_ids_list : TYPE, optional
        average ids list of targets. If empty, look in DF. The default is [].
    showplot : bool, optional
        choose to show plots. The default is False.
    verbose : bool, optional
        choose to show prints. The default is False.
    DF_fk : pandas DataFrame, optional
        fake injection dataframe. If None, look in DF. The default is None.

    Returns
    -------
    None.

    '''
    DF.avg_candidates_df[['FPa_flag%s'%filter for filter in DF.filters_list]]='rejected'    
    if len(avg_ids_list)==0: avg_ids_list=DF.avg_candidates_df.avg_ids.unique()
    print('Performing FP analysis on %s candidates'%len(avg_ids_list))
    for avg_ids in tqdm(avg_ids_list):
        for filter in DF.filters_list:
            mvs_ids_list=DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.avg_ids.isin([avg_ids])].mvs_ids.unique()
            if filter not in skip_filters:
                out=FP_analysis(DF,avg_ids,filter,AUC_lim,showplot=showplot,DF_fk=DF_fk,nbins=nbins,suffix=suffix)
                DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids.isin(mvs_ids_list),['%s_th'%filter,'%s_TP_above_th'%filter,'%s_TP_above_Nsigma'%filter,'%s_FP_above_th'%filter,'%s_FP_above_Nsigma'%filter,'%s_AUC'%filter,'%s_FPa_flag'%filter]]=out

                sel=(DF.mvs_candidates_df.mvs_ids.isin(mvs_ids_list)&(DF.mvs_candidates_df['%s_FPa_flag'%filter]=='accepted'))
                DF.avg_candidates_df.loc[DF.avg_candidates_df.avg_ids==avg_ids,['th%s'%filter,'TP_above_th%s'%filter,'TP_above_Nsigma%s'%filter,'FP_above_th%s'%filter,'FP_above_Nsigma%s'%filter,'AUC%s'%filter]]=DF.mvs_candidates_df.loc[sel,['%s_th'%filter,'%s_TP_above_th'%filter,'%s_TP_above_Nsigma'%filter,'%s_FP_above_th'%filter,'%s_FP_above_Nsigma'%filter,'%s_AUC'%filter]].astype(float).mean(axis=0).values    
                if DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids.isin(mvs_ids_list),['%s_FPa_flag'%filter]].apply(lambda x: x.str.contains('accepted',case=False)).any(axis=1).any(axis=0):
                    DF.avg_candidates_df.loc[DF.avg_candidates_df.avg_ids==avg_ids,'FPa_flag%s'%filter]='accepted' 

        if verbose:
            display(DF.avg_candidates_df.loc[DF.avg_candidates_df.avg_ids==avg_ids])
            display(DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids.isin(mvs_ids_list)])
        if showplot: plt.show()
        
        




