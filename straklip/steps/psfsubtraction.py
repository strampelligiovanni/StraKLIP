from stralog import getLogger
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tiles import Tile
import numpy.ma as ma
from astropy.io import fits
import pandas as pd
from utils_tile import perform_PSF_subtraction
from itertools import repeat
from ancillary import parallelization_package

def task_perform_KLIP_PSF_subtraction_on_tiles(DF,filter,cell,mvs_ids_list,label_dict,hdul_dict,KLIP_label_dict,skip_flags,label,kmodes,overwrite):
    if len(mvs_ids_list)==0:
        ids_list=DF.mvs_targets_df.loc[(DF.mvs_targets_df['cell_%s'%filter]==cell)&~DF.mvs_targets_df['flag_%s'%filter].isin([skip_flags])].mvs_ids.unique()
    else:
        ids_list=mvs_ids_list

    psf_ids_list=DF.mvs_targets_df.loc[DF.mvs_targets_df['flag_%s'%filter].str.contains('psf')&(DF.mvs_targets_df['cell_%s'%filter]==cell)].mvs_ids.unique()
    for id in ids_list:
        if not DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==id,'flag_%s'%filter].str.contains('rejected').values[0]:
            path2tile='%s/mvs_tiles/%s/tile_ID%i.fits'%(DF.path2out,filter,id)
            if not overwrite:
                try:
                    with fits.open(path2tile) as hdul:
                        for Kmode in kmodes:
                            hdul[f'{KLIP_label_dict[label].upper()}{Kmode}']
                    go4PSFsub=False
                except:
                    go4PSFsub=True
            else:
                go4PSFsub=True
            if go4PSFsub:
                getLogger(__name__).info(f'Working on tile: {path2tile}.')
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
                        path2ref = '%s/mvs_tiles/%s/tile_ID%i.fits' % (DF.path2out, filter, refid)
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

                    residuals,psf_models=perform_PSF_subtraction(targ_tiles['data'],ref_tiles['data'],kmodes=kmodes)
                    return_Datacube=True
                    for Kmode in residuals.columns: # loop over the residuals of the different kmodes
                        KLIP=Tile(data=residuals[Kmode].tolist()[0],x=(DF.tilebase-1)/2,y=(DF.tilebase-1)/2,tile_base=DF.tilebase,delta=0,inst=DF.inst,Python_origin=True)
                        KLIP.mk_tile(pad_data=False,legend=False,showplot=False,verbose=False,title='Kmode%i'%Kmode,kill_plots=True)
                        Datacube=KLIP.append_tile(path2tile,Datacube=Datacube,verbose=False,name='%s%i'%(KLIP_label_dict[label],Kmode),return_Datacube=return_Datacube,write=False)
                    for model in psf_models.columns:
                        if model == psf_models.columns[-1]:return_Datacube=False
                        PSF=Tile(data=psf_models[model].tolist()[0],x=(DF.tilebase-1)/2,y=(DF.tilebase-1)/2,tile_base=DF.tilebase,delta=0,inst=DF.inst,Python_origin=True)
                        PSF.mk_tile(pad_data=False,legend=False,showplot=False,verbose=False,title='Model%s'%model,kill_plots=True)
                        Datacube=PSF.append_tile(path2tile,Datacube=Datacube,verbose=False,name='Model%s'%model,return_Datacube=return_Datacube,write=False)
            else:
                getLogger(__name__).info(f'overwrite {overwrite} and all the kmodes {kmodes} layers are already saved in tile: {path2tile}. Skipping.')


def perform_KLIP_PSF_subtraction_on_tiles(DF,filter,label,workers=None,parallel_runs=True,mvs_ids_list=[],kmodes=[],skip_flags=['rejected','known_double'],overwrite=False,chunksize = None):
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
    kmodes : TYPE, optional
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
            for _ in executor.map(task_perform_KLIP_PSF_subtraction_on_tiles,repeat(DF),repeat(filter),cell_list,repeat(mvs_ids_list),repeat(label_dict),repeat(hdul_dict),repeat(KLIP_label_dict),repeat(skip_flags),repeat(label),repeat(kmodes),repeat(overwrite),chunksize=chunksize):
                pass

    else:
        for cell in cell_list:
            task_perform_KLIP_PSF_subtraction_on_tiles(DF,filter,cell,mvs_ids_list,label_dict,hdul_dict,KLIP_label_dict,skip_flags,label,kmodes,overwrite)

def perform_KLIP_PSF_subtraction_on_tiles(DF,filter,label,workers=None,parallel_runs=True,mvs_ids_list=[],kmodes=[],skip_flags=['rejected','known_double'],overwrite=False,chunksize = None):
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
    kmodes : TYPE, optional
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
            for _ in executor.map(task_perform_KLIP_PSF_subtraction_on_tiles,repeat(DF),repeat(filter),cell_list,repeat(mvs_ids_list),repeat(label_dict),repeat(hdul_dict),repeat(KLIP_label_dict),repeat(skip_flags),repeat(label),repeat(kmodes),repeat(overwrite),chunksize=chunksize):
                pass

    else:
        for cell in cell_list:
            task_perform_KLIP_PSF_subtraction_on_tiles(DF,filter,cell,mvs_ids_list,label_dict,hdul_dict,KLIP_label_dict,skip_flags,label,kmodes,overwrite)


def KLIP_PSF_subtraction(DF, filter, label, mvs_ids_list=[], kmodes=[], workers=None, parallel_runs=True,
                         skip_flags=['rejected', 'known_double'],overwrite=False, chunksize=None):
    '''
    This is a wrapper for KLIP PSF subtraction step

    '''
    perform_KLIP_PSF_subtraction_on_tiles(DF, filter, label,
                                          workers=workers,
                                          parallel_runs=parallel_runs,
                                          mvs_ids_list=mvs_ids_list,
                                          kmodes=kmodes,
                                          skip_flags=skip_flags,
                                          chunksize=chunksize,
                                          overwrite=overwrite)

def run(packet):
    dataset = packet['dataset']
    DF = packet['DF']
    if dataset.pipe_cfg.mktiles['cr_remove'] or dataset.pipe_cfg.mktiles['la_cr_remove']:
        label='crclean_data'
        getLogger(__name__).info(f'Performing KLIP PSF subtraction on CR cleaned tiles.')
    else:
        label='data'
        getLogger(__name__).info(f'Performing KLIP PSF subtraction on tiles.')

    for filter in dataset.data_cfg.filters:
        KLIP_PSF_subtraction(DF, filter,
                             label=label,
                             mvs_ids_list=[],
                             kmodes=dataset.pipe_cfg.psfsubtraction['kmodes'],
                             workers=dataset.pipe_cfg.ncpu,
                             parallel_runs=dataset.pipe_cfg.psfsubtraction['parallel_runs'],
                             skip_flags=dataset.pipe_cfg.psfsubtraction['skip_flags'],
                             overwrite=dataset.pipe_cfg.psfsubtraction['overwrite'])
    DF.save_dataframes(__name__)
