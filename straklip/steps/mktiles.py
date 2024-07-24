import os, sys
from utils_tile import small_tiles
from ancillary import parallelization_package
from utils_tile import allign_images
from tiles import Tile
from straklip import config
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from astropy.io import fits
from itertools import repeat
from IPython.display import display
from matplotlib.colors import PowerNorm,ListedColormap
from collections import Counter
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
    for cell in np.sort(DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(mvs_ids_list)][f'cell_{filter}'].unique()):
        skip_flags=['rejected','known_double']
        list_of_target4test=[]
        ids_list=DF.mvs_targets_df.loc[(DF.mvs_targets_df[f'cell_{filter}']==cell)&~DF.mvs_targets_df[f'flag_{filter}'].isin([skip_flags])].mvs_ids.unique()

        for id in ids_list:
            if not DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==id,f'flag_{filter}'].str.contains('rejected').values[0]:
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
                    display(DF.unq_targets_df.loc[(DF.unq_targets_df.unq_ids.isin(DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.mvs_ids.isin([np.array(ids_list)[k]])].unq_ids.unique()))|(DF.unq_targets_df.unq_ids.isin(DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.mvs_ids.isin([np.array(ids_list)[q]])].unq_ids.unique()))])
                    display(DF.mvs_targets_df.loc[(DF.mvs_targets_df.mvs_ids.isin([np.array(ids_list)[k]]))|(DF.mvs_tiles_df.mvs_ids.isin([np.array(ids_list)[q]]))])
                ids_skip=(DF.mvs_targets_df.loc[(DF.mvs_targets_df.mvs_ids.isin([np.array(ids_list)[k]]))|(DF.mvs_tiles_df.mvs_ids.isin([np.array(ids_list)[q]]))].mvs_ids.unique())
                DF.crossmatch_ids_df=DF.crossmatch_ids_df.loc[~DF.crossmatch_ids_df.mvs_ids.isin(ids_skip)]
    DF.unq_targets_df=DF.unq_targets_df.loc[DF.unq_targets_df.unq_ids.isin(DF.crossmatch_ids_df.unq_ids.unique())]
    DF.mvs_targets_df=DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(DF.crossmatch_ids_df.mvs_ids.unique())]
    print('All duplicants killed')

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

def task_mvs_tiles(DF,fitsname,ids_list,filter,use_xy_SN,use_xy_m,use_xy_cen,xy_shift_list,xy_dmax,legend,verbose,Python_origin,cr_remove,la_cr_remove,cr_radius,multiply_by_exptime,multiply_by_gain,multiply_by_PAM,overwrite):
    '''
    parallelized task for the update_mvs_tiles.
    '''
    out=[]
    delta=6
    path2fits=DF.path2data+'/'+fitsname+DF.fitsext+'.fits'
    hdul = fits.open(path2fits)
    if len(hdul)>=4:
        for id in ids_list:
            path2tile='%s/mvs_tiles/%s/tile_ID%i.fits'%(DF.path2out,filter,id)
            if not os.path.exists(path2tile) or overwrite:
                type_flag=DF.unq_targets_df.loc[DF.unq_targets_df.unq_ids==DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.mvs_ids==id].unq_ids.unique()[0]].type.values[0]
                x=DF.mvs_targets_df.loc[(DF.mvs_targets_df['fits_%s'%(filter)]==fitsname)&(DF.mvs_targets_df.mvs_ids==id),'x_%s'%filter].values[0]
                y=DF.mvs_targets_df.loc[(DF.mvs_targets_df['fits_%s'%(filter)]==fitsname)&(DF.mvs_targets_df.mvs_ids==id),'y_%s'%filter].values[0]
                ext=int(DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==id].ext.values[0])
                if x % 0.5==0: x-=0.001
                if y % 0.5==0: y-=0.001

                if not np.isnan(x) and not np.isnan(y):
                    getLogger(__name__).info(f'Making mvs tile {path2tile}')
                    exptime=DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==id,'exptime_%s'%filter].values[0]
                    SCI=hdul[ext].data.copy()
                    if multiply_by_exptime:
                        SCI*=exptime

                    if multiply_by_gain:
                        SCI*=DF.gain

                    if multiply_by_PAM:
                        phdul=fits.open(DF.path2pam+'/'+str(DF.PAMdict[ext]+'.fits'))
                        try:PAM=phdul[1].data
                        except:PAM=phdul[0].data
                        SCI*=PAM

                    ERR=hdul[ext+1].data
                    DQ=hdul[ext+2].data

                    if type_flag==2 and (use_xy_cen==True or use_xy_SN==True):
                        # flag='unresolved_double'
                        xy_m=True
                        xy_cen=False
                    else:
                        xy_cen=use_xy_cen
                        xy_m=use_xy_m

                    if xy_cen:
                        if cr_remove or la_cr_remove:
                            DQDATA=Tile(data=DQ,x=x,y=y,tile_base=DF.tilebase,delta=delta,inst=DF.inst,Python_origin=Python_origin)
                            DQDATA.mk_tile(pad_data=True,legend=legend,showplot=False,verbose=False,title='shiftedDQ',kill_plots=True,cbar=True)

                            IDATA=Tile(data=SCI,x=x,y=y,tile_base=DF.tilebase,delta=delta,dqdata=DQDATA.data,inst=DF.inst,Python_origin=Python_origin)
                            IDATA.mk_tile(xy_m=True,xy_dmax=xy_dmax,pad_data=True,legend=legend,showplot=False,verbose=False,title='CRcleanSCI',kill_plots=True,cr_remove=cr_remove, la_cr_remove=la_cr_remove,cr_radius=cr_radius,cbar=True)
                        else:

                            IDATA=Tile(data=SCI,x=x,y=y,tile_base=DF.tilebase,delta=delta,inst=DF.inst,Python_origin=Python_origin)
                            IDATA.mk_tile(pad_data=True,legend=False,showplot=False,verbose=False,xy_cen=True,xy_dmax=xy_dmax,title='OrigSCI',kill_plots=True,cbar=True)

                        deltax=IDATA.x_cen-(IDATA.tile_base-1)/2
                        deltay=IDATA.y_cen-(IDATA.tile_base-1)/2

                    elif xy_m:
                        if cr_remove or la_cr_remove:
                            DQDATA=Tile(data=DQ,x=x,y=y,tile_base=DF.tilebase,delta=delta,inst=DF.inst,Python_origin=Python_origin)
                            DQDATA.mk_tile(pad_data=True,legend=legend,showplot=False,verbose=False,title='shiftedDQ',kill_plots=True,cbar=True)
                            IDATA=Tile(data=SCI,x=x,y=y,tile_base=DF.tilebase,delta=delta,dqdata=DQDATA.data,inst=DF.inst,Python_origin=Python_origin)
                            IDATA.mk_tile(xy_m=True,xy_dmax=xy_dmax,pad_data=True,legend=legend,showplot=False,verbose=False,title='CRcleanSCI',kill_plots=True,cr_remove=cr_remove, la_cr_remove=la_cr_remove,cr_radius=cr_radius,cbar=True)
                        else:
                            IDATA=Tile(data=SCI,x=x,y=y,tile_base=DF.tilebase,delta=delta,inst=DF.inst,Python_origin=Python_origin)
                            IDATA.mk_tile(pad_data=True,legend=False,showplot=False,verbose=False,xy_m=True,xy_dmax=xy_dmax,title='OrigSCI',kill_plots=True,cbar=True)

                        deltax=IDATA.x_m-(IDATA.tile_base-1)/2
                        deltay=IDATA.y_m-(IDATA.tile_base-1)/2
                    else:
                        if cr_remove or la_cr_remove:
                            DQDATA=Tile(data=DQ,x=x,y=y,tile_base=DF.tilebase,delta=delta,inst=DF.inst,Python_origin=Python_origin)
                            DQDATA.mk_tile(pad_data=True,legend=legend,showplot=False,verbose=verbose,title='shiftedDQ',kill_plots=True,cbar=True)

                            IDATA=Tile(data=SCI,x=x,y=y,tile_base=DF.tilebase,delta=delta,dqdata=DQDATA.data,inst=DF.inst,Python_origin=Python_origin)
                            IDATA.mk_tile(xy_m=True,xy_dmax=xy_dmax,pad_data=True,legend=legend,showplot=False,verbose=False,title='CRcleanSCI',kill_plots=True,cr_remove=cr_remove, la_cr_remove=la_cr_remove,cr_radius=cr_radius,cbar=True)
                        else:

                            IDATA=Tile(data=SCI,x=x,y=y,tile_base=DF.tilebase,delta=delta,inst=DF.inst,Python_origin=Python_origin)
                            IDATA.mk_tile(pad_data=True,xy_dmax=xy_dmax,legend=False,showplot=False,verbose=False,title='OrigSCI',kill_plots=True,cbar=True)
                        deltax=0
                        deltay=0


                    if len(xy_shift_list)>0:
                        deltax+=xy_shift_list[0]
                        deltay+=xy_shift_list[1]

                    x,y=[x + deltax, y + deltay]

                    DATA=Tile(data=SCI,x=x,y=y,tile_base=DF.tilebase,delta=delta,inst=DF.inst,Python_origin=Python_origin)
                    DATA.mk_tile(pad_data=True,legend=legend,showplot=verbose,verbose=verbose,title=f'shiftedSCI {id}',kill_plots=not verbose,cbar=True)
                    Datacube=DATA.append_tile(path2tile,Datacube=None,verbose=False,name='SCI',return_Datacube=True)

                    EDATA=Tile(data=ERR,x=x,y=y,tile_base=DF.tilebase,delta=delta,inst=DF.inst,Python_origin=Python_origin)
                    EDATA.mk_tile(pad_data=True,legend=legend,showplot=False,verbose=verbose,title='shiftedERR',kill_plots=True,cbar=True)
                    Datacube=EDATA.append_tile(path2tile,Datacube=Datacube,verbose=False,name='ERR',return_Datacube=True)

                    return_Datacube=False
                    if cr_remove or la_cr_remove: return_Datacube=True
                    DQDATA=Tile(data=DQ,x=x,y=y,tile_base=DF.tilebase,delta=delta,inst=DF.inst,Python_origin=Python_origin)
                    DQDATA.mk_tile(pad_data=True,legend=legend,showplot=False,verbose=verbose,title='shiftedDQ',kill_plots=True,cbar=True)
                    Datacube=DQDATA.append_tile(path2tile,Datacube=Datacube,verbose=False,name='DQ',return_Datacube=return_Datacube)

                    if cr_remove or la_cr_remove:
                        CRDATA=Tile(data=SCI,x=x,y=y,tile_base=DF.tilebase,delta=delta,dqdata=DQDATA.data,inst=DF.inst,Python_origin=Python_origin)
                        CRDATA.mk_tile(pad_data=True,legend=legend,showplot=verbose,verbose=verbose,title=f'shiftedCRcleanSCI {id}',kill_plots=not verbose,cr_remove=cr_remove, la_cr_remove=la_cr_remove,cr_radius=cr_radius,cbar=True)
                        Datacube=CRDATA.append_tile(path2tile,Datacube=Datacube,verbose=False,name='CRcleanSCI',return_Datacube=False)
                    # return([id,x,y])
                    out.append([id,x,y])
                else:
                    getLogger(__name__).warning(f'x/y coordinates for MVS Tile {path2tile} are nan. Skipping.')
                    # return([id,np.nan,np.nan])
                    out.append([id,np.nan,np.nan])
            else:
                getLogger(__name__).info(f'MVS Tile {path2tile} already exist. Skipping.')
                # return ([])
                return None
    out=np.array(out)
    return(out)


def mk_mvs_tiles(DF,filter,mvs_ids_test_list=[],xy_SN=True,xy_m=False,xy_cen=False,xy_shift_list=[],xy_dmax=3,legend=False,verbose=False,workers=None,Python_origin=True,parallel_runs=True,cr_remove=False,la_cr_remove=False,cr_radius=3,chunksize = None,multiply_by_exptime=False,multiply_by_gain=False,multiply_by_PAM=False,overwrite=False):
    '''
    update the multi-visits tile dataframe with the tiles for each source

    '''
    getLogger(__name__).info(f'Working on multi-visits tiles on filter {filter}')
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
            for out in executor.map(task_mvs_tiles, repeat(DF), fitsname_list, ids_list_of_lists, repeat(filter),
                                  repeat(xy_SN), repeat(xy_m), repeat(xy_cen), repeat(xy_shift_list),
                                  repeat(xy_dmax), repeat(legend),
                                  repeat(verbose), repeat(Python_origin),
                                  repeat(cr_remove), repeat(la_cr_remove), repeat(cr_radius),
                                  repeat(multiply_by_exptime), repeat(multiply_by_gain),
                                  repeat(multiply_by_PAM),repeat(overwrite),
                                  chunksize=chunksize):
                # if len(out)>0 and out is not None:
                if out is not None:
                    DF.mvs_targets_df.loc[(DF.mvs_targets_df.mvs_ids.isin(out[:,0])), ['x_%s' % filter]] = out[:,1]
                    DF.mvs_targets_df.loc[(DF.mvs_targets_df.mvs_ids.isin(out[:,0])), ['y_%s' % filter]] = out[:,2]

    else:
        for elno in range(len(fitsname_list)):
            out = task_mvs_tiles(DF, fitsname_list[elno], ids_list_of_lists[elno], filter, xy_SN, xy_m, xy_cen, xy_shift_list, xy_dmax,
                           legend, verbose, Python_origin, cr_remove, la_cr_remove, cr_radius,
                           multiply_by_exptime, multiply_by_gain, multiply_by_PAM,overwrite)
            # if len(out) > 0 and out is not None:
            if out is not None:
                DF.mvs_targets_df.loc[(DF.mvs_targets_df.mvs_ids.isin(out[:,0])), ['x_%s' % filter]] = out[:,1]
                DF.mvs_targets_df.loc[(DF.mvs_targets_df.mvs_ids.isin(out[:,0])), ['y_%s' % filter]] = out[:,2]


def make_mvs_tiles(DF,filter,pipe_cfg, unq_ids_test_list=[],redo=False, debug=False,
                   xy_SN=False, xy_m=True,
                   xy_cen=False, xy_shift_list=[], xy_dmax=3, verbose=False, workers=None, look4duplicants=True,
                   showduplicants=False, Python_origin=False, parallel_runs=True, cr_remove=False,
                   la_cr_remove=False, cr_radius=3,
                   chunksize=None, multiply_by_exptime=False,
                   multiply_by_gain=False, multiply_by_PAM=False,
                   update_dataframe=False):
    '''
    This is a wrapper for the updating of tiles dataframe and the median tile target dataframe

    '''
    DF.xy_dmax = xy_dmax
    DF.xy_SN = xy_SN
    DF.xy_m = xy_m
    DF.xy_cen = xy_cen
    DF.look4duplicants = look4duplicants
    DF.Python_origin = Python_origin
    DF.cr_remove = cr_remove
    DF.la_cr_remove = la_cr_remove
    DF.cr_radius = cr_radius
    DF.multiply_by_exptime = multiply_by_exptime
    DF.multiply_by_gain = multiply_by_gain
    DF.multiply_by_PAM = multiply_by_PAM

    # zpt = DF.zpt
    if update_dataframe:
        getLogger(__name__).info(f'Updating the targets dataframe')
        update_mvs_targets(DF, pipe_cfg, filter)
    getLogger(__name__).info(f'Working on the tiles')
    if len(unq_ids_test_list) > 0:
        mvs_ids_test_list = np.sort(
            DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.unq_ids.isin(unq_ids_test_list)].mvs_ids.unique())
    else:
        mvs_ids_test_list = []
    if len(unq_ids_test_list) < 2:
        look4duplicants = False
    mk_mvs_tiles(DF, filter, mvs_ids_test_list=mvs_ids_test_list,
                 verbose=verbose, xy_SN=xy_SN, xy_m=xy_m, xy_cen=xy_cen,
                 xy_shift_list=xy_shift_list, xy_dmax=xy_dmax,
                 workers=workers, Python_origin=Python_origin, parallel_runs=parallel_runs,
                 cr_remove=cr_remove, la_cr_remove=la_cr_remove, cr_radius=cr_radius,
                 chunksize=chunksize,
                 multiply_by_exptime=multiply_by_exptime,
                 multiply_by_gain=multiply_by_gain, multiply_by_PAM=multiply_by_PAM,overwrite=redo)

    if look4duplicants:
        check4duplicants(DF, filter, mvs_ids_test_list, showduplicants=showduplicants)
    if debug:
         _ = small_tiles(DF.mvs_targets_df, pipe_cfg.paths['data'], pipe_cfg.paths['database'],
                         [filter], dict={}, nrows=10,ncols=10, crossmatch_ids_df=DF.crossmatch_ids_df,
                         ext=DF.fitsext, fistsroot=pipe_cfg.buildhdf['default_mvs_table']['fitsroot'])


def task_median_tiles(DF,id,filter,zfactor,alignment_box,legend,showplot,method,cr_remove,la_cr_remove,kill,
                      kill_plots,skip_flag,overwrite):
    '''
    Taks perfomed in the update_median_targets_tile.
    Note: all multivisits targets with FILTER_flag == 'rejected' are considered NaN
    when arranging the median tile. If all are 'rejected', the final median tile for
    that FILTER will be all NaN.

    '''

    target_images=[]
    mvs_ids_list=DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.unq_ids==id].mvs_ids.unique()
    PAV_3s=[]
    ROTAs=[]
    path2tile = '%s/median_tiles/%s/tile_ID%i.fits' % (DF.path2out, filter, id)
    getLogger(__name__).info(f'Making median tile {path2tile}')
    if not os.path.exists(path2tile) or overwrite:
        for mvs_ids in mvs_ids_list:
            try:
                if not DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==mvs_ids,'flag_%s'%filter].str.contains(skip_flag).values[0]:
                    sel_ids=(DF.mvs_targets_df.mvs_ids==mvs_ids)
                    DATA=Tile(x=(DF.tilebase-1)/2,y=(DF.tilebase-1)/2,tile_base=DF.tilebase,delta=0,inst=DF.inst,Python_origin=True)
                    DATA.load_tile('%s/mvs_tiles/%s/tile_ID%i.fits' % (DF.path2out, filter, mvs_ids),ext=1,
                                   raise_errors=False)
                    if not np.all(np.isnan(DATA.data)):
                        target_images.append(DATA.data)
                        PAV_3s.append(DF.mvs_targets_df.loc[sel_ids,'pav3_%s'%filter].values[0])
                        ROTAs.append(DF.mvs_targets_df.loc[sel_ids,'rota_%s'%filter].values[0])
                    else:
                        getLogger(__name__).critical(
                            f'Not able to make median tile {path2tile}. All data is NaN.')
                else:
                    getLogger(__name__).critical(
                        f'Not able to make median tile {path2tile}. {skip_flag}.')

            except:
                getLogger(__name__).critical(f'Not able to make media tile {path2tile}. Check your dataframe and paths!')
                raise ValueError


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
                        if not DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==mvs_ids,'flag_%s'%filter].str.contains(skip_flag).values[0] :
                            CRDATA=Tile(x=(DF.tilebase-1)/2,y=(DF.tilebase-1)/2,tile_base=DF.tilebase,delta=0,inst=DF.inst,Python_origin=True)
                            CRDATA.load_tile('%s/mvs_tiles/%s/tile_ID%i.fits' % (DF.path2out, filter, mvs_ids),ext=4)
                            crtarget_images.append(CRDATA.data)
                    if len(crtarget_images)>0:
                        crtarget_images=np.array(crtarget_images)
                        crtarget_tile,shift_list=allign_images(crtarget_images,ROTAs,PAV_3s,filter,legend=legend,showplot=showplot,verbose=False,zfactor=zfactor,alignment_box=alignment_box,title='%s CR clean Median Target'%(filter),method=method,tile_base=DF.tilebase,kill=kill,kill_plots=kill_plots)
                        crtarget_tile.append_tile(path2tile,Datacube=Datacube,verbose=False,name='CRcleanSCI',return_Datacube=False)
                except:
                    getLogger(__name__).critical(
                        f'Not able to make CR media tile {path2tile}. Check your dataframe and paths!')
                    raise ValueError
            if return_Datacube: Datacube.close()
    else:
        getLogger(__name__).info(f'Median Tile {path2tile} already exist. Skipping.')

def make_median_tiles(DF,filter,unq_ids_list=[],workers=None,
                   zfactor=10,alignment_box=3,legend=False,showplot=False,
                   parallel_runs=True,method='median',cr_remove=False,la_cr_remove=False,
                   chunksize = None,kill=False,kill_plots=False,
                   skip_flag='rejected',redo=False):
    '''
    Update the median targets dataframe tile.

    '''
    getLogger(__name__).info(f'Working on median tiles on {filter}')
    config.make_paths(config=None, paths=DF.path2out+'/median_tiles/%s'%filter)
    if len(unq_ids_list)==0: unq_ids_list=DF.unq_targets_df.unq_ids.unique()

    if parallel_runs:
        workers,chunksize,ntarget=parallelization_package(workers,len(unq_ids_list),chunksize = chunksize)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for _ in executor.map(task_median_tiles,repeat(DF),unq_ids_list,repeat(filter),
                                  repeat(zfactor),repeat(alignment_box),repeat(legend),repeat(showplot),
                                  repeat(method),repeat(cr_remove),repeat(la_cr_remove),repeat(kill),
                                  repeat(kill_plots),repeat(skip_flag),repeat(redo),chunksize=chunksize):
                pass

    else:
        for id in unq_ids_list:
            task_median_tiles(DF,id,filter,zfactor,alignment_box,legend,showplot,method,cr_remove,la_cr_remove,kill,
                      kill_plots,skip_flag,redo)


def run(packet):
    DF = packet['DF']
    dataset = packet['dataset']
    for filter in dataset.data_cfg.filters:
        make_mvs_tiles(DF,filter,dataset.pipe_cfg,
                        unq_ids_test_list=dataset.pipe_cfg.mktiles['unq_ids_list'],
                        xy_m=dataset.pipe_cfg.mktiles['xy_m'],
                        workers=int(dataset.pipe_cfg.ncpu),
                        cr_remove=dataset.pipe_cfg.mktiles['cr_remove'],
                        la_cr_remove=dataset.pipe_cfg.mktiles['la_cr_remove'],
                        parallel_runs=dataset.pipe_cfg.mktiles['parallel_runs'],
                        Python_origin=dataset.pipe_cfg.mktiles['python_origin'],
                        look4duplicants=dataset.pipe_cfg.mktiles['look4duplicants'],
                        multiply_by_exptime=dataset.pipe_cfg.mktiles['multiply_by_exptime'],
                        multiply_by_PAM=dataset.pipe_cfg.mktiles['multiply_by_PAM'],
                        multiply_by_gain=dataset.pipe_cfg.mktiles['multiply_by_gain'],
                        cr_radius=dataset.pipe_cfg.mktiles['cr_radius'] / dataset.pipe_cfg.instrument['pixelscale'],
                        redo=dataset.pipe_cfg.mktiles['redo'],
                        debug=dataset.pipe_cfg.mktiles['debug'],
                        xy_dmax=dataset.pipe_cfg.mktiles['xy_dmax'])


        make_median_tiles(DF, filter,
                            unq_ids_list=dataset.pipe_cfg.mktiles['unq_ids_list'],
                            workers=int(dataset.pipe_cfg.ncpu),
                            zfactor=dataset.pipe_cfg.mktiles['zfactor'],
                            alignment_box=dataset.pipe_cfg.mktiles['alignment_box'],
                            parallel_runs=dataset.pipe_cfg.mktiles['parallel_runs'],
                            cr_remove=dataset.pipe_cfg.mktiles['cr_remove'],
                            la_cr_remove=dataset.pipe_cfg.mktiles['la_cr_remove'],
                            kill_plots=dataset.pipe_cfg.mktiles['kill_plots'],
                            redo=dataset.pipe_cfg.mktiles['redo'])

    DF.save_dataframes(__name__)
