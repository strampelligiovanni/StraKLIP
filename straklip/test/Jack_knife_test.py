import astropy.io.fits as fits
import config, input_tables
from datetime import datetime
import numpy as np
import os
from tiles import Tile
import numpy.ma as ma
from utils_tile import perform_PSF_subtraction
from utils_photometry import KLIP_aperture_photometry_handler
import matplotlib.pylab as plt
import pickle
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from ancillary import parallelization_package
from tqdm import tqdm
import pyklip.rdi as rdi
from glob import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import sigmaclip

def perform_jackknife_test_on_tiles(DF,id,filter,cell,label_dict,hdul_dict,label,skipDQ,fitsoutdir,parallel_runs):
    #Load the target and the PSF reference and build the lists needed to assemble the PSFlib and run KLIP
    psf_ids_list=DF.mvs_targets_df.loc[DF.mvs_targets_df['flag_%s'%filter].str.contains('psf')&(DF.mvs_targets_df['cell_%s'%filter]==cell)].mvs_ids.unique()
    # residuals_dict = {}

    mkmode = DF.unq_candidates_df.loc[DF.unq_candidates_df.unq_ids == id, 'mkmode'].values[0]
    path2tile = '%s/mvs_tiles/%s/tile_ID%i.fits' % (DF.path2out, filter, id)
    DATA=Tile(x=(DF.tilebase-1)/2,y=(DF.tilebase-1)/2,tile_base=DF.tilebase,delta=0,inst=DF.inst,Python_origin=True)
    Datacube=DATA.load_tile(path2tile,ext=label_dict[label],verbose=False,return_Datacube=True,hdul_max=hdul_dict[label],mode='update',raise_errors=True)
    if not skipDQ:
        DQ=Tile(x=int((DF.tilebase-1)/2),y=int((DF.tilebase-1)/2),tile_base=DF.tilebase,inst=DF.inst)
        DQ.load_tile(path2tile,ext='dq',verbose=False,return_Datacube=False)
        DQ_list=list(set(DQ.data.ravel()))

    x = DATA.data.copy()
    if not skipDQ:
        mask_x=DQ.data.copy()
        for i in [i for i in DQ_list if i not in DF.dq2mask]:
            mask_x[(mask_x==i)]=0
        mx = ma.masked_array(x, mask=mask_x)
        mx.data[mx.mask]=-9999
        mx.data[mx.data<0]=0
        targ_tiles=np.array(mx.data)
    else:
        x[x<0]=0
        targ_tiles=np.array(x)
    filename = DF.path2out + f'/mvs_tiles/{filter}/tile_ID{id}.fits'
    elno=0
    if not np.all(np.isnan(targ_tiles)):
        ref_tiles=[]
        psfnames=[]
        for refid in DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(psf_ids_list)].mvs_ids.unique():
            path2ref = '%s/mvs_tiles/%s/tile_ID%i.fits' % (DF.path2out, filter, refid)
            REF=Tile(x=(DF.tilebase-1)/2,y=(DF.tilebase-1)/2,tile_base=DF.tilebase,delta=0,inst=DF.inst,Python_origin=True)
            REF.load_tile(path2ref,ext=label_dict[label],verbose=False,hdul_max=hdul_dict[label])
            if not skipDQ:
                DQREF=Tile(x=int((DF.tilebase-1)/2),y=int((DF.tilebase-1)/2),tile_base=DF.tilebase,inst=DF.inst)
                DQREF.load_tile(path2ref,ext='dq',verbose=False,return_Datacube=False)

            xref=REF.data.copy()
            if not skipDQ:
                mask_ref=DQREF.data.copy()
                for i in [i for i in DQ_list if i not in DF.dq2mask]:  mask_ref[(mask_ref==i)]=0
                mref = ma.masked_array(xref, mask=mask_ref)
                mref.data[mref.mask]=-9999
                if len(mref[mref<=-9999])>10:
                    psf_ids_list.pop(elno)
                else:
                    psfnames.append(path2ref)
                    ref_tiles.append(np.array(mref.data))
                    mref.data[mref.data<0]=0
            else:
                xref[xref<0]=0
                ref_tiles.append(np.array(xref))
                psfnames.append(path2ref)

            elno+=1

        if id not in psf_ids_list:
            psfnames.append(filename)
            ref_tiles.append(targ_tiles)

        iterables=[i for i in range(len(ref_tiles))]
        workers, chunksize, ntarget = parallelization_package(5, len(iterables), chunksize=None)
        if parallel_runs:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for residuals,elno in executor.map(task_PSF_subtraction, iterables,repeat(id),
                                    repeat(targ_tiles),repeat(ref_tiles), repeat(psf_ids_list), repeat(filename),
                                    repeat(psfnames),repeat(mkmode), repeat(fitsoutdir),chunksize=chunksize):
                    # residuals_dict[elno] = residuals
                    hdu = fits.PrimaryHDU(data=residuals)
                    hdu.writeto(fitsoutdir + f'residual_ID{id}_ref{elno}.fits',overwrite=True)
        else:
            # ##############For testing only ###############
            for i in range(len(ref_tiles)):
                print(f'Running test {i+1} of {len(ref_tiles)-1}')
                residuals,elno = task_PSF_subtraction(i,id,targ_tiles,ref_tiles,psf_ids_list,filename,psfnames,mkmode,fitsoutdir)
                # residuals_dict[elno] = residuals
                hdu = fits.PrimaryHDU(data=residuals)
                hdu.writeto(fitsoutdir + f'residual_ID{id}_ref{elno}.fits',overwrite=True)

    # return residuals_dict

def task_PSF_subtraction(elno,id,targ_tiles,ref_tiles,psf_ids_list,filename,psfnames,mkmode,path2out):
    # for each reference in the PSFlib, remove one and rerun the PSF subtraction. Do not remove the target if in the PSFlib,
    # it will be excluded by KLIP automatically.
    # print('Removing ref number: ',elno)
    ref_tiles_temp = []
    psfnames_temp = []
    w = np.where(id == psf_ids_list)[0]
    for i in range(len(psf_ids_list)):
        if (i != elno) or (i in w and i == w[0]):
            ref_tiles_temp.append(ref_tiles[i])
            psfnames_temp.append(psfnames[i])

    residuals, _ = perform_PSF_subtraction(targ_tiles,
                                            np.array(ref_tiles_temp),
                                            filename=filename,
                                            psfnames=psfnames_temp,
                                            kmodes=int(mkmode),
                                            outputdir=path2out,
                                            prefix=f'ref{elno}')

    return residuals, elno

if __name__ == "__main__":
    # id = 3
    filter = 'f850lp'
    label_dict = {'data': 1, 'crclean_data': 4}
    hdul_dict = {'data': 3, 'crclean_data': 4}
    run_test = False
    parallel_runs=False

    label = 'data'
    outputdir = "/Users/gstrampelli/PycharmProjects/Giovanni/work/analysis/FFP_drc/test/data/"
    pipe_cfg = '/Users/gstrampelli/PycharmProjects/Giovanni/work/pipeline_logs/FFP_drc/pipe.yaml'
    data_cfg = '/Users/gstrampelli/PycharmProjects/Giovanni/work/pipeline_logs/FFP_drc/data.yaml'
    pipe_cfg = config.configure_pipeline(pipe_cfg, pipe_cfg=pipe_cfg, data_cfg=data_cfg,
                                         dt_string=datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    data_cfg = config.configure_data(data_cfg, pipe_cfg)

    dataset = input_tables.Tables(data_cfg, pipe_cfg)
    aptype = dataset.pipe_cfg.klipphotometry['aptype']
    delta = dataset.pipe_cfg.klipphotometry['delta'],
    radius = dataset.pipe_cfg.klipphotometry['radius'],
    sat_thr = dataset.pipe_cfg.klipphotometry['sat_thr'],

    DF = config.configure_dataframe(dataset, load=True)
    print(f'Filter: {filter}')

    fitsoutdir = DF.path2out + f'/jeckknife/kliptemp/{filter}/'
    if not os.path.exists(fitsoutdir):
        os.makedirs(fitsoutdir, exist_ok=True)

    testoutdir = DF.path2out + f'/jeckknife/test/'
    if not os.path.exists(testoutdir):
        os.makedirs(testoutdir, exist_ok=True)

    for id in DF.mvs_candidates_df.mvs_ids.unique():
        print('Working on ID: ', id)
        if run_test:
            perform_jackknife_test_on_tiles(DF,id,filter,0,label_dict,hdul_dict,label,True, fitsoutdir,parallel_runs)

        else:
            x = DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids == id, f'x_tile_{filter}'].values[0].astype(int)
            y = DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids == id, f'y_tile_{filter}'].values[0].astype(int)
            fig, axes = plt.subplots(11, 5, figsize=(13, 30))#, sharex=True, sharey=True)
            elno1 = 0
            elno2 = 0
            for key in tqdm(np.sort([int(i.split('/')[-1].split('.fits')[0].split('ref')[-1]) for i in glob(fitsoutdir+f'residual_ID{id}_*')])):
                with fits.open(fitsoutdir+f'residual_ID{id}_ref{key}.fits') as hdul:
                    data = hdul[0].data[0]

                filtered_data, lower_bound, upper_bound = sigmaclip(data, low=5, high=5)
                STD = np.nanstd(filtered_data)
                Nsigma = np.round(data[y][x] / STD,2)

                im=axes[elno1][elno2].imshow(data/STD, origin='lower')
                axes[elno1][elno2].plot(x,y, 'ok',ms=1)
                axes[elno1][elno2].set_title(f'Ref {key}, nsigma = {Nsigma:.2f}')
                axes[elno1][elno2].tick_params(
                    axis='both',  # Changes apply to both x and y axes
                    which='both',  # Changes apply to major and minor ticks
                    bottom=False,  # Ticks along the bottom edge are off
                    top=False,  # Ticks along the top edge are off
                    left=False,  # Ticks along the left edge are off
                    right=False,  # Ticks along the right edge are off
                    labelbottom=False,  # Labels along the bottom edge are off
                    labelleft=False  # Labels along the left edge are off
                )

                divider = make_axes_locatable(axes[elno1][elno2])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(im, cax=cax)

                if elno2 == 4:
                    elno1 += 1
                    elno2 = 0
                else:
                    elno2 += 1

            for ax in axes.flat:
                if not ax.has_data():
                    ax.remove()

            plt.tight_layout()
            plt.savefig(f'{testoutdir}/jacknife_test_{filter}_{id}.png')
            plt.close()

        print()