import sys
sys.path.append('/')
from utils_dataframe import fk_writing
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from tiles import Tile
from fake_star import Fake_Star
from utils_tile import perform_PSF_subtraction
import numpy as np
from stralog import getLogger
from random import sample,randint,uniform,choice
from astropy.io import fits
from ancillary import PointsInCircum,round2closerint,parallelization_package
from utils_photometry import aperture_photometry_handler,KLIP_aperture_photometry_handler,photometry_AP
from photometry import Detection,flux_converter
import pandas as pd
from buildhdf import make_fk_dataframes

def psf_scale(psfdata):
    psfdata[psfdata<0]=0
    # psfdata+=(1-np.sum(psfdata))/(psfdata.shape[1]*psfdata.shape[0])
    psfdata/=np.sum(psfdata)
    return(psfdata)

def update_fk_dataframes(DF, parallel_runs=False, workers=None, NPSFstars=300, NPSFsample=30, inner_shift=0.25,
                    path2data=None,  showplot=False, aptype='4pixels', delta=1,
                    suffix='',multiply_by_exptime=False):
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
    getLogger(__name__).info('Updating the fake injection dataframe with detections from fake binaries')

    for filter in DF.filters:
        getLogger(__name__).info(f'Collecting background values for filter: {filter}')
        # elno = np.where(filter in DF.filters)[0][0]
        # if path2psfdir == None:
        path2psfdir = path2data+f'/mvs_tiles/{filter}/'
        # psf = []
        # else:
        #     hdul = fits.open(path2psfdir + psf_filename)
        #     psf = hdul[0].data
        #     hdul.close()

        # bkg_list = []
        # ebkg_list = []
        # nbkg_list = []

        mvs_psf_ids_list = DF.mvs_targets_df.loc[
            DF.mvs_targets_df[f'flag_{filter}'] == 'good_psf'].mvs_ids.unique()
        # if use_median_sky:
        #     for fitsfile in tqdm(DF.mvs_targets_df[f'fits_{filter}'].unique()):
        #         fits_image_filename = path2data
        #                 '%s\%s\%s\%s\%s' % (
        #         str(DF.path2fitsdir), DF.project, DF.target, DF.inst, fitsfile + '.fits'))
        #         hdul = fits.open(fits_image_filename)
        #         exptime = hdul[0].header['exptime']
        #         for ext in [1, 4]:
        #             data = hdul[ext].data
        #             data[data < 0] = 0
        #             if not DF.header_df.loc['multiply_by_exptime', 'Values']:
        #                 data /= exptime
        #             if DF.header_df.loc['multiply_by_gain', 'Values']:
        #                 data *= DF.gain
        #
        #             if DF.header_df.loc['multiply_by_PAM', 'Values']:
        #                 path2PAM = '%s/%s/%s/%s/PAM' % (DF.path2fitsdir, DF.project, DF.target, DF.inst)
        #                 phdul = fits.open(path2PAM + '/' + str(DF.PAMdict[0][ext] + '.fits'))
        #                 try:
        #                     PAM = phdul[1].data
        #                 except:
        #                     PAM = phdul[0].data
        #                 data *= PAM
        #
        #         sigma_clip = SigmaClip(sigma=3.)
        #         bkg_estimator = MedianBackground()
        #         bkg = Background2D(data, (10, 10), filter_size=(3, 3),
        #                            sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
        #         bkg_list.append(bkg.background_median)
        #         ebkg_list.append(bkg.background_rms_median)
        #
        #     bkg_list = np.array(bkg_list)
        #     ebkg_list = np.array(ebkg_list)
        #     nbkg_list = np.array(nbkg_list)
        #     bgk_sel = ~np.isnan(bkg_list)
        #     bkg_list = bkg_list[bgk_sel]
        #     ebkg_list = ebkg_list[bgk_sel]
        # else:
        bkg_list = DF.mvs_targets_df[f'sky_{filter}' ].values.astype(float) / DF.mvs_targets_df[
            f'exptime_{filter}'].values.astype(float)
        ebkg_list = DF.mvs_targets_df[f'esky_{filter}' ].values.astype(float) / DF.mvs_targets_df[
            f'exptime_{filter}'].values.astype(float)
        nbkg_list = DF.mvs_targets_df[f'nsky_{filter}'].values.astype(float)

        bgk_sel = ~np.isnan(bkg_list)
        bkg_list = bkg_list[bgk_sel]
        ebkg_list = ebkg_list[bgk_sel]
        nbkg_list = nbkg_list[bgk_sel]

        magmin = DF.mvs_targets_df.loc[(DF.mvs_targets_df[f'flag_{filter}'] == 'good_psf'), f'm_{filter}{suffix}'].min()  # &(DF.mvs_targets_df['m%s%s'%(filter[1:4],suffix)]>0)
        magmax = DF.mvs_targets_df.loc[(DF.mvs_targets_df[f'flag_{filter}'] == 'good_psf'), f'm_{filter}{suffix}'].max()
        zpt = DF.zpt[filter]
        fk_ids_list = [i for i in range(0, NPSFstars)]

        # getLogger(__name__).info(f'Generating the fake PSF library stars for filter {filter}')
        psf_list = []
        psf_ids_list = []
        for fk_ids in fk_ids_list:
            ID = int(choice(mvs_psf_ids_list))
            getLogger(__name__).info(f'Loading PSF from tile_ID{ID}.fits')
            exptime = DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids == ID, f'exptime_{filter}'].values[0]
            # if len(psf) == 0:
            with fits.open(str(path2psfdir) + f'/tile_ID{ID}.fits', memmap=False) as hdul:
                if multiply_by_exptime:
                    target_psf = hdul[-1].data / exptime
                else:
                    target_psf = hdul[-1].data
                psf_ids_list.append(ID)
            # else:
            #     target_psf = psf.copy()

            psf_list.append(
                task_fake_reference_infos(DF, fk_ids, target_psf, magmin, magmax, zpt, exptime, bkg_list, ebkg_list,
                                          inner_shift))

        magbin_list = DF.fk_targets_df.loc[filter].index.get_level_values('magbin').unique().astype(float)
        dmag_list = DF.fk_targets_df.loc[filter].index.get_level_values('dmag').unique().astype(float)
        sep_list = DF.fk_targets_df.loc[filter].index.get_level_values('sep').unique().astype(float)
        columns_no_injection = ['x', 'y', 'm', 'exptime'] + np.array(
            [[f'nsigma_kmode{kmode}', f'counts_kmode{kmode}', f'noise_kmode{kmode}', f'm_kmode{kmode}']
             for kmode in DF.kmodes]).ravel().tolist()
        columns = ['x', 'y', 'counts', 'm', 'exptime'] + np.array(
            [[f'nsigma_kmode{kmode}', f'counts_kmode{kmode}', f'noise_kmode{kmode}', f'm_kmode{kmode}']
             for kmode in DF.kmodes]).ravel().tolist()

        for magbin in magbin_list:
            for dmag in dmag_list:
                if parallel_runs:
                    workers, chunksize, ntarget = parallelization_package(workers, len(sep_list), verbose=False)
                    with ProcessPoolExecutor(max_workers=workers) as executor:
                        for out, out_no_injection in executor.map(task_fake_infos, repeat(DF), repeat(magbin),
                                                                  repeat(dmag), sep_list, repeat(filter), repeat(zpt),
                                                                  repeat(psf_list), repeat(psf_ids_list),
                                                                  repeat(NPSFsample), repeat(bkg_list),
                                                                  repeat(ebkg_list), repeat(nbkg_list),
                                                                  repeat(inner_shift), repeat(path2psfdir),
                                                                  repeat(multiply_by_exptime), repeat(showplot),
                                                                  repeat(aptype), repeat(delta), chunksize=chunksize):
                            DF=fk_writing(DF, filter, out_no_injection, 'fk_targets_df', columns_no_injection)
                            DF=fk_writing(DF, filter, out, 'fk_candidates_df', columns)
                            del out, out_no_injection
                else:
                    for sep in sep_list:
                        out, out_no_injection = task_fake_infos(DF, magbin, dmag, sep, filter, zpt, psf_list,
                                                                psf_ids_list, NPSFsample, bkg_list, ebkg_list,
                                                                nbkg_list, inner_shift, path2psfdir,
                                                                multiply_by_exptime, showplot, aptype, delta)
                        DF=fk_writing(DF, filter, out_no_injection, 'fk_targets_df', columns_no_injection)
                        DF=fk_writing(DF, filter, out, 'fk_candidates_df', columns)
                        del out, out_no_injection
        del psf_list, psf_ids_list, bkg_list, ebkg_list
    return(DF)


def task_fake_reference_infos(DF, elno, psf, magmin, magmax, zpt, exptime, bkg_list, ebkg_list, inner_shift):
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
    idx = randint(0, len(bkg_list) - 1)
    bkg = bkg_list[idx]
    ebkg = ebkg_list[idx]
    m3 = round(uniform(magmin, magmax + 1), 2)
    shift3 = [round(uniform(-inner_shift, inner_shift), 2), round(uniform(-inner_shift, inner_shift), 2)]
    c3 = 10 ** (-(m3 - zpt) / 2.5)
    PSF = Fake_Star(psf, c3 * exptime, shift=shift3, Sky=bkg * exptime, eSky=ebkg * exptime)
    im3 = np.array(PSF.star.tolist()) / exptime
    if psf.shape[1] % 2 == 0:
        x_cen = int((psf.shape[1]) / 2)
    else:
        x_cen = int((psf.shape[1] - 1) / 2)

    if psf.shape[0] % 2 == 0:
        y_cen = int((psf.shape[0]) / 2)
    else:
        y_cen = int((psf.shape[0] - 1) / 2)

    DATA3 = Tile(data=im3, x=x_cen, y=y_cen, tile_base=DF.tilebase, inst=DF.inst)
    DATA3.mk_tile(pad_data=False, verbose=False, xy_m=False, legend=False, showplot=False, keep_size=True,
                  kill_plots=True, cbar=True)
    return (DATA3.data.tolist())

def task_fake_infos(DF, magbin, dmag, sep, filter, zpt, psf_list, psf_ids_list, npsfs, bkg_list, ebkg_list,
                    nbkg_list, inner_shift, path2psfdir, multiply_by_exptime, showplot, aptype, delta):
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
    nstars=DF.fk_targets_df.index.get_level_values('fk_ids').nunique()
    getLogger(__name__).info(f'Generating {nstars} fake binaries for filter {filter}, magbin {magbin}, dmag {dmag}, sep {sep}')

    out_list = []
    out_no_injection_list = []

    for elno in DF.fk_targets_df.index.get_level_values('fk_ids').unique():
        idx = randint(0, len(bkg_list) - 1)
        ID = psf_ids_list[idx]
        bkg = bkg_list[idx]
        ebkg = ebkg_list[idx]

        exptime = DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids == ID, f'exptime_{filter}'].values[0]
        # if len(psf) == 0:
        with fits.open(str(path2psfdir) + '/tile_ID%s.fits' % ID, memmap=False) as hdul:
            if multiply_by_exptime:
                target_psf = hdul[-1].data / exptime
            else:
                target_psf = hdul[-1].data
        # else:
        #     target_psf = psf.copy()

        m1 = round(float(magbin), 2)  # uniform(magbin,magbin+1)
        shift1 = [round(uniform(-inner_shift, inner_shift), 2), round(uniform(-inner_shift, inner_shift), 2)]
        c1 = 10 ** (-(m1 - zpt) / 2.5)

        contrast = round(float(dmag), 2)  # round(uniform(dmag,dmag+1),2)
        m2 = m1 + contrast
        c2 = 10 ** (-(m2 - zpt) / 2.5)

        r = sep  # round(uniform(sep,sep+1),2)
        x, y = PointsInCircum(r)
        shift2 = [round(uniform(-inner_shift, inner_shift), 2), round(uniform(-inner_shift, inner_shift), 2)]
        if round2closerint([x + int((DF.tilebase - 1) / 2) + shift2[1]])[0] < 0 or \
                round2closerint([y + int((DF.tilebase - 1) / 2) + shift2[0]])[0] < 0 or \
                round2closerint([x + int((DF.tilebase - 1) / 2) + shift2[1]])[0] >= DF.tilebase or \
                round2closerint([y + int((DF.tilebase - 1) / 2) + shift2[0]])[0] >= DF.tilebase:
            cc = 0
            while round2closerint([x + int((DF.tilebase - 1) / 2) + shift2[1]])[0] < 0 or \
                    round2closerint([y + int((DF.tilebase - 1) / 2) + shift2[0]])[0] < 0 or \
                    round2closerint([x + int((DF.tilebase - 1) / 2) + shift2[1]])[0] >= DF.tilebase or \
                    round2closerint([y + int((DF.tilebase - 1) / 2) + shift2[0]])[0] >= DF.tilebase:
                x, y = PointsInCircum(r)
                if cc >= 1000: raise ValueError(
                    'Positions %i,%i (generated by a separation of %.2f), are outside the tile box 0,%i. Please choose smaller separations' % (
                    round2closerint([x + int((DF.tilebase - 1) / 2) + shift2[1]])[0],
                    round2closerint([y + int((DF.tilebase - 1) / 2) + shift2[0]])[0], r, DF.tilebase - 1))
                cc += 1

        if target_psf.shape[1] % 2 == 0:
            x_cen = int((target_psf.shape[1]) / 2)
        else:
            x_cen = int((target_psf.shape[1] - 1) / 2)

        if target_psf.shape[0] % 2 == 0:
            y_cen = int((target_psf.shape[0]) / 2)
        else:
            y_cen = int((target_psf.shape[0] - 1) / 2)
        target_psf = psf_scale(target_psf)
        if showplot:
            try:
                radius_in = DF.header_df.loc['radius_in', 'Values']
                radius1 = DF.header_df.loc['radius1_in', 'Values']
                radius2 = DF.header_df.loc['radius2_in', 'Values']
            except:
                radius_in = 5
                radius1 = 10
                radius2 = 15

            gain = DF.header_df.loc['gain', 'Values']

            print('> PSF, ID%i:' % ID)
            aperture_photometry_handler(DF, 0, filter, x=int((DF.tilebase - 1) / 2),
                                        y=int((DF.tilebase - 1) / 2), data=target_psf, zpt=0, ezpt=0,
                                        aptype='circular', noBGsub=True, sigma=3, kill_plots=False, Python_origin=True,
                                        delta=3, exptime=1, radius1=radius1, radius2=radius2, gain=gain)

        S0 = Fake_Star(target_psf, c1 * exptime, shift=shift1, Sky=bkg * exptime, eSky=ebkg * exptime)
        im0 = np.array(S0.star.tolist())
        S1 = Fake_Star(target_psf, c1 * exptime, shift=shift1, Sky=bkg * exptime, eSky=ebkg * exptime, PNoise=False)
        S2 = Fake_Star(target_psf, c2 * exptime, shift=shift2, PNoise=False)
        im2 = np.array(S2.star.tolist())
        pos = [y, x]
        S1.combine(S2.star, pos)
        im12 = np.array(S1.binary.tolist())

        DATA0 = Tile(data=im0, x=x_cen, y=y_cen, tile_base=DF.tilebase, inst=DF.inst)
        DATA0.mk_tile(pad_data=False, verbose=False, xy_m=False, legend=False, showplot=False, keep_size=False,
                      kill_plots=True, cbar=True, title='S0 ,m %i' % (m1))

        DATA12 = Tile(data=im12, x=x_cen, y=y_cen, tile_base=DF.tilebase, inst=DF.inst)
        DATA12.mk_tile(pad_data=False, verbose=False, xy_m=False, legend=False, showplot=False, keep_size=False,
                       kill_plots=True, cbar=True, title='Bin ,magbin %i,dmag %i, sep %.1f' % (magbin, dmag, sep))
        pos = [y + int((DF.tilebase - 1) / 2) + shift2[0], x + int((DF.tilebase - 1) / 2) + shift2[1]]
        y, x = round2closerint(pos)

        dt = Detection(im2, x_cen, y_cen)
        photometry_AP.aperture_mask(dt, aptype=aptype, ap_x=delta, ap_y=delta)
        photometry_AP.mask_aperture_data(dt)
        photometry_AP.aperture_stats(dt, aperture=dt.aperture, sigma=3)
        flux_converter.counts_and_errors(dt)

        if showplot:
            print('> Isolated primary with Sky:'.upper())
            aperture_photometry_handler(DF, 0, filter, x=x_cen, y=y_cen, data=im0, zpt=zpt, ezpt=0, aptype='circular',
                                        noBGsub=False, sigma=3, kill_plots=False, Python_origin=True, exptime=exptime,
                                        radius_a=radius_in, radius1=radius1, radius2=radius2, gain=gain)
            print('> Isolated companion without Sky:'.upper())
            print('> Input counts: %e' % (c2 * exptime))
            aperture_photometry_handler(DF, 0, filter, x=x_cen, y=y_cen, data=im2, zpt=zpt, ezpt=0, aptype='circular',
                                        noBGsub=True, sigma=3, kill_plots=False, Python_origin=True, exptime=exptime,
                                        radius_a=radius_in, radius1=radius1, radius2=radius2, gain=gain)
            print('> %s aperture on the same target:' % aptype)
            aperture_photometry_handler(DF, 0, filter, x=x_cen, y=y_cen, data=im2, zpt=zpt, ezpt=0, aptype=aptype,
                                        noBGsub=True, sigma=3, kill_plots=False, Python_origin=True, exptime=exptime,
                                        radius_a=radius_in, radius1=radius1, radius2=radius2, gain=gain, delta=delta)
            print('> Expected counts: %e' % dt.counts)

            print('> Binary with the two component combined:'.upper())
            print('> Centered on the primary:')
            aperture_photometry_handler(DF, 0, filter, x=x_cen, y=y_cen, data=im12, zpt=zpt, ezpt=0,
                                        aptype='circular', noBGsub=False, sigma=3, kill_plots=False, Python_origin=True,
                                        exptime=exptime, radius_a=radius_in, radius1=radius1, radius2=radius2,
                                        gain=gain)
            print('> Centered on the companion:')

        out_0_no_injection = [magbin, dmag, sep, elno, int((DF.tilebase - 1) / 2) + shift1[1],
                              int((DF.tilebase - 1) / 2) + shift1[0], m1, exptime]
        out_0 = [magbin, dmag, sep, elno, x + int((DF.tilebase - 1) / 2) + shift2[1],
                 y + int((DF.tilebase - 1) / 2) + shift2[0], dt.counts, m2, exptime]
        # try:
        if showplot:
            print('>. PSF subtraction with no injection')
        out_no_injection = perform_KLIP_PSF_subtraction_on_fakes(DF, filter, DATA0.data, psf_list, pos,
                                                                 DF.kmodes, npsfs, showplot, exptime, aptype,
                                                                 delta, noBGsub=False)
        if showplot:
            print('>. PSF subtraction with injection')
        out = perform_KLIP_PSF_subtraction_on_fakes(DF, filter, DATA12.data, psf_list, pos, DF.kmodes,
                                                        npsfs, showplot, exptime, aptype, delta, noBGsub=False)
        # except:
        #     raise ValueError([magbin, dmag, sep])
        out_0_no_injection.extend(out_no_injection)
        out_0.extend(out)
        out_no_injection_list.append(out_0_no_injection)
        out_list.append(out_0)
        del S0, S1, S2, DATA0, DATA12
    return (out_list, out_no_injection_list)

def perform_KLIP_PSF_subtraction_on_fakes(DF,filter,target,psf_list,pos,Kmodes_list,npsfs,showplot,exptime,aptype,delta,noBGsub):
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
    Kmodes_list : list
        kmode list to analyze.
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
    residuals,_=perform_PSF_subtraction(targ_tiles,ref_tiles,kmodes=Kmodes_list,no_PSF_models=True)
    zpt=DF.zpt[filter]
    kill_plots=True
    out_list=[]
    for kmode in residuals.columns:
        if not np.all(np.isnan(residuals[kmode].values[0])):
            if showplot:
                print('>. KLIPmode %i'%kmode)
                kill_plots=False
            counts,ecounts,Nsigma,Nap,mag,emag,spx,bpx,Sky,eSky,nSky,grow_corr=KLIP_aperture_photometry_handler(DF,0,filter,x=x,y=y,data=residuals[kmode].values[0],zpt=zpt,ezpt=0,aptype=aptype,noBGsub=noBGsub,sigma=3,kill_plots=kill_plots,Python_origin=True,delta=delta,exptime=exptime,gain=1)
            out_list.extend([Nsigma,counts,ecounts,mag])
        else:
            out_list.extend([np.nan,np.nan,np.nan,np.nan])
    return(out_list)

def run(packet):
    DF = packet['DF']
    dataset = packet['dataset']
    generate_fk_binaries

    if dataset.pipe_cfg.fpanalysis['redo']:
        make_fk_dataframes(DF, dataset)
        DF=update_fk_dataframes(DF,
                                parallel_runs=dataset.pipe_cfg.fpanalysis['parallel_runs'],
                                workers=dataset.pipe_cfg.ncpu,
                                NPSFstars=dataset.pipe_cfg.fpanalysis['NPSFstars'],
                                NPSFsample=dataset.pipe_cfg.fpanalysis['NPSFsample'],
                                inner_shift=dataset.pipe_cfg.fpanalysis['inner_shift'],
                                path2data=dataset.pipe_cfg.paths['out'],
                                showplot=dataset.pipe_cfg.fpanalysis['showplot'],
                                aptype=dataset.pipe_cfg.klipphotometry['aptype'],
                                delta=dataset.pipe_cfg.klipphotometry['delta'],
                                suffix=dataset.pipe_cfg.fpanalysis['suffix'],
                                multiply_by_exptime=dataset.pipe_cfg.mktiles['multiply_by_exptime'],
                                )

    # DF=evaluate_completeness_from_fk_inj(path2savedir=path2dir, Nvisit_range=Nvisit_range, AUC_lim=0.5, parallel_runs=True,
    #                                   workers=workers)
    # DF=mvs_plot_completness(save_figure=False, showplot=True, avg_ids_list=None, Nvisit_list=[...], MagBin_list=[...],
    #                      Kmodes_list=[...], ticks=np.arange(0.3, 1., 0.1))
    # DF=update_candidates_photometry_dataframes(label='data', aptype='4pixels', verbose=True, delta=2)
    # DF=false_positive_analysis(avg_ids_list=[], showplot=True, verbose=True)


    DF.save_dataframes(__name__)