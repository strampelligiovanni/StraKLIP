from utils_photometry import aperture_photometry_handler,KLIP_aperture_photometry_handler,photometry_AP,KLIP_throughput
from utils_plot import mvs_completeness_plots
from photometry import Detection,flux_converter
import pandas as pd
from tiles import Tile
from fake_star import Fake_Star
from utils_tile import perform_PSF_subtraction
from random import sample,randint,uniform,choice
from ancillary import PointsInCircum,round2closerint
from stralog import getLogger
from itertools import repeat
from ancillary import parallelization_package
from utils_false_positives import FP_analysis,get_roc_curve
import numpy as np
from sklearn import metrics
from astropy.io import fits

def psf_scale(psfdata):
    psfdata[psfdata<0]=0
    # psfdata+=(1-np.sum(psfdata))/(psfdata.shape[1]*psfdata.shape[0])
    psfdata/=np.sum(psfdata)
    return(psfdata)

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
        # idx = randint(0, len(bkg_list) - 1)
        ID = choice(psf_ids_list)
        # bkg = bkg_list[idx]
        # ebkg = ebkg_list[idx]
        exptime = DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids == ID, f'exptime_{filter}'].values[0]
        bkg = DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids == ID,f'sky_{filter}'].values.astype(float) / exptime
        ebkg = DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids == ID,f'esky_{filter}'].values.astype(float) / exptime
        # nbkg_list = DF.mvs_targets_df[f'nsky_{filter}'].values.astype(float)

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


def mk_completeness_from_fakes(DF, filter, Nvisit_range=None, magbin_list=None, AUC_lim=0.75, FP_lim=0.001,
                               DF_fk=None, chunksize=None, skip_filters=[], parallel_runs=False, workers=None):
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
    # if magbin_list == None:
    magbin_list = DF.fk_candidates_df.loc[filter].index.get_level_values('magbin').unique()
    dmag_list = DF.fk_candidates_df.loc[filter].index.get_level_values('dmag').unique()
    sep_list = DF.fk_candidates_df.loc[filter].index.get_level_values('sep').unique()
    values_list = []

    if parallel_runs:
        for Nvisit in Nvisit_range:
            workers, chunksize, ntarget = parallelization_package(workers, len(magbin_list), chunksize=chunksize)
            getLogger(__name__).info(f'Testing nvisit {Nvisit}.')

            FP_sel = FP_lim ** (1 / (Nvisit * len(DF.filters)))
            if DF_fk == None: DF_fk = DF
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for out_list in executor.map(task_completeness_from_fakes_infos, repeat(DF), repeat(filter),
                                                  repeat(Nvisit), magbin_list, repeat(dmag_list), repeat(sep_list),
                                                  repeat(FP_sel), repeat(AUC_lim), repeat(skip_filters),
                                                  chunksize=chunksize):
                    values_list.extend(out_list)

    else:
        for Nvisit in Nvisit_range:
            getLogger(__name__).info(f'Testing nvisit {Nvisit}.')
            FP_sel = FP_lim ** (1 / (Nvisit * len(DF.filters)))
            if DF_fk == None: DF_fk = DF
            for magbin in magbin_list:
                out_list = task_completeness_from_fakes_infos(DF, filter, Nvisit, magbin, dmag_list, sep_list,
                                                              FP_sel, AUC_lim, skip_filters)
                values_list.extend(out_list)

    getLogger(__name__).info(f'Writing entry in dataframe')

    for elno in range(len(values_list)):
        DF.fk_completeness_df.loc[
            (filter, values_list[elno][0], values_list[elno][1], values_list[elno][2], values_list[elno][3]), [
                'ratio_kmode%s' % values_list[elno][-1]]] = values_list[elno][4]
    return(DF)

def task_completeness_from_fakes_infos(DF, filter, Nvisit, magbin, dmag_list, sep_list, FP_sel, AUC_lim,
                                       skip_filters):
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
    out_list = []
    for dmag in dmag_list:
        for sep in sep_list:
            for Kmode in DF.kmodes:
                if filter not in skip_filters:
                    TPnsigma_inj_list = DF.fk_candidates_df.loc[
                        (filter, magbin, dmag, sep), ['nsigma_kmode%i' % (Kmode)]].values.ravel()
                    FPnsigma_list = DF.fk_targets_df.loc[
                        (filter, magbin, dmag, sep), ['nsigma_kmode%i' % (Kmode)]].values.ravel()

                    X, Y, th = get_roc_curve(FPnsigma_list, TPnsigma_inj_list, nbins=10000)
                    X = np.insert(X, 0, 0)
                    Y = np.insert(Y, 0, 0)

                    w = min(np.where(abs(X - FP_sel) == min(abs(X - FP_sel)))[0])
                    AUC = metrics.auc(X, Y)
                    Ratio_list = len(TPnsigma_inj_list[TPnsigma_inj_list >= th[w]]) / len(TPnsigma_inj_list)
                    # bins=np.arange(np.min(np.min(X),np.min(Y)),np.max(np.max(X),np.max(Y)),20)
                if AUC >= AUC_lim:
                    Ratio_median = round(np.nanmedian(Ratio_list), 3)
                else:
                    Ratio_median = 0
                out_list.append([Nvisit, magbin, dmag, sep, Ratio_median, Kmode])
                if np.any(np.isnan([Nvisit, magbin, dmag, sep, Ratio_median, Kmode])):
                    raise ValueError([AUC, AUC_lim, visit, magbin, dmag, sep, Ratio_median, Kmode])

    return (out_list)

def mvs_plot_completness(DF, filters_list=None, path2savedir=None, MagBin_list=[], Nvisit_list=[], avg_ids_list=[],
                         Kmodes_list=[], title=None, fx=7, fy=7, fz=20, ncolumns=4, xnew=None, ynew=None,
                         ticks=np.arange(0.3, 1., 0.1), show_IDs=False, save_completeness=False, save_figure=False,
                         skip_filters=[], showplot=False, parallel_runs=True, suffix=''):
    '''
    This is a wrapper for the mvs_completeness_plots

    Parameters
    ----------
    Nvisit_range : list, or int
        If a list, make a completness map for each of these numbers of visitis.
        If None, take it from catalogue. The default is None.
    path2savedir : str, optional
        path where to save the cell selection pdf. The default is './Plots/'.
    Kmodes_list : list, optional
        list of KLIPmode to use to evaluate the median completness for the tong plot. The default is [].
    title : str, optional
        title of the tong plot. The default is ''.
    fx : int, optional
        x dimension of each subplots. The default is 7.
    fy : int, optional
        y dimension of each subplots. The default is 7.
    fz : int, optional
        font size for the title. The default is 20.
    ncolumns : int, optional
        number of colum for the subplots. The default is 4.
    xnew : list or None, optional
        list of values for interpolate on the X axis. If None use the default X axixs of the dataframe. The default is None.
    ynew : list or None, optional
        list of values for interpolate on the Y axis. If None use the default X axixs of the dataframe. The default is None.
    ticks : list, optional
        mark a contourn line corresponding to these values on the tong plot. The default is np.arange(0.3,1.,0.1).
    show_IDs : bool, optional
        choose to show IDs for each candidate on the tong plot. The default is False.
    save_completeness : bool, optional
        chose to save the completeness of each candidate in the dataframe. The default is True.
    save_figure : bool, optional
        chose to save tong plots on HD. The default is True.
    num_of_chunks : int, optional
        number of chunk to split the targets. The default is None.

    Returns
    -------
    None.
    '''
    if MagBin_list is None: MagBin_list = DF.fk_completeness_df.index.get_level_values('magbin').unique()
    if filters_list is None: filters_list = DF.fk_completeness_df.index.get_level_values('filter').unique()
    if Nvisit_list is None: Nvisit_list = DF.fk_completeness_df.index.get_level_values('nvisit').unique()
    if Kmodes_list is None: Kmodes_list = DF.kmodes
    if path2savedir == None: path2savedir = './plots/'


    for filter in filters_list:
        if filter not in skip_filters:
            getLogger(__name__).info(f'Drawing filter {filter} mvs completeness plots')
            if len(MagBin_list) == 0: MagBin_list = DF.fk_completeness_df.index.get_level_values(
                'magbin').unique()
            mvs_completeness_plots(DF, filter=filter, path2savedir=path2savedir, MagBin_list=MagBin_list,
                                   Nvisit_list=Nvisit_list, avg_ids_list=avg_ids_list, Kmodes_list=Kmodes_list,
                                   title=title, fx=fx, fy=fy, fz=fz, ncolumns=ncolumns, xnew=xnew, ynew=ynew,
                                   ticks=ticks, show_IDs=show_IDs, save_completeness=save_completeness,
                                   save_figure=save_figure, showplot=showplot, suffix=suffix)


def update_candidates_photometry(DF, path2tile='./', avg_ids_list=[], label='data', aptype='4pixels', verbose=False, noBGsub=False,
                                 sigma=2.5, DF_fk=None, kill_plots=True, delta=3, skip_filters=[], sat_thr=np.inf,
                                 suffix=''):
    KLIP_label_dict = {'data': 'Kmode', 'crclean_data': 'crclean_Kmode'}
    if DF_fk == None: DF_fk = DF
    if len(avg_ids_list) == 0:
        avg_ids_list = DF.avg_candidates_df.avg_ids.unique()

    getLogger(__name__).info(f'Updating the candidates photometry. Loading a total of {len(avg_ids_list)} targets')

    for avg_ids in avg_ids_list:
        mvs_ids_list = DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids.isin(
            DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.avg_ids == avg_ids].mvs_ids.values)].mvs_ids.unique()
        if verbose:
            print('> Before:')
            display(DF.avg_candidates_df.loc[DF.avg_candidates_df.avg_ids == avg_ids])
            display(DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids.isin(mvs_ids_list)])
        zelno = 0
        for filter in DF.filters:
            if filter not in skip_filters:
                zpt = DF.mvs_targets_df[f'delta_{filter}'].unique()[0]
                ezpt = DF.mvs_targets_df[f'edelta_{filter}'].unique()[0]
                zelno += 1
                for mvs_ids in mvs_ids_list:
                    if DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids == mvs_ids, f'flag_{filter}'].values[
                        0] != 'rejected':
                        # print(DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,'%s_flag'%filter].values[0])
                        magbin = DF.mvs_targets_df.loc[
                            DF.mvs_targets_df.mvs_ids == mvs_ids, f'm_{filter}{suffix}'].values[0].astype(
                            int)
                        sep = \
                        DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids == mvs_ids, f'sep_{filter}'].values[
                            0]
                        mag = DF.mvs_candidates_df.loc[
                            DF.mvs_candidates_df.mvs_ids == mvs_ids, f'm_{filter}'].values[0]
                        x, y = DF.mvs_candidates_df.loc[
                            DF.mvs_candidates_df.mvs_ids == mvs_ids, [f'x_tile_{filter}',
                                                                        f'y_tile_{filter}']].values[0]
                        Kmode = DF.mvs_candidates_df.loc[
                            DF.mvs_candidates_df.mvs_ids == mvs_ids, f'kmode_{filter}'].astype(int).values[0]
                        exptime = \
                        DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids == mvs_ids, f'exptime_{filter}'].values[0]
                        if (not np.isnan(mag) and mag - magbin >= 0
                                and mag - magbin <= DF_fk.fk_candidates_df.index.get_level_values('dmag').astype(int).max()
                                and int(magbin) >= DF_fk.fk_candidates_df.index.get_level_values('magbin').astype(int).min()
                                and int(magbin) <= DF_fk.fk_candidates_df.index.get_level_values('magbin').astype(int).max()
                                and int(sep) >= DF_fk.fk_candidates_df.index.get_level_values('sep').astype(int).min()
                                and int(sep) <= DF_fk.fk_candidates_df.index.get_level_values('sep').astype(int).max()):
                            thrpt, ethrpt = KLIP_throughput(DF_fk, sep, filter, int(magbin), int(mag - magbin), Kmode,
                                                            verbose=False)
                            path2tile += f'/mvs_tiles/{filter}/tile_ID{ID}.fits'
                            KDATA = Tile(x=x, y=y, tile_base=DF.tilebase, inst=DF.inst)
                            KDATA.load_tile(path2tile, ext='%s%s' % (KLIP_label_dict[label], Kmode), verbose=False,
                                            return_Datacube=False)

                            if not np.all(np.isnan(KDATA.data)):
                                counts, ecounts, Nsigma, Nap, mag, emag, spx, bpx, Sky, eSky, nSky, grow_corr = KLIP_aperture_photometry_handler(
                                    DF, mvs_ids, filter, x=x, y=y, data=KDATA.data, zpt=zpt, ezpt=ezpt, aptype=aptype,
                                    noBGsub=False, sigma=sigma, kill_plots=True, Python_origin=True, delta=delta,
                                    sat_thr=sat_thr, exptime=exptime, thrpt=thrpt[0], ethrpt=ethrpt[
                                        0])  # (DF,mvs_ids,filter,x=x,y=y,data=KDATA.data,zpt=zpt,ezpt=ezpt,aptype=aptype,noBGsub=False,sigma=sigma,kill_plots=True,Python_origin=True,delta=delta,sat_thr=sat_thr,exptime=exptime,thrpt=thrpt[0],ethrpt=ethrpt[0],candidate=True)
                            else:
                                counts, ecounts, mag, emag = [np.nan, np.nan, np.nan, np.nan, np.nan]
                            DF.mvs_candidates_df.loc[
                                DF.mvs_candidates_df.mvs_ids == mvs_ids, [f'count_{filter}',
                                                                          f'ecounts_{filter}',
                                                                          f'nsigma_{filter}',
                                                                          f'm_{filter}',
                                                                          f'e_{filter}']] = [counts, ecounts,
                                                                                                     Nsigma, mag, emag]
                        else:
                            columns = DF.mvs_candidates_df.columns[
                                DF.mvs_candidates_df.columns.str.contains(filter[1:4])]
                            DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids == mvs_ids, columns] = np.nan
                            DF.mvs_candidates_df.loc[
                                DF.mvs_candidates_df.mvs_ids == mvs_ids, f'flag_{filter}'] = 'rejected'
                    else:
                        columns = DF.mvs_candidates_df.columns[
                            DF.mvs_candidates_df.columns.str.contains(filter[1:4])]
                        DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids == mvs_ids, columns] = np.nan
                        DF.mvs_candidates_df.loc[
                            DF.mvs_candidates_df.mvs_ids == mvs_ids, f'flag_{filter}'] = 'rejected'

                counts = DF.mvs_candidates_df.loc[
                    DF.mvs_candidates_df.mvs_ids.isin(mvs_ids_list), f'counts_{filter}'].values
                counts = counts.astype(float)
                ecounts = DF.mvs_candidates_df.loc[
                    DF.mvs_candidates_df.mvs_ids.isin(mvs_ids_list), f'ecounts_{filter}'].values
                ecounts = ecounts.astype(float)
                Mask = ~(np.isnan(counts))

                if len(counts[Mask]) > 0:
                    _, _, _, Mask = print_mean_median_and_std_sigmacut(counts, verbose=False, sigma=sigma)
                    counts = counts[Mask]
                    ecounts = ecounts[Mask]
                    c, ec = np.average(counts, weights=1 / ecounts ** 2, axis=0, returned=True)

                    mags = DF.mvs_candidates_df.loc[
                        DF.mvs_candidates_df.mvs_ids.isin(mvs_ids_list), f'm_{filter}'].values
                    emags = DF.mvs_candidates_df.loc[
                        DF.mvs_candidates_df.mvs_ids.isin(mvs_ids_list), f'e_{filter}'].values
                    mags = mags.astype(float)
                    emags = emags.astype(float)

                    mags = mags[Mask]
                    emags = emags[Mask]
                    m, w = np.average(mags, weights=1 / emags ** 2, axis=0, returned=True)
                    em = 1 / np.sqrt(w)
                else:
                    m, em = [np.nan, np.nan]
                sep = np.nanmean(DF.mvs_candidates_df.loc[
                                     DF.mvs_candidates_df.mvs_ids.isin(mvs_ids_list), [f'sep_{filter}' for filter in
                                                                                         DF.filters]].values)
                DF.avg_candidates_df.loc[DF.avg_candidates_df.avg_ids == avg_ids, f'm_{filter}'] = np.round(m,
                                                                                                                      3)
                DF.avg_candidates_df.loc[DF.avg_candidates_df.avg_ids == avg_ids, f'e_{filter}'] = np.round(
                    em, 3)
                DF.avg_candidates_df.loc[DF.avg_candidates_df.avg_ids == avg_ids, 'sep'] = np.round(sep, 3)
                dmag = \
                DF.avg_candidates_df.loc[DF.avg_candidates_df.avg_ids == avg_ids, f'm_{filter}'].values[0] - \
                DF.avg_targets_df.loc[DF.avg_targets_df.avg_ids == avg_ids, f'm_{filter}'].values[0]
                if dmag < 0:
                    columns = DF.avg_candidates_df.columns[DF.avg_candidates_df.columns.str.contains(filter[1:4])]
                    DF.avg_candidates_df.loc[DF.avg_candidates_df.avg_ids == avg_ids, columns] = np.nan
        if verbose:
            print('> After:')
            display(DF.avg_candidates_df.loc[DF.avg_candidates_df.avg_ids == avg_ids])
            display(DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids.isin(mvs_ids_list)])

    bad_mvs_ids = []
    bad_avg_ids = []
    for avg_ids in DF.avg_candidates_df.avg_ids.unique():
        mvs_ids_list = DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.avg_ids.isin([avg_ids])].mvs_ids.unique()
        if DF.mvs_candidates_df.loc[
            DF.mvs_candidates_df.mvs_ids.isin(mvs_ids_list), [f'flag_{filter}' for filter in
                                                                DF.filters]].apply(
                lambda x: x.str.contains('rejected', case=False)).all(axis=1).all(axis=0):
            bad_mvs_ids.extend(mvs_ids_list)
            bad_avg_ids.append(avg_ids)
    DF.mvs_candidates_df = DF.mvs_candidates_df.loc[~DF.mvs_candidates_df.mvs_ids.isin(bad_mvs_ids)].reset_index(
        drop=True)
    DF.avg_candidates_df = DF.avg_candidates_df.loc[~DF.avg_candidates_df.avg_ids.isin(bad_avg_ids)].reset_index(
        drop=True)
    return(DF)