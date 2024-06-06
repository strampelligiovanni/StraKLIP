from utils_photometry import aperture_photometry_handler,KLIP_aperture_photometry_handler,photometry_AP
from photometry import Detection,flux_converter
import pandas as pd
from tiles import Tile
from fake_star import Fake_Star
from utils_tile import perform_PSF_subtraction
from random import sample,randint,uniform
from ancillary import PointsInCircum,round2closerint

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
    print('> ', filter)
    if magbin_list == None: magbin_list = DF.fk_candidates_df.loc[filter].index.get_level_values(
        'magbin').unique()
    dmag_list = DF.fk_candidates_df.loc[filter].index.get_level_values('dmag').unique()
    sep_list = DF.fk_candidates_df.loc[filter].index.get_level_values('sep').unique()
    values_list = []

    if parallel_runs:
        for Nvisit in Nvisit_range:
            workers, chunksize, ntarget = parallelization_package(workers, len(magbin_list), chunksize=chunksize)
            print('Testing nvisit %i. Working on magbin:' % Nvisit)
            FP_sel = FP_lim ** (1 / (Nvisit * len(DF.filters_list)))
            if DF_fk == None: DF_fk = DF
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for out_list in tqdm(executor.map(task_completeness_from_fakes_infos, repeat(DF), repeat(filter),
                                                  repeat(Nvisit), magbin_list, repeat(dmag_list), repeat(sep_list),
                                                  repeat(FP_sel), repeat(AUC_lim), repeat(skip_filters),
                                                  chunksize=chunksize)):
                    values_list.extend(out_list)

    else:
        for Nvisit in Nvisit_range:
            print('Testing nvisit %i. Working on magbin:' % Nvisit)
            FP_sel = FP_lim ** (1 / (Nvisit * len(DF.filters_list)))
            if DF_fk == None: DF_fk = DF
            for magbin in tqdm(magbin_list):
                out_list = task_completeness_from_fakes_infos(DF, filter, Nvisit, magbin, dmag_list, sep_list,
                                                              FP_sel, AUC_lim, skip_filters)
                values_list.extend(out_list)

    print('Writing entry in dataframe:')
    for elno in tqdm(range(len(values_list))):
        DF.fk_completeness_df.loc[
            (filter, values_list[elno][0], values_list[elno][1], values_list[elno][2], values_list[elno][3]), [
                'ratio_Kmode%s' % values_list[elno][-1]]] = values_list[elno][4]
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
            for Kmode in DF.Kmodes_list:
                if filter not in skip_filters:
                    TPnsigma_inj_list = DF.fk_candidates_df.loc[
                        (filter, magbin, dmag, sep), ['Nsigma_Kmode%i' % (Kmode)]].values.ravel()
                    FPnsigma_list = DF.fk_targets_df.loc[
                        (filter, magbin, dmag, sep), ['Nsigma_Kmode%i' % (Kmode)]].values.ravel()

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
                    print(AUC, AUC_lim)
                    print([Nvisit, magbin, dmag, sep, Ratio_median, Kmode])
                    sys.exit()

    # sys.exit()
    return (out_list)
