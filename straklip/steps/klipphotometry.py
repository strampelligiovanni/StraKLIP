import numpy as np
import sys, os,
sys.path.append('/')
import numpy as np
from stralog import getLogger
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from ancillary import parallelization_package
from astropy.io import fits
from pathlib import Path
from tiles import Tile

def task_mvs_tiles_and_photometry(DF, fitsname, ids_list, filter, use_xy_SN, use_xy_m, use_xy_cen, xy_shift_list,
                                  xy_dmax, bpx_list, spx_list, legend, showplot, overwrite, verbose, Python_origin,
                                  cr_remove, la_cr_remove, cr_radius, kill_plots, ee_df, zpt, radius_in, radius1_in,
                                  radius2_in, sat_thr, grow_curves, r_in, p, gstep, multiply_by_exptime,
                                  multiply_by_gain, multiply_by_PAM):
    '''
    parallelized task for the update_mvs_tiles.

    Parameters
    ----------
    fitsname : str
        name of the image.
    ids_list : list
        list of ids corresponding to sources in the image.
    filter : str
        filter name.
    use_xy_SN : bool
        choose to use image with better SN to recenter the tile.
    use_xy_m : bool
        choose to use maximum in image to recenter the tile.
    use_xy_cen : bool
        choose to use centroid in image to recenter the tile.
    xy_dmax : int
        maximum distance from original coordinates to look for maximum.
    bpx_list: list
        list of values to flag as bad pixels using the data quality image (coud be real bad pixels, hot/warm pixels, cosmic rays etc.)
    spx_list: list
        list of values to flag as saturated pixels using the data quality image
    legend : bool, optional
        choose to show legends in plots. The default is False.
    showplot : bool, optional
        choose to show plots. The default is False.
    verbose : bool, optional
        choose to show prints. The default is False.
    Python_origin : bool
        Choose to specify the origin of the xy input coordinates. For exmaple python array star counting from 0,
        so a position obtained on a python image will have 0 as first pixel.
        On the other hand, normal catalogs start counting from 1 (see coordinate on ds9 for example)
        so we need to subtract 1 to make them compatible when we use those coordinates on python
        The default is True
    cr_remove : bool, optional
        choose to apply cosmic ray removal. The default is False.
    la_cr_remove : bool, optional
        choose to apply L.A. cosmic ray removal. The default is False.
    cr_radius : int, optional
        minimum distance from center where to not apply the cosmic ray filter. The default is 3.
    close : bool, optional
        choose to close plot istances. The default is True.
    kill_plots:
        choose to kill all plots created. The default is False.
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
    None.

    '''
    # out=[]
    hdul = fits.open(Path(DF.fits_path / str(fitsname + '.fits')))
    if len(hdul) >= 4:
        phot = []
        for id in ids_list:
            path2tile = '%s/%s/%s/%s/mvs_tiles/%s/tile_ID%i.fits' % (
            path2data, DF.project, DF.target, DF.inst, filter, id)
            type_flag = DF.avg_targets_df.loc[DF.avg_targets_df.avg_ids == DF.crossmatch_ids_df.loc[
                DF.crossmatch_ids_df.mvs_ids == id].avg_ids.unique()[0]].type.values[0]
            x = DF.mvs_targets_df.loc[(DF.mvs_targets_df['%s_%s' % (filter, DF.fits_ext)] == fitsname) & (
                        DF.mvs_targets_df.mvs_ids == id), 'x%s' % filter[1:4]].values[0]
            y = DF.mvs_targets_df.loc[(DF.mvs_targets_df['%s_%s' % (filter, DF.fits_ext)] == fitsname) & (
                        DF.mvs_targets_df.mvs_ids == id), 'y%s' % filter[1:4]].values[0]
            ext = int(DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids == id].ext.values[0])
            if x % 0.5 == 0: x -= 0.001
            if y % 0.5 == 0: y -= 0.001

            if not np.isnan(x) and not np.isnan(y):
                flag = 'good_target'
                if overwrite or not os.path.isfile(path2tile):
                    exptime = DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids == id, '%s_exptime' % filter].values[
                        0]
                    SCI = hdul[ext].data.copy()
                    if multiply_by_exptime:
                        SCI *= exptime

                    if multiply_by_gain:
                        SCI *= DF.gain

                    if multiply_by_PAM:
                        path2PAM = '%s/%s/%s/%s/PAM' % (DF.path2fitsdir, DF.project, DF.target, DF.inst)
                        phdul = fits.open(path2PAM + '/' + str(DF.PAMdict[0][ext] + '.fits'))
                        try:
                            PAM = phdul[1].data
                        except:
                            PAM = phdul[0].data
                        SCI *= PAM

                    ERR = hdul[ext + 1].data
                    DQ = hdul[ext + 2].data

                    if type_flag == 2 and (use_xy_cen == True or use_xy_SN == True):
                        flag = 'unresolved_double'
                        xy_m = True
                        xy_cen = False
                    else:
                        xy_cen = use_xy_cen
                        xy_m = use_xy_m

                    if xy_cen:
                        if cr_remove or la_cr_remove:
                            DQDATA = Tile(data=DQ, x=x, y=y, tile_base=DF.tile_base, delta=6, inst=DF.inst,
                                          Python_origin=Python_origin)
                            DQDATA.mk_tile(pad_data=True, legend=legend, showplot=False, verbose=verbose,
                                           title='shiftedDQ', kill_plots=True, cbar=True)

                            IDATA = Tile(data=SCI, x=x, y=y, tile_base=DF.tile_base, delta=6, dqdata=DQDATA.data,
                                         inst=DF.inst, Python_origin=Python_origin)
                            IDATA.mk_tile(xy_m=True, pad_data=True, legend=legend, showplot=False, verbose=verbose,
                                          title='CRcleanSCI', kill_plots=True, cr_remove=cr_remove,
                                          la_cr_remove=la_cr_remove, cr_radius=cr_radius, cbar=True)
                        else:

                            IDATA = Tile(data=SCI, x=x, y=y, tile_base=DF.tile_base, delta=6, inst=DF.inst,
                                         Python_origin=Python_origin)
                            IDATA.mk_tile(pad_data=True, legend=False, showplot=False, verbose=False, xy_cen=True,
                                          xy_dmax=xy_dmax, title='OrigSCI', kill_plots=True, cbar=True)

                        deltax = IDATA.x_cen - (IDATA.tile_base - 1) / 2
                        deltay = IDATA.y_cen - (IDATA.tile_base - 1) / 2

                    elif xy_m:
                        if cr_remove or la_cr_remove:
                            DQDATA = Tile(data=DQ, x=x, y=y, tile_base=DF.tile_base, delta=6, inst=DF.inst,
                                          Python_origin=Python_origin)
                            DQDATA.mk_tile(pad_data=True, legend=legend, showplot=False, verbose=verbose,
                                           title='shiftedDQ', kill_plots=True, cbar=True)
                            IDATA = Tile(data=SCI, x=x, y=y, tile_base=DF.tile_base, delta=6, dqdata=DQDATA.data,
                                         inst=DF.inst, Python_origin=Python_origin)
                            IDATA.mk_tile(xy_m=True, pad_data=True, legend=legend, showplot=False, verbose=verbose,
                                          title='CRcleanSCI', kill_plots=True, cr_remove=cr_remove,
                                          la_cr_remove=la_cr_remove, cr_radius=cr_radius, cbar=True)
                        else:
                            IDATA = Tile(data=SCI, x=x, y=y, tile_base=DF.tile_base, delta=6, inst=DF.inst,
                                         Python_origin=Python_origin)
                            IDATA.mk_tile(pad_data=True, legend=False, showplot=False, verbose=False, xy_m=True,
                                          xy_dmax=xy_dmax, title='OrigSCI', kill_plots=True, cbar=True)

                        deltax = IDATA.x_m - (IDATA.tile_base - 1) / 2
                        deltay = IDATA.y_m - (IDATA.tile_base - 1) / 2

                    else:
                        if cr_remove or la_cr_remove:
                            DQDATA = Tile(data=DQ, x=x, y=y, tile_base=DF.tile_base, delta=6, inst=DF.inst,
                                          Python_origin=Python_origin)
                            DQDATA.mk_tile(pad_data=True, legend=legend, showplot=False, verbose=verbose,
                                           title='shiftedDQ', kill_plots=True, cbar=True)

                            IDATA = Tile(data=SCI, x=x, y=y, tile_base=DF.tile_base, delta=6, dqdata=DQDATA.data,
                                         inst=DF.inst, Python_origin=Python_origin)
                            IDATA.mk_tile(xy_m=True, pad_data=True, legend=legend, showplot=False, verbose=verbose,
                                          title='CRcleanSCI', kill_plots=True, cr_remove=cr_remove,
                                          la_cr_remove=la_cr_remove, cr_radius=cr_radius, cbar=True)
                        else:

                            IDATA = Tile(data=SCI, x=x, y=y, tile_base=DF.tile_base, delta=6, inst=DF.inst,
                                         Python_origin=Python_origin)
                            IDATA.mk_tile(pad_data=True, legend=False, showplot=False, verbose=False, title='OrigSCI',
                                          kill_plots=True, cbar=True)
                        deltax = 0
                        deltay = 0

                    if len(xy_shift_list) > 0:
                        deltax += xy_shift_list[0]
                        deltay += xy_shift_list[1]

                    # making the tile datacube and save it
                    DF.mvs_targets_df.loc[(DF.mvs_targets_df['%s_%s' % (filter, DF.fits_ext)] == fitsname) & (
                                DF.mvs_targets_df.mvs_ids == id), ['x%s' % filter[1:4], 'y%s' % filter[1:4]]] = [
                        x + deltax, y + deltay]
                    x = DF.mvs_targets_df.loc[(DF.mvs_targets_df['%s_%s' % (filter, DF.fits_ext)] == fitsname) & (
                                DF.mvs_targets_df.mvs_ids == id), 'x%s' % filter[1:4]].values[0]  # -0.001
                    y = DF.mvs_targets_df.loc[(DF.mvs_targets_df['%s_%s' % (filter, DF.fits_ext)] == fitsname) & (
                                DF.mvs_targets_df.mvs_ids == id), 'y%s' % filter[1:4]].values[0]  # -0.001

                    DATA = Tile(data=SCI, x=x, y=y, tile_base=DF.tile_base, delta=6, inst=DF.inst,
                                Python_origin=Python_origin)
                    DATA.mk_tile(pad_data=True, legend=legend, showplot=False, verbose=verbose, title='shiftedSCI',
                                 kill_plots=True, cbar=True)
                    Datacube = DATA.append_tile(path2tile, Datacube=None, verbose=False, name='SCI',
                                                return_Datacube=True)

                    EDATA = Tile(data=ERR, x=x, y=y, tile_base=DF.tile_base, delta=6, inst=DF.inst,
                                 Python_origin=Python_origin)
                    EDATA.mk_tile(pad_data=True, legend=legend, showplot=False, verbose=verbose, title='shiftedERR',
                                  kill_plots=True, cbar=True)
                    Datacube = EDATA.append_tile(path2tile, Datacube=Datacube, verbose=False, name='ERR',
                                                 return_Datacube=True)

                    return_Datacube = False
                    if cr_remove or la_cr_remove: return_Datacube = True
                    DQDATA = Tile(data=DQ, x=x, y=y, tile_base=DF.tile_base, delta=6, inst=DF.inst,
                                  Python_origin=Python_origin)
                    DQDATA.mk_tile(pad_data=True, legend=legend, showplot=False, verbose=verbose, title='shiftedDQ',
                                   kill_plots=True, cbar=True)
                    Datacube = DQDATA.append_tile(path2tile, Datacube=Datacube, verbose=False, name='DQ',
                                                  return_Datacube=return_Datacube)

                    if cr_remove or la_cr_remove:
                        CRDATA = Tile(data=SCI, x=x, y=y, tile_base=DF.tile_base, delta=6, dqdata=DQDATA.data,
                                      inst=DF.inst, Python_origin=Python_origin)
                        CRDATA.mk_tile(pad_data=True, legend=legend, showplot=False, verbose=verbose,
                                       title='shiftedCRcleanSCI', kill_plots=True, cr_remove=cr_remove,
                                       la_cr_remove=la_cr_remove, cr_radius=cr_radius, cbar=True)
                        Datacube = CRDATA.append_tile(path2tile, Datacube=Datacube, verbose=False, name='CRcelanSCI',
                                                      return_Datacube=False)

                if not DF.skip_photometry:
                    phot.append(
                        mvs_aperture_photometry(DF, filter, ee_df, zpt, fitsname=fitsname, mvs_ids_list_in=[id],
                                                bpx_list=bpx_list, spx_list=spx_list, la_cr_remove=la_cr_remove,
                                                cr_radius=cr_radius, radius_in=radius_in, radius1_in=radius1_in,
                                                radius2_in=radius2_in, sat_thr=sat_thr, kill_plots=kill_plots,
                                                grow_curves=grow_curves, r_in=r_in, p=p, gstep=gstep, flag=flag,
                                                multiply_by_exptime=multiply_by_exptime,
                                                multiply_by_gain=multiply_by_gain, multiply_by_PAM=multiply_by_PAM))
                else:
                    bpx, spx = read_dq_from_tile(DF, path2tile, bpx_list=bpx_list, spx_list=spx_list)
                    phot.append(
                        [id, ext, np.nan, np.nan, np.nan, np.nan, np.nan, spx, bpx, np.nan, np.nan, np.nan, np.nan,
                         np.nan, np.nan, np.nan, flag])
            else:
                phot.append(
                    [id, ext, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, 'rejected'])

    else:
        phot = []
        for id in ids_list: phot.append(
            [id, ext, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
             np.nan, np.nan, 'rejected'])
    if not kill_plots:
        print('> m%s e%s spx%s bpx%s %s_r %s_rsky1 %s_rsky2' % (
        filter[1:4], filter[1:4], filter[1:4], filter[1:4], filter, filter, filter))
        print('> ', phot)
    return (phot)

def mk_mvs_tiles_and_photometry(DF, filter, mvs_ids_test_list=[], overwrite=True, xy_SN=True, xy_m=False,
                                xy_cen=False, xy_shift_list=[], xy_dmax=3, bpx_list=[], spx_list=[], legend=False,
                                showplot=False, showplot_final=False, verbose=False, workers=None, Python_origin=True,
                                parallel_runs=True, cr_remove=False, la_cr_remove=False, cr_radius=3, kill_plots=False,
                                chunksize=None, ee_df=None, zpt=0, radius_in=10, radius1_in=10, radius2_in=15,
                                sat_thr=np.inf, grow_curves=True, r_in=1, p=100, gstep=0.1, multiply_by_exptime=False,
                                multiply_by_gain=False, multiply_by_PAM=False):
    '''
    update the multi-visits tile dataframe with the tiles for each source

    Parameters
    ----------
    mvs_ids_test_list : list, optional
        list of multivisits ids to test. The default is [].
    overwrite: bool, optional
        if False, skip IDs that already have tiles in directory. Default is True.
    xy_SN : bool
        choose to use image with better SN to recenter the tile.
    xy_m : bool
        choose to use maximum in image to recenter the tile.
    xy_cen : bool
        choose to use centroid in image to recenter the tile.
    xy_dmax : int
        maximum distance from original coordinates to look for maximum.
    bpx_list: list
        list of values to flag as bad pixels using the data quality image (coud be real bad pixels, hot/warm pixels, cosmic rays etc.)
    spx_list: list
        list of values to flag as saturated pixels using the data quality image
    legend : bool, optional
        choose to show legends in plots. The default is False.
    showplot : bool, optional
        choose to show plots. The default is False.
    showplot_final : bool, optional
        choose to show fianl plot after all manipolatins has been applyed.
        The default is False.
    verbose : bool, optional
        choose to show prints. The default is False.
    workers : int, optional
        number of workers to split the work accross multiple CPUs. The default is 3.
    Python_origin : bool, optional
        Choose to specify the origin of the xy input coordinates. For exmaple python array star counting from 0,
        so a position obtained on a python image will have 0 as first pixel.
        On the other hand, normal catalogs start counting from 1 (see coordinate on ds9 for example)
        so we need to subtract 1 to make them compatible when we use those coordinates on python
        The default is True
    parallel_runs: bool, optional
        Choose to to split the workload over different CPUs. The default is True
    la_cr_remove : bool, optional
        choose to apply L.A. cosmic ray removal. The default is False.
    cr_radius : int, optional
        minimum distance from center where to not apply the cosmic ray filter. The default is 3.
    close : bool, optional
        choose to close plot istances. The default is True.
    kill_plots:
        choose to kill all plots created. The default is False.
    num_of_chunks : int, optional
        number of chunk to split the targets. The default is None.
    chunksize : int, optional
        size of each chunk for the parallelization process. The default is None.
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
    None.

    '''
    # if __name__ == 'utils_straklip':
    q = np.where(filter == np.array(DF.filters_list))[0][0]

    ids_list_of_lists = []
    fitsname_list = []
    getLogger(__name__).info("Working on %s" % filter)
    # mk_dir('%s/%s/%s/%s/mvs_tiles/%s' % (path2data, DF.project, DF.target, DF.inst, filter), verbose=True)
    fits_dict = {}
    skip_IDs_list_list = []

    if len(mvs_ids_test_list) != 0:
        fitsname_test_list = DF.mvs_targets_df.loc[
            DF.mvs_targets_df.mvs_ids.isin(mvs_ids_test_list) & ~DF.mvs_targets_df.mvs_ids.isin(
                skip_IDs_list_list), '%s_%s' % (filter, DF.fits_ext)].tolist()
        for index, row in DF.mvs_targets_df.loc[DF.mvs_targets_df['%s_%s' % (filter, DF.fits_ext)].isin(
                fitsname_test_list) & DF.mvs_targets_df.mvs_ids.isin(
                mvs_ids_test_list) & ~DF.mvs_targets_df.mvs_ids.isin(skip_IDs_list_list)].groupby(
                '%s_%s' % (filter, DF.fits_ext)):
            fits_dict[index] = row.mvs_ids.tolist()
    else:
        fitsname_test_list = DF.mvs_targets_df.loc[
            ~DF.mvs_targets_df.mvs_ids.isin(skip_IDs_list_list), '%s_%s' % (
            filter, DF.fits_ext)].unique().tolist()
        for index, row in DF.mvs_targets_df.loc[DF.mvs_targets_df['%s_%s' % (filter, DF.fits_ext)].isin(
                fitsname_test_list) & ~DF.mvs_targets_df.mvs_ids.isin(skip_IDs_list_list)].groupby(
                '%s_%s' % (filter, DF.fits_ext)):
            fits_dict[index] = row.mvs_ids.tolist()

    fits_dict = (dict(sorted(fits_dict.items(), key=lambda item: item[1])))
    ids_list_of_lists = list(fits_dict.values())
    fitsname_list = list(fits_dict.keys())
    if parallel_runs:
        workers, chunksize, ntarget = parallelization_package(workers, len(fitsname_list), chunksize=chunksize)
        getLogger(__name__).info('Loading a total of %i images' % ntarget)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for phot in tqdm(
                    executor.map(task_mvs_tiles_and_photometry, repeat(DF), fitsname_list, ids_list_of_lists,
                                 repeat(filter), repeat(xy_SN), repeat(xy_m), repeat(xy_cen), repeat(xy_shift_list),
                                 repeat(xy_dmax), repeat(bpx_list), repeat(spx_list), repeat(legend),
                                 repeat(showplot), repeat(overwrite), repeat(verbose), repeat(Python_origin),
                                 repeat(cr_remove), repeat(la_cr_remove), repeat(cr_radius), repeat(kill_plots),
                                 repeat(ee_df), repeat(zpt[q]), repeat(radius_in), repeat(radius1_in),
                                 repeat(radius2_in), repeat(sat_thr), repeat(grow_curves), repeat(r_in), repeat(p),
                                 repeat(gstep), repeat(multiply_by_exptime), repeat(multiply_by_gain),
                                 repeat(multiply_by_PAM), chunksize=chunksize)):
                phot = np.array(phot)
                for elno in range(len(phot)):
                    sel = (DF.mvs_targets_df.mvs_ids == phot[elno, 0].astype(float)) & (
                                DF.mvs_targets_df.ext == phot[elno, 1].astype(float))
                    DF.mvs_targets_df.loc[
                        sel, ['counts%s' % filter[1:4], 'ecounts%s' % filter[1:4], 'Nap%s' % filter[1:4],
                              'm%s' % filter[1:4], 'e%s' % filter[1:4], 'spx%s' % filter[1:4],
                              'bpx%s' % filter[1:4], '%s_r' % filter, '%s_rsky1' % filter, '%s_rsky2' % filter,
                              'sky%s' % filter[1:4], 'esky%s' % filter[1:4], 'nsky%s' % filter[1:4],
                              'grow_corr%s' % filter[1:4]]] = phot[elno, 2:-1].astype(float)
                    DF.mvs_targets_df.loc[sel, ['%s_flag' % filter]] = phot[elno, -1:]

    else:
        for elno in tqdm(range(len(fitsname_list))):
            if len(xy_shift_list) > 0:
                w = np.where(np.array(fitsname_list)[elno] == np.array(fitsname_test_list))[0][0]
                use_xy_shift_list = xy_shift_list[w]
            else:
                use_xy_shift_list = []
            phot = task_mvs_tiles_and_photometry(DF, fitsname_list[elno], ids_list_of_lists[elno], filter, xy_SN,
                                                 xy_m, xy_cen, use_xy_shift_list, xy_dmax, bpx_list, spx_list,
                                                 legend, showplot, overwrite, verbose, Python_origin, cr_remove,
                                                 la_cr_remove, cr_radius, kill_plots, ee_df, zpt[q], radius_in,
                                                 radius1_in, radius2_in, sat_thr, grow_curves, r_in, p, gstep,
                                                 multiply_by_exptime, multiply_by_gain, multiply_by_PAM)
            phot = np.array(phot)
            for elno in range(len(phot)):
                sel = (DF.mvs_targets_df.mvs_ids == phot[elno, 0].astype(float)) & (
                            DF.mvs_targets_df.ext == phot[elno, 1].astype(float))
                DF.mvs_targets_df.loc[
                    sel, ['counts%s' % filter[1:4], 'ecounts%s' % filter[1:4], 'Nap%s' % filter[1:4],
                          'm%s' % filter[1:4], 'e%s' % filter[1:4], 'spx%s' % filter[1:4], 'bpx%s' % filter[1:4],
                          '%s_r' % filter, '%s_rsky1' % filter, '%s_rsky2' % filter, 'sky%s' % filter[1:4],
                          'esky%s' % filter[1:4], 'nsky%s' % filter[1:4], 'grow_corr%s' % filter[1:4]]] = phot[elno,
                                                                                                          2:-1].astype(
                    float)
                DF.mvs_targets_df.loc[sel, ['%s_flag' % filter]] = phot[elno, -1:]


def update_median_candidates_tile(DF, avg_ids_list, column_name='Kmode', workers=None, zfactor=10, alignment_box=3,
                                  parallel_runs=True, chunksize=None, label='data', kill_plots=True, skip_filters=[]):
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
    # if __name__ == 'utils_straklip':
    getLogger(__name__).info('Loading a total of %i images' % len(avg_ids_list))
    for filter in DF.filters_list:
        if filter not in skip_filters:
            getLogger(__name__).info('Working in %s: ' % filter)
            if parallel_runs:
                workers, chunksize, ntarget = parallelization_package(workers, len(avg_ids_list),
                                                                      chunksize=chunksize)
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    for _ in tqdm(
                            executor.map(task_median_candidate_infos, repeat(DF), avg_ids_list, repeat(filter),
                                         repeat(column_name), repeat(zfactor), repeat(alignment_box), repeat(label),
                                         chunksize=chunksize)):
                        pass
            else:
                for id in tqdm(avg_ids_list):
                    task_median_candidate_infos(DF, id, filter, column_name, zfactor, alignment_box, label)


def update_candidates(DF, avg_ids_list=[], suffix='', d=1., skip_filters='F658N', aptype='4pixels', verbose=False,
                      workers=None, noBGsub=False, sigma=2.5, min_mag_list=[], max_mag_list=[], DF_fk=None,
                      parallel_runs=True, update_header=True, chunksize=None, label='data', kill_plots=True, delta=3,
                      radius=3, sat_thr=np.inf, mkd=2, mad=0.1):
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
    DF.header_df.loc['mad', 'Values'] = mad
    DF.header_df.loc['mkd', 'Values'] = mkd

    if update_header:
        update_header_photometry(DF, suffix=suffix, skip_filters=skip_filters, aptype=aptype, verbose=False,
                                 workers=workers, sigma=sigma, min_mag_list=min_mag_list, max_mag_list=max_mag_list,
                                 DF_fk=DF_fk, parallel_runs=parallel_runs, chunksize=chunksize, label=label,
                                 delta=delta, sat_thr=sat_thr)

    else:
        if verbose: display(DF.header_df)

    if parallel_runs:
        getLogger(__name__).info('Working on the candidates')
        workers, chunksize, ntarget = parallelization_package(workers, len(avg_ids_list), chunksize=chunksize)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for Kmode_final, candidate_df in tqdm(
                    executor.map(task_mvs_candidates_infos, repeat(DF), avg_ids_list, repeat(d), repeat(skip_filters),
                                 repeat(aptype), repeat(verbose), repeat(False), repeat(3), repeat(DF_fk),
                                 repeat(label), repeat(kill_plots), repeat(delta), repeat(radius), repeat(sat_thr),
                                 repeat(mkd), repeat(mad), chunksize=chunksize)):
                if Kmode_final != None: update_candidates_with_detection(DF, candidate_df, Kmode_final, verbose)

    else:
        for avg_id in tqdm(avg_ids_list):
            try:
                Kmode_final, candidate_df = task_mvs_candidates_infos(DF, avg_id, d, skip_filters, aptype, verbose,
                                                                      False, 3, DF_fk, label, kill_plots, delta, radius,
                                                                      sat_thr, mkd, mad)
                if Kmode_final != None: update_candidates_with_detection(DF, candidate_df, Kmode_final, verbose)
            except:
                raise ValueError('Something wrong with avg_id %s, Please check' % avg_id)

    getLogger(__name__).info('> Pruning the mvs_candidate_df:')
    mvs_bad_ids_list = []
    for avg_ids in tqdm(DF.avg_candidates_df.avg_ids):
        mvs_ids_list = DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.avg_ids == avg_ids].mvs_ids.unique()

        sel_mvs_ids = (DF.mvs_candidates_df.mvs_ids.isin(mvs_ids_list))
        sel_flags = (('good_candidate' == DF.mvs_candidates_df[['%s_flag' % i for i in DF.filters_list]]).astype(
            int).sum(axis=1) < 1)

        mvs_bad_ids = DF.mvs_candidates_df.loc[sel_mvs_ids & sel_flags].mvs_ids.unique()
        mvs_bad_ids_list.extend(mvs_bad_ids)

    DF.mvs_candidates_df = DF.mvs_candidates_df.loc[
        ~DF.mvs_candidates_df.mvs_ids.isin(mvs_bad_ids_list)].reset_index(drop=True)
    getLogger(__name__).info('> Finishing up the candidate dfs:')
    selected_avg_ids = DF.crossmatch_ids_df.loc[
        DF.crossmatch_ids_df.mvs_ids.isin(DF.mvs_candidates_df.mvs_ids)].avg_ids.unique()
    DF.avg_candidates_df = DF.avg_candidates_df.loc[
        DF.avg_candidates_df.avg_ids.isin(selected_avg_ids)].reset_index(drop=True)
    for avg_ids in tqdm(DF.avg_candidates_df.avg_ids):
        mvs_ids_list = DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.avg_ids == avg_ids].mvs_ids.unique()
        DF.avg_candidates_df.loc[DF.avg_candidates_df.avg_ids == avg_ids, 'mKmode'] = np.nanmedian(
            DF.mvs_candidates_df.loc[
                DF.mvs_candidates_df.mvs_ids.isin(mvs_ids_list), ['%s_Kmode' % filter for filter in
                                                                    DF.filters_list]].values)
        DF.avg_candidates_df.loc[DF.avg_candidates_df.avg_ids == avg_ids, 'sep'] = np.nanmean(
            DF.mvs_candidates_df.loc[
                DF.mvs_candidates_df.mvs_ids.isin(mvs_ids_list), ['%s_sep' % filter for filter in
                                                                    DF.filters_list]].values)

        for filter in DF.filters_list:
            if filter not in skip_filters and not \
            DF.avg_candidates_df.loc[DF.avg_candidates_df.avg_ids == avg_ids, 'm%s' % filter[1:4]].isna().values[0]:
                DF.avg_candidates_df.loc[
                    DF.avg_candidates_df.avg_ids == avg_ids, 'Nsigma%s' % filter[1:4]] = np.nanmedian(
                    DF.mvs_candidates_df.loc[
                        DF.mvs_candidates_df.mvs_ids.isin(mvs_ids_list), '%s_Nsigma' % filter].values)
                try:
                    DF.avg_candidates_df.loc[(DF.avg_candidates_df.avg_ids == avg_ids), 'MagBin%s' % filter[1:4]] = \
                    DF.avg_targets_df.loc[
                        (DF.avg_targets_df.avg_ids == avg_ids), 'm%s%s' % (filter[1:4], suffix)].values[0].astype(int)
                except:
                    DF.avg_candidates_df[
                        (DF.avg_candidates_df.avg_ids == avg_ids), 'MagBin%s' % filter[1:4]] = np.nan
    display(DF.avg_candidates_df)
def update_candidate_dataframe(DF, avg_ids_list=[], suffix='', verbose=False, workers=None, min_mag_list=[],
                               max_mag_list=[], aptype='4pixels', no_median_tile_update=False, zfactor=10,
                               alignment_box=3, parallel_runs=True, update_header=True, chunksize=None, label='data',
                               kill_plots=True, delta=3, radius=3, skip_filters=[], sat_thr=np.inf, mkd=2, mad=0.2,
                               PSF_sub_flags='good|unresolved'):
    '''
    This is a wrapper for the updating of candidates dataframe and the median tile candidate dataframe

    Parameters
    ----------
    avg_ids_list : list, optional
        list of average ids to test. The default is [].
    suffix: str, optional
        suffix to append to mag label. For example, if original photometry is present in the catalog, it canbe use with suffix='_o'.
        Default is ''.
    verbose : bool, optional
        choose to show prints.
    workers : TYPE, optional
        DESCRIPTION. The default is None.
    min_mag_list : TYPE, optional
        DESCRIPTION. The default is [].
    max_mag_list : TYPE, optional
        DESCRIPTION. The default is [].
    aptype : (circular,square,4pixels), optional
        defin the aperture type to use during aperture photometry.
        The default is '4pixels'.
    sigmaclip : TYPE, optional
        DESCRIPTION. The default is True.
    noBGsub : TYPE, optional
        DESCRIPTION. The default is False.
    sigma : TYPE, optional
        DESCRIPTION. The default is 2.5.
    no_median_tile_update : bool, optional
        skip update_median_candidates_tile. The default is False.
    zfactor : bool, optional
        zoom factor to use to oversample the image during the allignment process. The default is 10.
    alignment_box : bool, optional
        half base of the square box centered at the center of the tile employed by the allignment process to allign the images.
        The box is constructed as alignment_box-x,x+alignment_box and alignment_box-y,y+alignment_box, where x,y are the
        coordinate of the center of the tile.
        The default is 2.
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
    getLogger(__name__).info('> Updating the candidates dataframe')
    if len(avg_ids_list) == 0:
        mvs_ids_list = DF.mvs_targets_df.loc[(
            DF.mvs_targets_df[['%s_flag' % (filter) for filter in DF.filters_list]].apply(
                lambda x: x.str.contains(PSF_sub_flags, case=False)).any(axis=1))].mvs_ids.unique()
        avg_ids_list_in = DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.mvs_ids.isin(mvs_ids_list)].avg_ids.unique()
        # avg_ids_list_in=DF.avg_candidates_df.avg_ids.unique()
    else:
        avg_ids_list_in = []
        for avg_id in avg_ids_list:
            if avg_id in DF.avg_candidates_df.avg_ids.unique():
                avg_ids_list_in.append(avg_id)

    update_candidates(DF, avg_ids_list=avg_ids_list_in, suffix=suffix, verbose=verbose, aptype=aptype,
                      min_mag_list=min_mag_list, max_mag_list=max_mag_list, workers=workers,
                      parallel_runs=parallel_runs, update_header=update_header, label=label, kill_plots=kill_plots,
                      delta=delta, radius=radius, skip_filters=skip_filters, sat_thr=sat_thr, mkd=mkd, mad=mad)
    if not no_median_tile_update:
        getLogger(__name__).info('> Updating the median tiles for the candidates')
        # avg_ids_list_in=DF.avg_candidates_df.avg_ids.unique()
        update_median_candidates_tile(DF, avg_ids_list=avg_ids_list_in, workers=workers, zfactor=zfactor,
                                      alignment_box=alignment_box, parallel_runs=parallel_runs, chunksize=chunksize,

                                      label=label, kill_plots=kill_plots, skip_filters=skip_filters)


def run(packet):
    DF = packet['DF']
    dataset = packet['dataset']

    update_candidate_dataframe(DF,
                               avg_ids_list=dataset.pipe_cfg.klipphotometry['avg_ids_list'],
                               suffix=dataset.pipe_cfg.klipphotometry['suffix'],
                               verbose=dataset.pipe_cfg.klipphotometry['verbose'],
                               workers=dataset.pipe_cfg.ncpu,
                               min_mag_list=dataset.pipe_cfg.klipphotometry['min_mag_list'],
                               max_mag_list=dataset.pipe_cfg.klipphotometry['max_mag_list'],
                               aptype=dataset.pipe_cfg.klipphotometry['aptype'],
                               no_median_tile_update=dataset.pipe_cfg.klipphotometry['no_median_tile_update'],
                               zfactor=dataset.pipe_cfg.klipphotometry['zfactor'],
                               alignment_box=dataset.pipe_cfg.klipphotometry['alignment_box'],
                               parallel_runs=dataset.pipe_cfg.klipphotometry['parallel_runs'],
                               update_header=dataset.pipe_cfg.klipphotometry['update_header'],
                               chunksize=dataset.pipe_cfg.klipphotometry['chunksize'],
                               label=dataset.pipe_cfg.klipphotometry['label'],
                               kill_plots=dataset.pipe_cfg.klipphotometry['kill_plots'],
                               delta=dataset.pipe_cfg.klipphotometry['delta'],
                               radius=dataset.pipe_cfg.klipphotometry['radius'],
                               skip_filters=dataset.pipe_cfg.klipphotometry['skip_filters'],
                               sat_thr=dataset.pipe_cfg.klipphotometry['sat_thr'],
                               mkd=dataset.pipe_cfg.klipphotometry['mkd'],
                               mad=dataset.pipe_cfg.klipphotometry['mad'],
                               PSF_sub_flags=dataset.pipe_cfg.klipphotometry['PSF_sub_flags'])

    DF.save_dataframes(__name__)