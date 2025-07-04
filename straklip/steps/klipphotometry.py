import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from straklip.tiles import Tile
from straklip.utils.utils_photometry import mvs_aperture_photometry,read_dq_from_tile,KLIP_aperture_photometry_handler
from straklip.utils.ancillary import print_mean_median_and_std_sigmacut,rotate_point,parallelization_package
# from straklip.utils.utils_tile import allign_images
from straklip.utils.utils_tile import rotate_fits_north_up_east_left, merge_images
from straklip.utils.utils_dataframe import create_empty_df
from straklip.steps.buildhdf import make_candidates_dataframes
from straklip.stralog import getLogger
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from astropy.io import fits
from pathlib import Path
from IPython.display import display
from scipy.spatial import distance_matrix
from scipy.stats import sigmaclip

def task_mvs_tiles_and_photometry(DF, fitsname, ids_list, filter, use_xy_SN, use_xy_m, use_xy_cen, xy_shift_list,
                                  xy_dmax, bpx_list, spx_list, legend, showplot, overwrite, verbose, Python_origin,
                                  cr_remove, la_cr_remove, cr_radius, kill_plots, ee_df, zpt, radius_ap, radius_sky_inner,
                                  radius_sky_outer, sat_thr, grow_curves, r_in, p, gstep, multiply_by_exptime,
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
            # path2tile = '%s/%s/%s/%s/mvs_tiles/%s/tile_ID%i.fits' % (
            # path2data, DF.project, DF.target, DF.inst, filter, id)
            path2tile='%s/mvs_tiles/%s/tile_ID%i.fits' % (DF.path2out, filter, mvs_ids)
            type_flag = DF.unq_targets_df.loc[DF.unq_targets_df.unq_ids == DF.crossmatch_ids_df.loc[
                DF.crossmatch_ids_df.mvs_ids == id].unq_ids.unique()[0]].type.values[0]
            x = DF.mvs_targets_df.loc[(DF.mvs_targets_df[f'fits_{filter}'] == fitsname) & (
                        DF.mvs_targets_df.mvs_ids == id), f'x_{filter}'].values[0]
            y = DF.mvs_targets_df.loc[(DF.mvs_targets_df[f'fits_{filter}'] == fitsname) & (
                        DF.mvs_targets_df.mvs_ids == id), f'y_{filter}'].values[0]
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
                    DF.mvs_targets_df.loc[(DF.mvs_targets_df[f'fits_{filter}'] == fitsname) & (
                                DF.mvs_targets_df.mvs_ids == id), [f'x_{filter}' , f'y_{filter}']] = [
                        x + deltax, y + deltay]
                    x = DF.mvs_targets_df.loc[(DF.mvs_targets_df[f'fits_{filter}'] == fitsname) & (
                                DF.mvs_targets_df.mvs_ids == id),f'x_{filter}'].values[0]  # -0.001
                    y = DF.mvs_targets_df.loc[(DF.mvs_targets_df[f'fits_{filter}'] == fitsname) & (
                                DF.mvs_targets_df.mvs_ids == id), f'y_{filter}'].values[0]  # -0.001

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
                        Datacube = CRDATA.append_tile(path2tile, Datacube=Datacube, verbose=False, name='CRcleanSCI',
                                                      return_Datacube=False)

                if not DF.skip_photometry:
                    phot.append(
                        mvs_aperture_photometry(DF, filter, ee_df, zpt, fitsname=fitsname, mvs_ids_list_in=[id],
                                                bpx_list=bpx_list, spx_list=spx_list, la_cr_remove=la_cr_remove,
                                                cr_radius=cr_radius, radius_ap=radius_ap, radius_sky_inner=radius_sky_inner,
                                                radius_sky_outer=radius_sky_outer, sat_thr=sat_thr, kill_plots=kill_plots,
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
        getLogger(__name__).debug('m%s e%s spx%s bpx%s %s_r %s_rsky1 %s_rsky2' % (
        filter[1:4], filter[1:4], filter[1:4], filter[1:4], filter, filter, filter))
        getLogger(__name__).debug('phot: ', phot)
    return (phot)

# def mk_mvs_tiles_and_photometry(DF, filter, mvs_ids_test_list=[], overwrite=True, xy_SN=True, xy_m=False,
#                                 xy_cen=False, xy_shift_list=[], xy_dmax=3, bpx_list=[], spx_list=[], legend=False,
#                                 showplot=False, showplot_final=False, verbose=False, workers=None, Python_origin=True,
#                                 parallel_runs=True, cr_remove=False, la_cr_remove=False, cr_radius=3, kill_plots=False,
#                                 chunksize=None, ee_df=None, zpt=0, radius_ap=10, radius_sky_inner=10, radius_sky_outer=15,
#                                 sat_thr=np.inf, grow_curves=True, r_in=1, p=100, gstep=0.1, multiply_by_exptime=False,
#                                 multiply_by_gain=False, multiply_by_PAM=False):
#     '''
#     update the multi-visits tile dataframe with the tiles for each source
#
#     Parameters
#     ----------
#     mvs_ids_test_list : list, optional
#         list of multivisits ids to test. The default is [].
#     overwrite: bool, optional
#         if False, skip IDs that already have tiles in directory. Default is True.
#     xy_SN : bool
#         choose to use image with better SN to recenter the tile.
#     xy_m : bool
#         choose to use maximum in image to recenter the tile.
#     xy_cen : bool
#         choose to use centroid in image to recenter the tile.
#     xy_dmax : int
#         maximum distance from original coordinates to look for maximum.
#     bpx_list: list
#         list of values to flag as bad pixels using the data quality image (coud be real bad pixels, hot/warm pixels, cosmic rays etc.)
#     spx_list: list
#         list of values to flag as saturated pixels using the data quality image
#     legend : bool, optional
#         choose to show legends in plots. The default is False.
#     showplot : bool, optional
#         choose to show plots. The default is False.
#     showplot_final : bool, optional
#         choose to show fianl plot after all manipolatins has been applyed.
#         The default is False.
#     verbose : bool, optional
#         choose to show prints. The default is False.
#     workers : int, optional
#         number of workers to split the work accross multiple CPUs. The default is 3.
#     Python_origin : bool, optional
#         Choose to specify the origin of the xy input coordinates. For exmaple python array star counting from 0,
#         so a position obtained on a python image will have 0 as first pixel.
#         On the other hand, normal catalogs start counting from 1 (see coordinate on ds9 for example)
#         so we need to subtract 1 to make them compatible when we use those coordinates on python
#         The default is True
#     parallel_runs: bool, optional
#         Choose to to split the workload over different CPUs. The default is True
#     la_cr_remove : bool, optional
#         choose to apply L.A. cosmic ray removal. The default is False.
#     cr_radius : int, optional
#         minimum distance from center where to not apply the cosmic ray filter. The default is 3.
#     close : bool, optional
#         choose to close plot istances. The default is True.
#     kill_plots:
#         choose to kill all plots created. The default is False.
#     num_of_chunks : int, optional
#         number of chunk to split the targets. The default is None.
#     chunksize : int, optional
#         size of each chunk for the parallelization process. The default is None.
#     multiply_by_exptime:
#         to properly evaluate the error you need the images total counts (not divided by exptime).
#         If that the case, this option will multiply the image but the corresponding exptime. The default is False.
#     multiply_by_gain:
#         to properly evaluate evaluate photometry you need to convert the counts in electrons (multiply by the gain).
#         If that not the case, this option will multiply the image but the corresponding gain. The default is False.
#     multiply_by_PAM:
#         to properly evaluate photometry accross the FLT you need a not distorted images.
#         If that the case, this option will multiply the image but the corresponding PAM. The default is False.
#
#     Returns
#     -------
#     None.
#
#     '''
#     # if __name__ == 'utils_straklip':
#     q = np.where(filter == np.array(DF.filters))[0][0]
#
#     ids_list_of_lists = []
#     fitsname_list = []
#     getLogger(__name__).info("Working on %s" % filter)
#     # mk_dir('%s/%s/%s/%s/mvs_tiles/%s' % (path2data, DF.project, DF.target, DF.inst, filter), verbose=True)
#     fits_dict = {}
#     skip_IDs_list_list = []
#
#     if len(mvs_ids_test_list) != 0:
#         fitsname_test_list = DF.mvs_targets_df.loc[
#             DF.mvs_targets_df.mvs_ids.isin(mvs_ids_test_list) & ~DF.mvs_targets_df.mvs_ids.isin(
#                 skip_IDs_list_list), f'fits_{filter}'].tolist()
#         for index, row in DF.mvs_targets_df.loc[DF.mvs_targets_df[f'fits_{filter}'].isin(
#                 fitsname_test_list) & DF.mvs_targets_df.mvs_ids.isin(
#                 mvs_ids_test_list) & ~DF.mvs_targets_df.mvs_ids.isin(skip_IDs_list_list)].groupby(
#                 '%s_%s' % (filter, DF.fits_ext)):
#             fits_dict[index] = row.mvs_ids.tolist()
#     else:
#         fitsname_test_list = DF.mvs_targets_df.loc[
#             ~DF.mvs_targets_df.mvs_ids.isin(skip_IDs_list_list), f'fits_{filter}'].unique().tolist()
#         for index, row in DF.mvs_targets_df.loc[DF.mvs_targets_df[f'fits_{filter}'].isin(
#                 fitsname_test_list) & ~DF.mvs_targets_df.mvs_ids.isin(skip_IDs_list_list)].groupby(
#                 '%s_%s' % (filter, DF.fits_ext)):
#             fits_dict[index] = row.mvs_ids.tolist()
#
#     fits_dict = (dict(sorted(fits_dict.items(), key=lambda item: item[1])))
#     ids_list_of_lists = list(fits_dict.values())
#     fitsname_list = list(fits_dict.keys())
#     if parallel_runs:
#         workers, chunksize, ntarget = parallelization_package(workers, len(fitsname_list), chunksize=chunksize)
#         getLogger(__name__).info('Loading a total of %i images' % ntarget)
#         with ProcessPoolExecutor(max_workers=workers) as executor:
#             for phot in executor.map(task_mvs_tiles_and_photometry, repeat(DF), fitsname_list, ids_list_of_lists,
#                                  repeat(filter), repeat(xy_SN), repeat(xy_m), repeat(xy_cen), repeat(xy_shift_list),
#                                  repeat(xy_dmax), repeat(bpx_list), repeat(spx_list), repeat(legend),
#                                  repeat(showplot), repeat(overwrite), repeat(verbose), repeat(Python_origin),
#                                  repeat(cr_remove), repeat(la_cr_remove), repeat(cr_radius), repeat(kill_plots),
#                                  repeat(ee_df), repeat(zpt[q]), repeat(radius_ap), repeat(radius_sky_inner),
#                                  repeat(radius_sky_outer), repeat(sat_thr), repeat(grow_curves), repeat(r_in), repeat(p),
#                                  repeat(gstep), repeat(multiply_by_exptime), repeat(multiply_by_gain),
#                                  repeat(multiply_by_PAM), chunksize=chunksize):
#                 phot = np.array(phot)
#                 for elno in range(len(phot)):
#                     sel = (DF.mvs_targets_df.mvs_ids == phot[elno, 0].astype(float)) & (
#                                 DF.mvs_targets_df.ext == phot[elno, 1].astype(float))
#                     DF.mvs_targets_df.loc[
#                         sel, [f'counts_{filter}', f'ecounts_{filter}', f'nap_{filter}',
#                               f'm_{filter}', f'e_{filter}' , f'spx_{filter}',
#                               f'bpx_{filter}', f'%s_r' % filter, f'%s_rsky1' % filter, f'%s_rsky2' % filter,
#                               f'sky_{filter}', f'esky_{filter}', f'nsky_{filter}',
#                               f'grow_corr_{filter}' % filter[1:4]]] = phot[elno, 2:-1].astype(float)
#                     DF.mvs_targets_df.loc[sel, ['flag_%s' % filter]] = phot[elno, -1:]
#
#     else:
#         for elno in range(len(fitsname_list)):
#             if len(xy_shift_list) > 0:
#                 w = np.where(np.array(fitsname_list)[elno] == np.array(fitsname_test_list))[0][0]
#                 use_xy_shift_list = xy_shift_list[w]
#             else:
#                 use_xy_shift_list = []
#             phot = task_mvs_tiles_and_photometry(DF, fitsname_list[elno], ids_list_of_lists[elno], filter, xy_SN,
#                                                  xy_m, xy_cen, use_xy_shift_list, xy_dmax, bpx_list, spx_list,
#                                                  legend, showplot, overwrite, verbose, Python_origin, cr_remove,
#                                                  la_cr_remove, cr_radius, kill_plots, ee_df, zpt[q], radius_ap,
#                                                  radius_sky_inner, radius_sky_outer, sat_thr, grow_curves, r_in, p, gstep,
#                                                  multiply_by_exptime, multiply_by_gain, multiply_by_PAM)
#             phot = np.array(phot)
#             for elno in range(len(phot)):
#                 sel = (DF.mvs_targets_df.mvs_ids == phot[elno, 0].astype(float)) & (
#                             DF.mvs_targets_df.ext == phot[elno, 1].astype(float))
#                 DF.mvs_targets_df.loc[
#                     sel, [f'counts_{filter}', f'ecounts_{filter}', f'nap_{filter}',
#                           f'm_{filter}', f'e_{filter}', f'spx_{filter}', f'bpx_{filter}',
#                           f'%s_r', f'rsky1_{filter}', f'rsky2_{filter}', f'sky_{filter}',
#                           f'esky_{filter}', f'nsky_{filter}', f'grow_corr_{filter}']] = phot[elno,
#                                                                                                           2:-1].astype(
#                     float)
#                 DF.mvs_targets_df.loc[sel, ['flag_%s' % filter]] = phot[elno, -1:]

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

    # if verbose: getLogger(__name__).info('Verbose mode: ',verbose)
    # if verbose:
    #     display(DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.unq_ids==avg_id])
    #     display(DF.unq_targets_df.loc[DF.unq_targets_df.unq_ids==avg_id])
    #     display(DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.unq_ids==avg_id].mvs_ids)])

    candidate_df=create_empty_df(['filter','mvs_ids'],['counts','ecounts','nsky','Nsigma','mag','emag','flag','ROTA','PA_V3','std','sep'],multy_index=True,levels=[DF.filters,DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.unq_ids==avg_id].mvs_ids])
    for filter in DF.filters:
        if filter not in skip_filters:
            getLogger(__name__).info(f'Performing aperture photometry on filter {filter}')
            mvs_ids=DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.unq_ids==avg_id].mvs_ids.unique()
            for id in mvs_ids:
                if not DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==id,f'flag_{filter}'].str.contains('rejected').values[0]:
                    getLogger(__name__).info(f'Performing aperture photometry on mvs_ids {id}')
                    x,y=[int((DF.tilebase-1)/2),int((DF.tilebase-1)/2)]
                    path2tile=DF.path2out+f'/mvs_tiles/{filter}/tile_ID{id}.fits'
                    DATA=Tile(x=int((DF.tilebase-1)/2),y=int((DF.tilebase-1)/2),tile_base=DF.tilebase,inst=DF.inst)
                    DATA.load_tile(path2tile,ext=label_dict[label],verbose=False,return_Datacube=False)
                    DQ=Tile(x=int((DF.tilebase-1)/2),y=int((DF.tilebase-1)/2),tile_base=DF.tilebase,inst=DF.inst)
                    DQ.load_tile(path2tile,ext=3,verbose=False,return_Datacube=False)

                    forcedSky=DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==id,[f'sky_{filter}',f'esky_{filter}',f'nsky_{filter}']].values[0]
                    exptime=DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==id,f'exptime_{filter}'].values[0]
                    counts,ecounts,Nsigma,Nap,mag,emag,spx,bpx,Sky,eSky,nSky,grow_corr=KLIP_aperture_photometry_handler(DF,id,filter,x=x,y=y,data=DATA.data,dqdata=DQ.data,aptype=aptype,noBGsub=True,forcedSky=forcedSky,sigma=sigma,kill_plots=True,Python_origin=True,delta=delta,sat_thr=sat_thr,exptime=exptime)#,multiply_by_exptime=multiply_by_exptime,multiply_by_gain=multiply_by_gain,multiply_by_PAM=multiply_by_PAM)
                    candidate_df.loc[(filter,id),['counts','ecounts','nsky','Nsigma','mag','emag']]=[counts,ecounts,Nap,Nsigma,mag,emag]
                else:
                    getLogger(__name__).info(f'Rejected aperture photometry on mvs_ids {id}')
        else:
            getLogger(__name__).info(f'Rejected aperture photometry on filter {filter}')

    if verbose:
        plt.show()
    return(candidate_df)

# TO DO: move this task at psfsubtraction stage, instead of klipphotometry
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
    hdul_dict={'data':1,'crclean_data':2}
    KLIP_label_dict={'data':'Kmode','crclean_data':'crclean_Kmode'}
    path2tile=DF.path2out+f'/median_tiles/{filter}/tile_ID{id}.fits'

    cand_mvs_ids_list=DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids.isin(DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.unq_ids==id].mvs_ids.unique())].mvs_ids.unique()

    sel_flag=(DF.mvs_targets_df[f'flag_{filter}']!='rejected')
    mvs_ids_list=DF.mvs_targets_df.loc[sel_flag&DF.mvs_targets_df.mvs_ids.isin(DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.unq_ids==id].mvs_ids.unique())].mvs_ids.unique()
    sel_ids = DF.mvs_targets_df.mvs_ids.isin(mvs_ids_list) & DF.mvs_targets_df.mvs_ids.isin(cand_mvs_ids_list)

    IMAGE=Tile(x=(DF.tilebase-1)/2,y=(DF.tilebase-1)/2,tile_base=DF.tilebase,delta=0,inst=DF.inst,Python_origin=True)
    Datacube=IMAGE.load_tile(path2tile,return_Datacube=True,hdul_max=hdul_dict[label],verbose=False,mode='update',raise_errors=False)

    getLogger(__name__).info(f'Making median candidate tiles for KLIPmodes: {DF.kmodes}')
    for Kmode in DF.kmodes:
        getLogger(__name__).info(f'KLIPmode: {Kmode} ,unq_ids: {id}, mvs_ids: {mvs_ids_list}')


        rotated_images = []
        for mvs_ids in mvs_ids_list:
            input_fits = '%s/mvs_tiles/%s/tile_ID%i.fits' % (DF.path2out, filter, mvs_ids)
            rotated_image = rotate_fits_north_up_east_left(input_fits,ext='%s%s'%(KLIP_label_dict[label],Kmode))
            rotated_images.append(rotated_image)


        candidate_tile = merge_images(np.array(rotated_images), tile_base=DF.tilebase, inst=DF.inst,verbose=False,title='%s Median Target'%(filter),showplot=False,kill_plots=True)
        candidate_tile.append_tile(path2tile,Datacube=Datacube,verbose=False,name=f'{(KLIP_label_dict[label])}{Kmode}',return_Datacube=False,write=False)


def task_mvs_candidates_infos(DF,avg_id,d,skip_filters,needs_filters,aptype,verbose,noBGsub,sigma,DF_fk,label,kill_plots,delta,radius,sat_thr,mfk,mdf,mad,mak):
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
    mfk: int, optional
        minimum filter detections per KLIPmode to accept a candidate
    mdf: int, optional
        minimum detections per filter to accept a candidate
    mad: int, optional
        minimum arecsecond distance from center to accept a candidate
    mak: int, optional
        minimum acceptable klipmmode to accept a  candidate
    Returns
    -------
    None.

    '''
    getLogger(__name__).info(f'Working on candidate avg_id: {avg_id}')
    KLIP_label_dict={'data':'Kmode','crclean_data':'crclean_Kmode'}
    minimum_px_distance_from_center=mad/DF.pixscale
    origin=(int((DF.tilebase-1)/2),int((DF.tilebase-1)/2)) #these are the postons of the peak in each tile
    if not kill_plots:
        if verbose: kill_plots=False
        else: kill_plots=True

    if verbose:
        getLogger(__name__).info('Verbose mode: ',verbose)
        display(DF.mvs_targets_df.loc[DF.crossmatch_ids_df.unq_ids==avg_id])
    filters_list=[filter for filter in DF.filters if filter not in ['F658N']]
    temporary_candidate_df=create_empty_df(['Kmode','filter','mvs_ids'],['x_tile','y_tile','x_rot','y_rot','pdc','ROTA','PA_V3','Nsigma','flag','counts','ecounts','mag','emag'],multy_index=True,levels=[DF.kmodes,filters_list,DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.unq_ids==avg_id].mvs_ids])
    sub_ids=DF.crossmatch_ids_df.loc[(DF.crossmatch_ids_df.unq_ids==avg_id)].mvs_ids.unique()
    for Kmode in DF.kmodes:
        for filter in DF.filters:
            if filter not in skip_filters:
                zpt=DF.mvs_targets_df[f'delta_{filter}'].unique()
                ezpt=DF.mvs_targets_df[f'edelta_{filter}'].unique()
                elno=0
                if verbose: fig,ax=plt.subplots(1,len(sub_ids),figsize=(5*len(sub_ids),5))
                else: fig,ax=[None,None]

                for mvs_ids in sub_ids:
                    if not DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==mvs_ids,f'flag_{filter}'].str.contains('rejected').values[0] and Kmode >= mak:
                        PA_V3=DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==mvs_ids,f'pav3_{filter}'].values[0]
                        ROTA=DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==mvs_ids,f'rota_{filter}'].values[0]
                        path2tile=DF.path2out+f'/mvs_tiles/{filter}/tile_ID{mvs_ids}.fits'
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
                                exptime=DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids==mvs_ids,f'exptime_{filter}'].values[0]
                                if not np.all(np.isnan(KDATA.data)) and Xrot>=0 and Xrot<=DF.tilebase and Yrot>=0 and Yrot<=DF.tilebase:
                                    if not kill_plots:getLogger(__name__).debug('Kmode %s, %s, mvs_id %s'%(Kmode, filter,mvs_ids))
                                    counts,ecounts,Nsigma,Nap,mag,emag,spx,bpx,Sky,eSky,nSky,grow_corr=KLIP_aperture_photometry_handler(DF,mvs_ids,filter,x=x,y=y,data=KDATA.data,zpt=zpt,ezpt=ezpt,aptype=aptype,noBGsub=False,sigma=sigma,kill_plots=kill_plots,Python_origin=True,delta=delta,radius_a=radius,sat_thr=sat_thr,exptime=exptime)#,multiply_by_exptime=multiply_by_exptime,multiply_by_gain=multiply_by_gain,multiply_by_PAM=multiply_by_PAM)
                                    counts,ecounts,_,Nap,mag,emag,spx,bpx,Sky,eSky,nSky,grow_corr=KLIP_aperture_photometry_handler(DF,mvs_ids,filter,x=x,y=y,data=KDATA.data,zpt=zpt,ezpt=ezpt,aptype=aptype,noBGsub=False,sigma=sigma,kill_plots=kill_plots,Python_origin=True,delta=delta,radius_a=radius,sat_thr=sat_thr,exptime=exptime)#,multiply_by_exptime=multiply_by_exptime,multiply_by_gain=multiply_by_gain,multiply_by_PAM=multiply_by_PAM)

                                    filtered_data, lower_bound, upper_bound = sigmaclip(KDATA.data, low=5, high=5)
                                    STD = np.nanstd(filtered_data)
                                    Nsigma = KDATA.data[y][x] / STD

                                    if Nsigma> 0 and not np.isnan(mag):
                                        temporary_candidate_df.loc[(Kmode,filter,mvs_ids),['counts','ecounts','Nsigma','mag','emag']]=[counts,ecounts,Nsigma,mag[0],emag[0]]
                                        temporary_candidate_df.loc[(Kmode,filter,mvs_ids),['sep']]=dist
                                        temporary_candidate_df.loc[(Kmode,filter,mvs_ids),['x_tile','y_tile','x_rot','y_rot','pdc','ROTA','PA_V3','flag']]=[KDATA.x_m,KDATA.y_m,Xrot,Yrot,pdc,ROTA,PA_V3,'good_candidate']
                                        getLogger(__name__).info(f'Candidate avg_id {avg_id}, filter {filter}, kmode {Kmode}, good_candidate')

                                    else:
                                        temporary_candidate_df.loc[(Kmode,filter,mvs_ids),['x_tile','y_tile','x_rot','y_rot','pdc','ROTA','PA_V3','Nsigma','flag']]=[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,'rejected']
                                        getLogger(__name__).info(f'Candidate avg_id {avg_id}, mvs_ids {mvs_ids}, filter {filter}, kmode {Kmode}, rejected because sigma <0')
                                        plt.close()
                                else:
                                    temporary_candidate_df.loc[(Kmode,filter,mvs_ids),['x_tile','y_tile','x_rot','y_rot','pdc','ROTA','PA_V3','Nsigma','flag']]=[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,'rejected']
                                    getLogger(__name__).info(f'Candidate avg_id {avg_id}, mvs_ids {mvs_ids}, filter {filter}, kmode {Kmode}, rejected because rotated coordinates puts star out of the tile')
                                    plt.close()
                            else:
                                temporary_candidate_df.loc[(Kmode,filter,mvs_ids),['x_tile','y_tile','x_rot','y_rot','pdc','ROTA','PA_V3','Nsigma','flag']]=[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,'rejected']
                                getLogger(__name__).info(
                                    f'Candidate avg_id {avg_id}, mvs_ids {mvs_ids}, filter {filter}, kmode {Kmode}, rejected because rotated coordinates puts star out of the tile')
                                plt.close()
                        else:
                            temporary_candidate_df.loc[(Kmode,filter,mvs_ids),['x_tile','y_tile','x_rot','y_rot','pdc','ROTA','PA_V3','Nsigma','flag']]=[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,'rejected']
                            getLogger(__name__).info(
                                f'Candidate avg_id {avg_id}, mvs_ids {mvs_ids}, filter {filter}, kmode {Kmode}, rejected beacuse tile is NaN')
                            plt.close()
                    else:
                        temporary_candidate_df.loc[(Kmode,filter,mvs_ids),['x_tile','y_tile','x_rot','y_rot','ROTA','pdc','PA_V3','Nsigma','flag']]=[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,'rejected']
                        getLogger(__name__).info(f'Candidate avg_id {avg_id}, mvs_ids {mvs_ids}, filter {filter}, kmode {Kmode}, rejected')
                        plt.close()


    Nsigma_score=[]
    Kmode_idx_list=[]
    for Kmode in DF.kmodes:
        if (temporary_candidate_df.loc[(Kmode)].flag != 'rejected').astype(int).sum(axis=0) >= 1:
            pos=temporary_candidate_df.loc[temporary_candidate_df.flag!='rejected'].loc[(Kmode),['y_rot','x_rot']].values
            pair_dist = np.round(distance_matrix(pos, pos).astype(float),3)
            num=pos.shape[0]
            pair_dist[np.r_[:num], np.r_[:num]] = np.nan

            temp_filters_list=temporary_candidate_df.loc[temporary_candidate_df.flag!='rejected'].loc[Kmode].index.get_level_values('filter').values
            distance_matrix_df=pd.DataFrame(data=pair_dist,columns=temp_filters_list,index=temp_filters_list)
            sel_flag=(temporary_candidate_df.loc[(Kmode)].flag!='rejected')
            distance_matrix_df['mvs_ids']=temporary_candidate_df.loc[(Kmode)].loc[sel_flag].index.get_level_values('mvs_ids').values
            columns=distance_matrix_df.columns[:-1].unique()
            minimum_filters_detections=np.max([int((len(distance_matrix_df[columns].columns))/2)+1,1])
            for filter in distance_matrix_df.index.get_level_values(0).unique():
                try:
                    selected_distance_matrix_df=(((distance_matrix_df.loc[filter,columns].to_frame().T<=1.5)&(distance_matrix_df.loc[filter,columns].to_frame().T>=0))|(distance_matrix_df.loc[filter,columns].to_frame().T.isna()))
                    filter_id_sel_list=(selected_distance_matrix_df.astype(int).sum(axis=1).values>=minimum_filters_detections)
                    sel_mvs_ids=temporary_candidate_df.index.get_level_values('mvs_ids').isin(distance_matrix_df.loc[filter].to_frame().T.mvs_ids.values[filter_id_sel_list])
                    if verbose:
                        getLogger(__name__).debug('Printing Kmode %i distance_matrix_df'%Kmode)
                        display(distance_matrix_df.loc[filter,columns].to_frame().T)
                except:
                    selected_distance_matrix_df=(((distance_matrix_df.loc[filter,columns]<=1.5)&(distance_matrix_df.loc[filter,columns]>=0))|(distance_matrix_df.loc[filter,columns].isna()))
                    filter_id_sel_list=(selected_distance_matrix_df.astype(int).sum(axis=1).values>=minimum_filters_detections)
                    sel_mvs_ids=temporary_candidate_df.index.get_level_values('mvs_ids').isin(distance_matrix_df.loc[filter].mvs_ids.values[filter_id_sel_list])
                    if verbose:
                        getLogger(__name__).debug('Printing Kmode %i distance_matrix_df'%Kmode)
                        display(distance_matrix_df.loc[filter,columns])

                if temporary_candidate_df.loc[(Kmode,filter,sel_mvs_ids),'Nsigma'].count() >= mdf:
                    temporary_candidate_df.loc[(Kmode,filter,~sel_mvs_ids)]=np.nan
                    temporary_candidate_df.loc[(Kmode,filter,~sel_mvs_ids),'flag']='rejected'
                else:
                    temporary_candidate_df.loc[(Kmode,filter)]=np.nan
                    temporary_candidate_df.loc[(Kmode,filter),'flag']='rejected'

            # if len(temporary_candidate_df.loc[(Kmode),'flag'].loc[temporary_candidate_df.loc[(Kmode),'flag']!='rejected'].index.unique(level='filter')) >= mfk:
            if np.all((temporary_candidate_df.loc[(Kmode),'flag'] != 'rejected').values):
                Nsigma_score.append(np.nanmean(temporary_candidate_df.loc[(Kmode),'Nsigma'].values))
                Kmode_idx_list.append(True)
                if verbose:
                    getLogger(__name__).debug('Selected candidate df:')
                    getLogger(__name__).debug(temporary_candidate_df.loc[Kmode])
            elif len(temporary_candidate_df.loc[(Kmode, needs_filters),'flag'].loc[temporary_candidate_df.loc[(Kmode, needs_filters),'flag']!='rejected'].index.unique(level='filter')) >= mfk:
                Nsigma_score.append(np.nanmean(temporary_candidate_df.loc[((Kmode, needs_filters)),'Nsigma'].values))
                Kmode_idx_list.append(True)
                if verbose:
                    getLogger(__name__).debug('Selected candidate df only in filters: ', needs_filters)
                    getLogger(__name__).debug(temporary_candidate_df.loc[Kmode])
            else:
                temporary_candidate_df.loc[Kmode] = np.nan
                temporary_candidate_df.loc[Kmode, 'flag'] = 'rejected'
                Nsigma_score.append(0)
                Kmode_idx_list.append(False)
        else:
            temporary_candidate_df.loc[Kmode]=np.nan
            temporary_candidate_df.loc[Kmode,'flag']='rejected'
            Nsigma_score.append(0)
            Kmode_idx_list.append(False)

    Nsigma_score=np.array(Nsigma_score)
    # Nsigma_sel_score=np.array(Nsigma_score)[Kmode_idx_list]

    def is_part_of_three_contiguous_true(lst, index):
        """
        Check if the value at a given index is part of three contiguous True values.

        :param lst: List of booleans (True/False).
        :param index: The index to check.
        :return: True if the value at the index is part of three contiguous True values, False otherwise.
        """
        # Determine the start and end of the window
        start = max(0, index - 2)  # Ensure start does not go below 0
        end = min(len(lst) - 1, index + 2)  # Ensure end does not exceed the list length

        # Check for three contiguous True values within the valid window
        for i in range(start, end - 1):  # Check all possible windows of size 3
            if lst[i] and lst[i + 1] and lst[i + 2]:
                return True
        return False

    Kmode_final = None
    candidate_df = None
    # if len(Nsigma_sel_score)>0:
    if len(Nsigma_score) > 0:
        # q = np.where(Nsigma_score == np.nanmax(Nsigma_sel_score))[0][-1]
        # q = np.where(Nsigma_score == np.nanmax(Nsigma_score))[0][-1]
        for q in np.argsort(Nsigma_score)[::-1]:
            if is_part_of_three_contiguous_true([True if i>=5 else False for i in Nsigma_score],q):# and Nsigma >= 5:
                Kmode_final=DF.kmodes[q]
                candidate_df=temporary_candidate_df.loc[(Kmode_final)]
                candidate_df.loc[candidate_df.flag=='rejected',['x_tile','y_tile','x_rot','y_rot','ROTA','PA_V3','Nsigma','counts','ecounts','mag','emag']]=np.nan

                getLogger(__name__).info(f'EUREKA!!!!! We have a Candidate for id {avg_id}. Kmode selected # {Kmode_final}')
                if verbose:
                    getLogger(__name__).debug(candidate_df)
                break

    if Kmode_final is None:
        getLogger(__name__).info(f'BOOMER!!!!! No Candidate found for id {avg_id}')
    # else:
    #     Kmode_final=None
    #     candidate_df=None
    #     getLogger(__name__).info(f'BOOMER!!!!! No Candidate found for id {avg_id}')

    return(Kmode_final,candidate_df)

def update_median_candidates_tile(DF, unq_ids_list, column_name='Kmode', workers=None, zfactor=10, alignment_box=3,
                                  parallel_runs=True, chunksize=None, label='data', kill_plots=True, skip_filters=[]):
    '''
    Update the median candidate dataframe tile.

    Parameters
    ----------
    unq_ids_list : list
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
    getLogger(__name__).info('Loading a total of %i images' % len(unq_ids_list))
    for filter in DF.filters:
        if filter not in skip_filters:
            getLogger(__name__).info('Working in %s: ' % filter)
            if parallel_runs:
                workers, chunksize, ntarget = parallelization_package(workers, len(unq_ids_list),
                                                                      chunksize=chunksize)
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    for _ in executor.map(task_median_candidate_infos, repeat(DF), unq_ids_list, repeat(filter),
                                         repeat(column_name), repeat(zfactor), repeat(alignment_box), repeat(label),
                                         chunksize=chunksize):
                        pass
            else:
                for id in unq_ids_list:
                    task_median_candidate_infos(DF, id, filter, column_name, zfactor, alignment_box, label)

def update_companion_ZPT(DF,suffix='',skip_filters='',aptype='4pixels',verbose=False,workers=None,noBGsub=True,sigma=2.5,min_mag_list=[],max_mag_list=[],DF_fk=None,parallel_runs=True,chunksize = None,kill_plots=True,label='data',delta=3,sat_thr=np.inf):
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
    if DF_fk==None: DF_fk=DF
    unq_ids=DF.unq_targets_df.unq_ids.unique()
    ############################################################ ZPT ##############################################################
    getLogger(__name__).info('Working on the zero points for candidates')
    if parallel_runs:
        workers,chunksize,ntarget=parallelization_package(workers,len(unq_ids),chunksize = chunksize)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for candidate_df in executor.map(task_mvs_targets_infos,repeat(DF),unq_ids,repeat(skip_filters),repeat(aptype),repeat(verbose),repeat(noBGsub),repeat(sigma),repeat(kill_plots),repeat(label),repeat(delta),repeat(sat_thr),chunksize=chunksize):
                for filter in candidate_df.index.get_level_values('filter'):
                    DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(candidate_df.index.get_level_values('mvs_ids').unique()),f'counts_{filter}_ap']=candidate_df.loc[(filter),'counts'].values.astype('float')
                    DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(candidate_df.index.get_level_values('mvs_ids').unique()),f'ecounts_{filter}_ap']=candidate_df.loc[(filter),'ecounts'].values.astype('float')
                    DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(candidate_df.index.get_level_values('mvs_ids').unique()),f'nsky_{filter}_ap']=candidate_df.loc[(filter),'nsky'].values.astype('float')
                    DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(candidate_df.index.get_level_values('mvs_ids').unique()),f'm_{filter}_ap']=candidate_df.loc[(filter),'mag'].values.astype('float')
                    DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(candidate_df.index.get_level_values('mvs_ids').unique()),f'e_{filter}_ap']=candidate_df.loc[(filter),'emag'].values.astype('float')
    else:
        for id in unq_ids:
            candidate_df =task_mvs_targets_infos(DF,id,skip_filters,aptype,verbose,noBGsub,sigma,kill_plots,label,delta,sat_thr)
            for filter in candidate_df.index.get_level_values('filter'):
                DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(candidate_df.index.get_level_values('mvs_ids').unique()),f'counts_{filter}_ap']=candidate_df.loc[(filter),'counts'].values.astype('float')
                DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(candidate_df.index.get_level_values('mvs_ids').unique()),f'ecounts_{filter}_ap']=candidate_df.loc[(filter),'ecounts'].values.astype('float')
                DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(candidate_df.index.get_level_values('mvs_ids').unique()),f'nsky_{filter}_ap']=candidate_df.loc[(filter),'nsky'].values.astype('float')
                DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(candidate_df.index.get_level_values('mvs_ids').unique()),f'm_{filter}_ap']=candidate_df.loc[(filter),'mag'].values.astype('float')
                DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids.isin(candidate_df.index.get_level_values('mvs_ids').unique()),f'e_{filter}_ap']=candidate_df.loc[(filter),'emag'].values.astype('float')


    elno = 0
    for filter in DF.filters:
        # sat_sel = (DF.mvs_targets_df['spx_%s'%filter] <= 0)
        type_sel = (DF.mvs_targets_df[f'flag_{filter}'].str.contains('psf'))
        # emag_sel = (DF.mvs_targets_df['e_%s'%filter] < 0.01)
        if len(min_mag_list) == 0:
            min_mag = np.nanmin(DF.mvs_targets_df[f'm_{filter}{suffix}'].values)
        else:
            min_mag = min_mag_list[elno]
        if len(max_mag_list) == 0:
            max_mag = np.nanmax(DF.mvs_targets_df[f'm_{filter}{suffix}'].values)
        else:
            max_mag = max_mag_list[elno]
        mag_sel = (DF.mvs_targets_df.loc[type_sel,f'm_{filter}{suffix}'] >= min_mag) & (DF.mvs_targets_df.loc[type_sel,f'm_{filter}{suffix}'] < max_mag)
        dmags=DF.mvs_targets_df.loc[mag_sel&type_sel,f'm_{filter}{suffix}'].values-DF.mvs_targets_df.loc[mag_sel&type_sel,f'm_{filter}_ap'].values
        dmags=dmags.astype(float)
        dmag_mean_sc,dmag_median_sc,dmag_std_sc,Mask=print_mean_median_and_std_sigmacut(dmags,pre='%s '%filter,verbose=False,sigma=sigma,nonan=True)
        # DF.header_df.loc['Delta%s'%filter,'Values']=dmag_median_sc
        # DF.header_df.loc['eDelta%s'%filter,'Values']=dmag_std_sc
        # DF.deltas=dmag_median_sc
        # DF.edeltas=dmag_std_sc
        DF.mvs_targets_df[f'delta_{filter}']=dmag_median_sc
        DF.mvs_targets_df[f'edelta_{filter}']=dmag_std_sc
        DF.mvs_targets_df[f'm_{filter}_ap']+=dmag_median_sc
        DF.mvs_targets_df[f'e_{filter}_ap']=np.sqrt(DF.mvs_targets_df[f'e_{filter}_ap'].values.astype(float)**2+dmag_std_sc**2)
        elno+=1
    # display(DF.header_df)
    return (DF)


def update_candidates(DF, unq_ids_list=[], suffix='', d=1., skip_filters=['F658N'], needs_filters=[], aptype='4pixels', verbose=False,
                      workers=None, noBGsub=False, sigma=2.5, min_mag_list=[], max_mag_list=[], DF_fk=None,
                      parallel_runs=True, chunksize=None, label='data', kill_plots=True, delta=3,
                      radius=3, sat_thr=np.inf, mfk=1, mdf=2, mad=0.1, mak=1):
    '''
    update the multivisits candidates dataframe with candidates infos

    Parameters
    ----------
    unq_ids_list : list, optional
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
    num_of_chunks : int, optional
        number of chunk to split the targets. The default is None.
    chunksize : int, optional
        size of each chunk for the parallelization process. The default is None.
    mfk: int, optional
        minimum filter detections per KLIPmode to accept a candidate
    mdf: int, optional
        minimum detections per filter to accept a candidate
    mad: int, optional
        minimum arecsecond distance from center to accept a candidate

    Returns
    -------
    None.

    '''
    DF=update_companion_ZPT(DF, suffix=suffix, skip_filters=skip_filters, aptype=aptype, verbose=False,
                             workers=workers, sigma=sigma, min_mag_list=min_mag_list, max_mag_list=max_mag_list,
                             DF_fk=DF_fk, parallel_runs=parallel_runs, chunksize=chunksize, label=label,
                             delta=delta, sat_thr=sat_thr)

    if parallel_runs:
        getLogger(__name__).info('Working on the candidates')
        workers, chunksize, ntarget = parallelization_package(workers, len(unq_ids_list), chunksize=chunksize)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for Kmode_final, candidate_df in executor.map(task_mvs_candidates_infos, repeat(DF), unq_ids_list, repeat(d), repeat(skip_filters),
                                 repeat(needs_filters), repeat(aptype), repeat(verbose), repeat(False), repeat(3), repeat(DF_fk),
                                 repeat(label), repeat(kill_plots), repeat(delta), repeat(radius), repeat(sat_thr), repeat(mfk),
                                 repeat(mdf), repeat(mad), repeat(mak), chunksize=chunksize):
                if Kmode_final != None: update_candidates_with_detection(DF, candidate_df, Kmode_final, verbose)

    else:
        for avg_id in unq_ids_list:
            # try:
            Kmode_final, candidate_df = task_mvs_candidates_infos(DF, avg_id, d, skip_filters, needs_filters, aptype, verbose,
                                                                  False, 3, DF_fk, label, kill_plots, delta, radius,
                                                                  sat_thr, mfk, mdf, mad,mak)
            if Kmode_final != None: DF=update_candidates_with_detection(DF, candidate_df, Kmode_final, verbose)
            # except:
            #     raise ValueError('Something wrong with avg_id %s, Please check' % avg_id)

    selected_unq_ids = DF.crossmatch_ids_df.loc[
        DF.crossmatch_ids_df.mvs_ids.isin(DF.mvs_candidates_df.mvs_ids)].unq_ids.unique()
    DF.unq_candidates_df = DF.unq_candidates_df.loc[
        DF.unq_candidates_df.unq_ids.isin(selected_unq_ids)].reset_index(drop=True)
    for unq_ids in DF.unq_candidates_df.unq_ids:
        mvs_ids_list = DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.unq_ids == unq_ids].mvs_ids.unique()

        for filter in DF.filters:
            if filter not in skip_filters and not \
            DF.unq_candidates_df.loc[DF.unq_candidates_df.unq_ids == unq_ids, f'm_{filter}'].isna().values[0]:
                DF.unq_candidates_df.loc[
                    DF.unq_candidates_df.unq_ids == unq_ids, f'nsigma_{filter}'] = np.nanmedian(
                    DF.mvs_candidates_df.loc[
                        DF.mvs_candidates_df.mvs_ids.isin(mvs_ids_list), f'nsigma_{filter}'].values)

                DF.unq_candidates_df.loc[DF.unq_candidates_df.unq_ids == unq_ids, 'mkmode'] = np.nanmedian(
                    DF.mvs_candidates_df.loc[
                        DF.mvs_candidates_df.mvs_ids.isin(mvs_ids_list), [f'kmode_{filter}' for filter in
                                                                          DF.filters]].values)
                DF.unq_candidates_df.loc[DF.unq_candidates_df.unq_ids == unq_ids, 'sep'] = np.nanmean(
                    DF.mvs_candidates_df.loc[
                        DF.mvs_candidates_df.mvs_ids.isin(mvs_ids_list), [f'sep_{filter}' for filter in
                                                                          DF.filters]].values)
                try:
                    DF.unq_candidates_df.loc[(DF.unq_candidates_df.unq_ids == unq_ids), f'magbin_{filter}'] = \
                    DF.unq_targets_df.loc[
                        (DF.unq_targets_df.unq_ids == unq_ids), f'm_{filter}{suffix}'].values[0].astype(int)
                except:
                    DF.unq_candidates_df[
                        (DF.unq_candidates_df.unq_ids == unq_ids), f'magBin_{filter}'] = np.nan
    return(DF)

def pruning_catalogs(DF):
    getLogger(__name__).info('Pruning the candidates df')
    mvs_bad_ids_list = []
    for unq_ids in DF.unq_candidates_df.unq_ids:
        mvs_ids_list = DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.unq_ids == unq_ids].mvs_ids.unique()

        sel_mvs_ids = (DF.mvs_candidates_df.mvs_ids.isin(mvs_ids_list))
        sel_flags = (('good_candidate' == DF.mvs_candidates_df[['flag_%s' % i for i in DF.filters]]).astype(
            int).sum(axis=1) < 1)

        mvs_bad_ids = DF.mvs_candidates_df.loc[sel_mvs_ids & sel_flags].mvs_ids.unique()
        mvs_bad_ids_list.extend(mvs_bad_ids)

    DF.mvs_candidates_df = DF.mvs_candidates_df.loc[
        ~DF.mvs_candidates_df.mvs_ids.isin(mvs_bad_ids_list)].reset_index(drop=True)

    unq_ids_list=DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.mvs_ids.isin(DF.mvs_candidates_df.mvs_ids)].unq_ids.unique()
    DF.unq_candidates_df = DF.unq_candidates_df.loc[
        DF.unq_candidates_df.unq_ids.isin(unq_ids_list)].reset_index(drop=True)

    getLogger(__name__).info('Candidates df pruned.')
    return(DF)

def update_candidates_with_detection(DF,candidate_df,Kmode_final,verbose):
    for filter in candidate_df.index.get_level_values('filter').unique():
        for mvs_ids in candidate_df.loc[(filter)].index.get_level_values('mvs_ids').unique():
            if candidate_df.loc[candidate_df.index.get_level_values('mvs_ids')==mvs_ids].loc[(filter),'flag'].values[0]!='rejected':
                try:
                    DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,f'x_tile_{filter}']=candidate_df.loc[candidate_df.index.get_level_values('mvs_ids')==mvs_ids].loc[(filter),'x_tile'].values[0]
                    DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,f'y_tile_{filter}']=candidate_df.loc[candidate_df.index.get_level_values('mvs_ids')==mvs_ids].loc[(filter),'y_tile'].values[0]
                    DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,f'x_rot_{filter}']=candidate_df.loc[candidate_df.index.get_level_values('mvs_ids')==mvs_ids].loc[(filter),'x_rot'].values[0]
                    DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,f'y_rot_{filter}']=candidate_df.loc[candidate_df.index.get_level_values('mvs_ids')==mvs_ids].loc[(filter),'y_rot'].values[0]

                    DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,f'counts_{filter}']=candidate_df.loc[candidate_df.index.get_level_values('mvs_ids')==mvs_ids].loc[(filter),'counts'].values[0]
                    DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,f'ecounts_{filter}']=candidate_df.loc[candidate_df.index.get_level_values('mvs_ids')==mvs_ids].loc[(filter),'ecounts'].values[0]
                    DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,f'm_{filter}']=candidate_df.loc[candidate_df.index.get_level_values('mvs_ids')==mvs_ids].loc[(filter),'mag'].values[0]
                    DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,f'e_{filter}']=candidate_df.loc[candidate_df.index.get_level_values('mvs_ids')==mvs_ids].loc[(filter),'emag'].values[0]

                    DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,f'rota_{filter}']=candidate_df.loc[candidate_df.index.get_level_values('mvs_ids')==mvs_ids].loc[(filter),'ROTA'].values[0]
                    DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,f'pav3_{filter}']=candidate_df.loc[candidate_df.index.get_level_values('mvs_ids')==mvs_ids].loc[(filter),'PA_V3'].values[0]
                    DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,f'nsigma_{filter}']=candidate_df.loc[candidate_df.index.get_level_values('mvs_ids')==mvs_ids].loc[(filter),'Nsigma'].values[0]
                    DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,f'kmode_{filter}']=Kmode_final
                    DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,f'flag_{filter}']=candidate_df.loc[candidate_df.index.get_level_values('mvs_ids')==mvs_ids].loc[(filter),'flag'].values[0]
                    DF.mvs_candidates_df.loc[DF.mvs_candidates_df.mvs_ids==mvs_ids,f'sep_{filter}']=candidate_df.loc[candidate_df.index.get_level_values('mvs_ids')==mvs_ids].loc[(filter),'sep'].values[0]
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
            DF.unq_candidates_df.loc[DF.unq_candidates_df.unq_ids.isin(DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.mvs_ids.isin(candidate_df.index.get_level_values('mvs_ids').unique())].unq_ids),'m_%s'%filter]=np.round(m,3)
            DF.unq_candidates_df.loc[DF.unq_candidates_df.unq_ids.isin(DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.mvs_ids.isin(candidate_df.index.get_level_values('mvs_ids').unique())].unq_ids),'e_%s'%filter]=np.round(em,3)
            DF.unq_candidates_df.loc[DF.unq_candidates_df.unq_ids.isin(DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.mvs_ids.isin(candidate_df.index.get_level_values('mvs_ids').unique())].unq_ids),'n_%s'%filter]=len(emags)
    if verbose:
        display(DF.unq_candidates_df.loc[DF.unq_candidates_df.unq_ids.isin(DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.mvs_ids.isin(candidate_df.index.get_level_values('mvs_ids').unique())].unq_ids)])
        display(DF.mvs_candidates_df.loc[(DF.mvs_candidates_df.mvs_ids.isin(candidate_df.index.get_level_values('mvs_ids').unique()))])
    return(DF)

def update_candidate_dataframe(DF, unq_ids_list=[], suffix='', verbose=False, workers=None, min_mag_list=[],
                               max_mag_list=[], aptype='4pixels', no_median_tile_update=False, zfactor=10,
                               alignment_box=3, parallel_runs=True, chunksize=None, label='data',
                               kill_plots=True, delta=3, radius=3, skip_filters=[], needs_filters=[], sat_thr=np.inf, mfk=1, mdf=2, mad=0.2, mak=1,
                               PSF_sub_flags='good|unresolved',pruning=False):
    '''
    This is a wrapper for the updating of candidates dataframe and the median tile candidate dataframe

    Parameters
    ----------
    unq_ids_list : list, optional
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
    num_of_chunks : int, optional
        number of chunk to split the targets. The default is None.
    chunksize : int, optional
        size of each chunk for the parallelization process. The default is None.
    mfk: int, optional
        minimum filter detections per KLIPmode to accept a candidate
    mdf: int, optional
        minimum detections per filter to accept a candidate
    mad: int, optional
        minimum arecsecond distance from center to accept a candidate

    Returns
    -------
    None.

    '''
    getLogger(__name__).info('Updating the candidates dataframe')
    if len(unq_ids_list) == 0:
        mvs_ids_list = DF.mvs_targets_df.loc[(
            DF.mvs_targets_df[['flag_%s' % (filter) for filter in DF.filters]].apply(
                lambda x: x.str.contains(PSF_sub_flags, case=False)).any(axis=1))].mvs_ids.unique()
        unq_ids_list_in = DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.mvs_ids.isin(mvs_ids_list)].unq_ids.unique()
        # unq_ids_list_in=DF.unq_candidates_df.unq_ids.unique()
    else:
        unq_ids_list_in = []
        for avg_id in unq_ids_list:
            if avg_id in DF.unq_candidates_df.unq_ids.unique():
                unq_ids_list_in.append(avg_id)

    DF=update_candidates(DF, unq_ids_list=unq_ids_list_in, suffix=suffix, verbose=verbose, aptype=aptype,
                      min_mag_list=min_mag_list, max_mag_list=max_mag_list, workers=workers,
                      parallel_runs=parallel_runs, label=label, kill_plots=kill_plots,
                      delta=delta, radius=radius, skip_filters=skip_filters , needs_filters=needs_filters, sat_thr=sat_thr, mfk=mfk, mdf=mdf, mad=mad, mak=mak)

    # if not no_median_tile_update:
    #     getLogger(__name__).info('Updating the median tiles for the candidates')
    #     # unq_ids_list_in=DF.unq_candidates_df.unq_ids.unique()
    #     update_median_candidates_tile(DF, unq_ids_list=unq_ids_list_in, workers=workers, zfactor=zfactor,
    #                                   alignment_box=alignment_box, parallel_runs=parallel_runs, chunksize=chunksize,
    #                                   label=label, kill_plots=kill_plots, skip_filters=skip_filters)
    if pruning:
        DF = pruning_catalogs(DF)

    return(DF)

def run(packet):
    DF = packet['DF']
    dataset = packet['dataset']

    if dataset.pipe_cfg.klipphotometry['redo']: make_candidates_dataframes(DF)

    DF=update_candidate_dataframe(DF,
                               unq_ids_list=dataset.pipe_cfg.unq_ids_list,
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
                               label=dataset.pipe_cfg.klipphotometry['label'],
                               kill_plots=dataset.pipe_cfg.klipphotometry['kill_plots'],
                               delta=dataset.pipe_cfg.klipphotometry['delta'],
                               radius=dataset.pipe_cfg.klipphotometry['radius'],
                               skip_filters=dataset.pipe_cfg.klipphotometry['skip_filters'],
                               needs_filters=dataset.pipe_cfg.klipphotometry['needs_filters'],
                               sat_thr=dataset.pipe_cfg.klipphotometry['sat_thr'],
                               mfk=dataset.pipe_cfg.klipphotometry['mfk'],
                               mdf=dataset.pipe_cfg.klipphotometry['mdf'],
                               mad=dataset.pipe_cfg.klipphotometry['mad'],
                               mak=dataset.pipe_cfg.klipphotometry['mak'],
                               PSF_sub_flags=dataset.pipe_cfg.psfsubtraction['PSF_sub_flags'],
                               pruning=dataset.pipe_cfg.klipphotometry['pruning'])

    DF.save_dataframes(__name__)