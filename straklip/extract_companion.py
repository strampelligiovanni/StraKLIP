import sys,os

from datetime import datetime
import config, input_tables
import matplotlib.pylab as plt
from astropy.io import fits
import astropy.io.fits as pyfits
from pyklip.kpp.utils.mathfunc import *
from pyklip.instruments.Instrument import GenericData
import pyklip.rdi as rdi
import pyklip
import pyklip.fmlib.fmpsf as fmpsf
import pyklip.fm as fm
import pyklip.fitpsf as fitpsf
import corner


def get_MODEL_from_data(psf,centers, d):
    psf[psf < 0] = 0
    PSF = psf[centers[0] - d:centers[0] + d + 1, centers[1] - d:centers[1] + d + 1].copy()
    PSF = np.tile(PSF, (1, 1))
    return(PSF)

def setup_DATASET(DF, id, filter, numbasis):  # , remove_companion=False):
    filename = DF.path2out + f'/mvs_tiles/{filter}/tile_ID{id}.fits'
    hdulist = pyfits.open(filename)
    data = hdulist['SCI'].data
    data[data < 0] = 0
    centers = [int((data.shape[1] - 1) / 2), int((data.shape[0] - 1) / 2)]

    residuals=np.array([hdulist[f'KMODE{nb}'].data for nb in numbasis])
    dataset = GenericData([data], [centers], filenames=[filename])
    return(dataset, residuals)

def generate_psflib(DF,id,dataset,filter,d,KL=1,dir='./'):
    data = dataset.input[0]
    centers = dataset._centers[0]
    psf_list = [data]
    psf_names = [DF.path2out + f'/mvs_tiles/{filter}/tile_ID{id}.fits']
    models_list = []
    for psfid in DF.mvs_targets_df.loc[(DF.mvs_targets_df[f'flag_{filter}'] == 'good_psf')&(DF.mvs_targets_df.mvs_ids != id)].mvs_ids.unique():
        hdul = fits.open(DF.path2out + f'/mvs_tiles/{filter}/tile_ID{psfid}.fits')
        data = hdul['SCI'].data
        data[data < 0] = 0
        model = hdul[f'MODEL{KL}'].data
        model[model < 0] = 0
        model_peak = np.nanmax(model)
        models_list.append(model/model_peak)
        psf_list.append(data)
        psf_names.append(DF.path2out + f'/mvs_tiles/{filter}/tile_ID{psfid}.fits')
        # pa_v3.append(DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids == psfid, f'pav3_{filter}'].values[0])


    PSF = get_MODEL_from_data(np.median(models_list,axis=0), centers, d)

    # make the PSF library
    # we need to compute the correlation matrix of all images vs each other since we haven't computed it before
    psflib = rdi.PSFLibrary(np.array(psf_list), centers, np.array(psf_names), compute_correlation=True)

    # save the correlation matrix to disk so that we also don't need to recomptue this ever again
    # In the future we can just pass in the correlation matrix into the PSFLibrary object rather than having it compute it
    psflib.save_correlation(dir,overwrite=True)

    # now we need to prepare the PSF library to reduce this dataset
    # what this does is tell the PSF library to not use files from this star in the PSF library for RDI
    psflib.prepare_library(dataset)
    # return(psflib,psf_list)
    return(psflib,psf_list,PSF)

def run_FMAstrometry(dataset,PSF,psflib,filter,separation_pixels,position_angle,guess_flux,guess_contrast,numbasis,chaindir,boxsize,dr,fileprefix,outputdir,fitkernel,corr_len_guess,corr_len_range,xrange,yrange,frange,nwalkers,nburn,nsteps,nthreads,wkl):
    # ####################################################################################################
    # setup FM guesses
    # initialize the FM Planet PSF class
    fm_class = fmpsf.FMPlanetPSF(inputs_shape=dataset.input.shape,
                                 input_wvs=[1],
                                 numbasis=numbasis,
                                 sep=separation_pixels,
                                 pa=position_angle,
                                 dflux=guess_flux,
                                 input_psfs=np.array([PSF]))

    # PSF subtraction parameters
    # run KLIP-FM
    fm.klip_dataset(dataset,
                    fm_class,
                    mode='RDI',
                    outputdir=outputdir,
                    fileprefix=fileprefix,
                    annuli=1,
                    subsections=1,
                    movement=1.,
                    numbasis=numbasis,
                    maxnumbasis=np.nanmax(numbasis),
                    aligned_center=dataset._centers[0],
                    psf_library=psflib,
                    mute_progression=True,
                    corr_smooth=0)

    # Open the FM dataset.
    with fits.open(outputdir + f"{fileprefix}-fmpsf-KLmodes-all.fits") as hdul:
        fm_frame = hdul[0].data[wkl]
        fm_centx = hdul[0].header['PSFCENTX']
        fm_centy = hdul[0].header['PSFCENTY']
    with fits.open(outputdir + f"{fileprefix}-klipped-KLmodes-all.fits") as hdul:
        data_frame = hdul[0].data[wkl]
        data_centx = hdul[0].header['PSFCENTX']
        data_centy = hdul[0].header['PSFCENTY']

    # Initialize pyKLIP FMAstrometry class.
    fma = fitpsf.FMAstrometry(guess_sep=separation_pixels,
                              guess_pa=position_angle,
                              fitboxsize=boxsize)
    fma.generate_fm_stamp(fm_image=fm_frame[0],
                          fm_center=[fm_centx, fm_centy],
                          padding=5)
    fma.generate_data_stamp(data=data_frame[0],
                            data_center=[data_centx, data_centy],
                            dr=dr,
                            exclusion_radius=5)

    corr_len_label = r'$l$'
    fma.set_kernel(fitkernel, [corr_len_guess], [corr_len_label])
    fma.set_bounds(xrange, yrange, frange, [corr_len_range])


    # Make sure that the noise map is invertible.
    noise_map_max = np.nanmax(fma.noise_map)
    fma.noise_map[np.isnan(fma.noise_map)] = noise_map_max
    fma.noise_map[fma.noise_map == 0.] = noise_map_max

    # Run the MCMC fit.
    chain_output = os.path.join(chaindir, f'{filter}-bka_chain.pkl')
    fma.fit_astrometry(nwalkers=nwalkers,
                       nburn=nburn,
                       nsteps=nsteps,
                       numthreads=nthreads,
                       chain_output=chain_output)

    con = np.round(fma.fit_flux.bestfit * guess_contrast,3)
    econ = np.round(fma.fit_flux.error_2sided * guess_contrast,3)
    xshift = -(fma.raw_RA_offset.bestfit - delta_x)
    yshift = fma.raw_Dec_offset.bestfit - delta_y
    print(f'dx: {xshift}, dy: {yshift}, contrast: {con}, error: {econ}')
    return(fma,con,econ)

def get_sep_and_posang(dataset,x2,y2,pxsc_arcsec):
    # Coordinates of the primary and companion in pixels
    x1, y1 = dataset._centers[0][0], dataset._centers[0][1]  # Primary star's coordinates

    # Calculate separation in pixels
    separation_pixels = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # Convert separation to arcseconds
    separation_arcsec = separation_pixels * pxsc_arcsec

    # Calculate position angle in degrees
    # Position angle is measured from north to east
    delta_x = x1 - x2
    delta_y = y2 - y1
    position_angle = np.degrees(np.arctan2(delta_x, delta_y))# % 360

    print(f"Separation: {separation_arcsec:.2f} arcsec")
    print(f"Position Angle: {position_angle:.2f} degrees")
    return(separation_pixels, separation_arcsec, position_angle, delta_x, delta_x)

if __name__ == "__main__":
    id = 16
    # xycomp_list= [[None, None],[None, None]]
    xycomp_list= [[None, None]]

    # filters = ['f814w', 'f850lp']
    filters = ['f850lp']
    KL=7

    pxsc_arcsec = 0.04
    guess_contrast = 1e-1
    outputdir = f"/Users/gstrampelli/PycharmProjects/FFP_binaries/out/extraction/ID{id}/"
    chaindir = f'{outputdir}/chains'
    os.makedirs(chaindir, exist_ok=True)

    pyklip.fm.debug = True
    pipe_cfg = '/Users/gstrampelli/PycharmProjects/FFP_binaries/pipeline_logs/pipe.yaml'
    data_cfg = '/Users/gstrampelli/PycharmProjects/FFP_binaries/pipeline_logs/data.yaml'
    pipe_cfg = config.configure_pipeline(pipe_cfg, pipe_cfg=pipe_cfg, data_cfg=data_cfg,
                                         dt_string=datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    data_cfg = config.configure_data(data_cfg, pipe_cfg)

    dataset = input_tables.Tables(data_cfg, pipe_cfg)
    DF = config.configure_dataframe(dataset, load=True)
    numbasis = np.array(DF.kmodes)

    for elno in range(len(filters)):
        filter=filters[elno]
        xcomp, ycomp = xycomp_list[elno]
        dataset, residuals = setup_DATASET(DF, id, filter, numbasis)
        psflib, psf_list, PSF = generate_psflib(DF, id, dataset, filter, d=7, KL=50 , dir = outputdir+f"{filter}_corr_matrix.fits")

        if np.all([x is None for x in [xcomp, ycomp]]):
            max_value = np.nanmax(np.median(residuals, axis=0))
            max_index = np.where(np.median(residuals, axis=0) == max_value)
            xcomp, ycomp = [max_index[1][0], max_index[0][0]]

        separation_pixels, separation_arcsec, position_angle, delta_x, delta_y = get_sep_and_posang(dataset, xcomp, ycomp, pxsc_arcsec)

        guess_flux = np.nanmax(dataset.input[0])*guess_contrast
        wkl=np.where(KL == np.array(numbasis))[0]
        fma, con, econ = run_FMAstrometry(dataset,PSF,psflib,filter,separation_pixels,position_angle,guess_flux,guess_contrast,
                                          numbasis,chaindir,boxsize=7,dr=5,
                                          fileprefix=f"{filter}",outputdir=outputdir,
                                          fitkernel='diag',corr_len_guess=3, corr_len_range=2,
                                          xrange=3,yrange=3,frange=1,nwalkers=100,
                                          nburn=500,nsteps=1000,nthreads=4,
                                          wkl=np.where(KL == np.array(numbasis))[0])

        fma.sampler.flatchain[:, 2] *= guess_contrast
        # Plot the MCMC fit results.
        all_labels = [r"x", r"y", r"$\alpha$"]
        all_labels = np.append(all_labels, fma.covar_param_labels)
        fig = corner.corner(fma.sampler.flatchain, labels=all_labels, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt='.4f')
        path = os.path.join(outputdir,f'{filter}-corner.png')
        fig.savefig(path)
        plt.close(fig)

        fig = fma.best_fit_and_residuals()
        path = os.path.join(outputdir,f'{filter}-residuals.png')
        fig.savefig(path)
        plt.close(fig)