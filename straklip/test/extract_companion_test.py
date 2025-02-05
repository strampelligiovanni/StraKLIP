import sys,os
sys.path.append('/')
sys.path.append('/')
sys.path.append('/')
sys.path.append('//')
from datetime import datetime
import config, input_tables
import matplotlib.pylab as plt
from astropy.io import fits
import astropy.io.fits as pyfits
from pyklip.kpp.utils.mathfunc import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyklip.instruments.Instrument import GenericData
import pyklip.rdi as rdi
import pyklip
import pyklip.fmlib.fmpsf as fmpsf
import pyklip.fm as fm
import pyklip.fitpsf as fitpsf
import pyklip.parallelized as parallelized
import pyklip.fakes as fakes
import pickle

def get_MODEL_from_data(psf,centers, d):
    psf-=np.nanmedian(psf)
    psf[psf < 0] = 0
    PSF = psf[centers[0] - d:centers[0] + d + 1, centers[1] - d:centers[1] + d + 1].copy()
    PSF = np.tile(PSF, (1, 1))
    return(PSF)

def setup_DATASET(DF, id, filter, d, numbasis, KL=1):
    filename = DF.path2out + f'/mvs_tiles/{filter}/tile_ID{id}.fits'
    hdulist = pyfits.open(filename)
    data = hdulist['SCI'].data
    data[data < 0] = 0
    centers = [int((data.shape[1] - 1) / 2), int((data.shape[0] - 1) / 2)]

    # now let's generate a dataset to reduce for KLIP. This contains data at both roll angles
    dataset = GenericData([data], [centers], filenames=[filename])
    PSF = get_MODEL_from_data(hdulist[f'MODEL{KL}'].data, centers, d)
    residuals=np.array([hdulist[f'KMODE{nb}'].data for nb in numbasis])

    return(dataset, PSF, residuals)


def generate_psflib(DF,id,dataset,filter):
    data = dataset.input[0]
    centers = dataset._centers[0]
    psf_list = [data]
    psf_names = [DF.path2out + f'/mvs_tiles/{filter}/tile_ID{id}.fits']
    # pa_v3 = []
    for psfid in DF.mvs_targets_df.loc[(DF.mvs_targets_df[f'flag_{filter}'] == 'good_psf')&(DF.mvs_targets_df.mvs_ids != id)].mvs_ids.unique():
        hdul = fits.open(DF.path2out + f'/mvs_tiles/{filter}/tile_ID{psfid}.fits')
        data = hdul['SCI'].data
        data[data < 0] = 0
        psf_list.append(data)
        psf_names.append(DF.path2out + f'/mvs_tiles/{filter}/tile_ID{psfid}.fits')
        # pa_v3.append(DF.mvs_targets_df.loc[DF.mvs_targets_df.mvs_ids == psfid, f'pav3_{filter}'].values[0])

    # make the PSF library
    # we need to compute the correlation matrix of all images vs each other since we haven't computed it before
    psflib = rdi.PSFLibrary(np.array(psf_list), centers, np.array(psf_names), compute_correlation=True)

    # save the correlation matrix to disk so that we also don't need to recomptue this ever again
    # In the future we can just pass in the correlation matrix into the PSFLibrary object rather than having it compute it
    psflib.save_correlation("/Users/gstrampelli/PycharmProjects/Giovanni/work/analysis/FFP_drc/test/corr_matrix.fits",
                            overwrite=True)

    # now we need to prepare the PSF library to reduce this dataset
    # what this does is tell the PSF library to not use files from this star in the PSF library for RDI
    psflib.prepare_library(dataset)
    return(psflib)

def run_FMAstrometry(dataset,PSF,psflib,filter,separation_pixels,position_angle,guess_flux,numbasis,chaindir,boxsize,dr,exclr,fileprefix,outputdir,fitkernel,corr_len_guess,xrange,yrange,frange,nwalkers,nburn,nsteps,nthreads,wkl):
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
    # You should change these to be suited to your data!
    # run KLIP-FM
    fm.klip_dataset(dataset,
                    fm_class,
                    mode='RDI',
                    outputdir=outputdir,
                    fileprefix=fileprefix,
                    annuli=1,
                    subsections=1,
                    movement=0.,
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

    # Test the postion you are extracting the companion match with the companion
    plt.figure()
    plt.imshow(data_frame[0], origin='lower')
    # plt.plot(x2, y2, 'ok')
    plt.title('Residuals')
    plt.show()

    plt.figure()
    plt.imshow(fm_frame[0], origin='lower')
    # plt.plot(x2, y2, 'ok')
    plt.title('Forwarded model')
    plt.show()

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
                            exclusion_radius=exclr)

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

    con = np.round(fma.fit_flux.bestfit * guess_flux,3)
    econ = np.round(fma.fit_flux.error_2sided * guess_flux,3)
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
    id = 52
    d = 7
    filter = 'f850lp'
    # filter = 'f814w'
    KL=3

    pxsc_arcsec = 0.04
    guess_flux = 1e-1
    guess_spec = 1
    path2throughput_dict = f"/Users/gstrampelli/PycharmProjects/Giovanni/work/analysis/FFP_drc/test/data/ID{id}/contrast_curves/inj_companions/{filter}"
    outputdir = f"/Users/gstrampelli/PycharmProjects/Giovanni/work/analysis/FFP_drc/test/data/ID{id}/extraction/"
    chaindir = f'{outputdir}/chains'
    os.makedirs(chaindir, exist_ok=True)

    boxsize = 5 #35
    dr = 5
    fitkernel = 'diag'
    corr_len_guess = 3.
    corr_len_range = 2.
    xrange = 3.
    yrange = 3.
    frange = 1.
    nwalkers = 50
    nburn = 100
    nsteps = 100
    nthreads = 4
    exclr = 3
    x2, y2 = 18, 21.5  # Companion's coordinates

    seps = [2, 4, 6, 8, 10, 15, 20]

    pyklip.fm.debug = True
    pipe_cfg = '/Users/gstrampelli/PycharmProjects/Giovanni/work/pipeline_logs/FFP_drc/pipe.yaml'
    data_cfg = '/Users/gstrampelli/PycharmProjects/Giovanni/work/pipeline_logs/FFP_drc/data.yaml'
    pipe_cfg = config.configure_pipeline(pipe_cfg, pipe_cfg=pipe_cfg, data_cfg=data_cfg,
                                         dt_string=datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    data_cfg = config.configure_data(data_cfg, pipe_cfg)
    numbasis = np.array(pipe_cfg.psfsubtraction['kmodes'])  # KL basis cutoffs you want to try

    dataset = input_tables.Tables(data_cfg, pipe_cfg)
    DF = config.configure_dataframe(dataset, load=True)

    dataset, PSF, _ = setup_DATASET(DF, id, filter, d, numbasis)
    psflib = generate_psflib(DF, id, dataset, filter)

    separation_pixels, separation_arcsec, position_angle, delta_x, delta_y = get_sep_and_posang(dataset, x2, y2, pxsc_arcsec)

    wkl=np.where(KL == np.array(DF.kmodes))[0]
    fma, con, econ = run_FMAstrometry(dataset,PSF,psflib,filter,separation_pixels,position_angle,guess_flux,numbasis,chaindir,boxsize,dr,exclr, f"{filter}",outputdir,fitkernel,corr_len_guess,xrange,yrange,frange,nwalkers,nburn,nsteps,nthreads,wkl)

    # Plot the MCMC fit results.
    fig = fma.make_corner_plot()
    path = os.path.join(outputdir,f'{filter}-corner.png')
    fig.savefig(path)
    plt.close(fig)

    fig = fma.best_fit_and_residuals()
    path = os.path.join(outputdir,f'{filter}-residuals.png')
    fig.savefig(path)
    plt.close(fig)

    with open(path2throughput_dict+f"/{filter}_algo_throughput_dict.pkl", 'rb') as f:
        algo_throughput_dict = pickle.load(f)
    with open(path2throughput_dict+f"/{filter}_algo_throughput_error_dict.pkl", 'rb') as f:
        algo_throughput_error_dict = pickle.load(f)

    closest_throughput_index = np.argmin(np.abs(separation_pixels - seps))
    algo_throughput = np.round(algo_throughput_dict[KL][closest_throughput_index],2)
    algo_throughput_plus = np.round(algo_throughput_error_dict[KL][0][closest_throughput_index],2)
    algo_throughput_minus = np.round(algo_throughput_error_dict[KL][1][closest_throughput_index],2)

    corr_cc=np.round(con/algo_throughput,3)

    err_corr_cc_plus=np.round(corr_cc*np.sqrt((econ[0]/con)**2+(algo_throughput_plus/algo_throughput)**2),3)
    err_corr_cc_minus=np.round(corr_cc*np.sqrt((econ[1]/con)**2+(algo_throughput_minus/algo_throughput)**2),3)

    print(f'Algorithm throughput as KL {KL} and sep {separation_pixels}: {algo_throughput} +/- [{algo_throughput_plus}, {algo_throughput_plus}] ')
    print(f'Corrected contrast: {corr_cc} +/- [{err_corr_cc_plus}, {err_corr_cc_minus}] ')
    print()