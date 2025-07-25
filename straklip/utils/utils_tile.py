"""
utilities functions that can be use by or with the tile class
"""
import os,math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pyklip.rdi as rdi
import pyklip.parallelized as parallelized
from straklip.tiles import Tile
from straklip.utils.ancillary import truncate_colormap
from straklip.stralog import getLogger
from scipy.ndimage import zoom,rotate,fourier_shift
from skimage.registration import phase_cross_correlation
from functools import reduce
from astropy.visualization import simple_norm
from pyklip.instruments.Instrument import GenericData
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from reproject.mosaicking import find_optimal_celestial_wcs

# TODO: this function is rotating too much, it should only rotate the image to make North up and East left, not to make the image square
def rotate_fits_north_up_east_left(input_fits, ext=1,hext=1):
    """
    Rotate a FITS image so North is up and East is left, pad to keep square, and save to file.

    Parameters
    ----------
    input_fits : str
        Path to input FITS file.
    output_fits : str
        Path to output FITS file.
    ext : int
        FITS data extension to use (default: 1).
    hext : int
        FITS header extension to use (default: 1).
    """
    # Open FITS and get data/WCS
    with fits.open(input_fits) as hdul:
        data = hdul[ext].data
        header = hdul[hext].header
        wcs_in = WCS(header)

        # Find optimal WCS for North up, East left, keeping square shape
        wcs_out, shape_out = find_optimal_celestial_wcs([(data, wcs_in)], auto_rotate=True)
        # # Make sure output is square
        max_dim = max(shape_out)
        shape_out = (max_dim, max_dim)

        # Reproject
        array, _ = reproject_interp((data, wcs_in), wcs_out, shape_out=shape_out)

    return array

# def allign_images(target_images,rot_angles,PAV_3s,filter,fig=None,ax=None,shift_list=None,cmap='Greys_r',tile_base=15,inst='WFC3',simplenorm='linear',min_percent=0,max_percent=100,power=1,log=1000,xy_m=True,xy_cen=False,legend=False,showplot=False,verbose=False,cbar=True,title='',xy_dmax=None,zfactor=10,alignment_box=0,step=1,Python_origin=True,method='median',kill=False,kill_plots=True,mk_arrow=False):
#     '''
#     shift, derotate and allign each image in the input list of images to the first one
#
#     Parameters
#     ----------
#     target_images : list
#         list of input images.
#     PAV_3s : float
#         list of position angles to use to derotate each image.
#     filter : str
#         filter name.
#     fig : matplotlib figure, optional
#         figure for plot. The default is None.
#     ax : matplotlib axis, optional
#         axixs for plot. The default is None.
#     shift_list : list, optional
#         input list of shift. The default is None.
#     cmap : str, optional
#         color map. The default is 'Greys_r'.
#     tile_base : TYPE
#         side of the square tile. MUST be odd
#     inst : str
#         instrument related to the data input.
#     simplenorm : TYPE, optional
#         use simplenorm normalization. The default is None.
#     min_percent: float, optional
#         The percentile value used to determine the pixel value of minimum cut level.
#         The default is 0.0. min_percent overrides percent.
#     max_percent: float, optional
#         The percentile value used to determine the pixel value of maximum cut level.
#         The default is 100.0. max_percent overrides percent.
#     power: float, optional
#         The power index for stretch='power'. The default is 1.0.
#     log: float, optional
#         The log index for stretch='log'. The default is 1000.
#     xy_m : bool, optional
#         choose to look for the position of the maximum in the tile.
#         The default is True.
#     xy_cen : bool, optional
#         choose to look for the position of the centroid in the tile.
#         The default is False.
#     legend : bool, optional
#         choose to show legends in plots. The default is False.
#     showplot : bool, optional
#         choose to show plots. The default is False.
#     verbose : bool, optional
#         choose to show prints. The default is False.
#     cbar : bool, optional
#         choose to show color bar in the plot. The default is True.
#     title : str, optional
#         title for the plot. The default is ''.
#     xy_dmax: int, optional
#         distance from the center to look for maximum. The default is 2
#     zfactor : int, optional
#         zoom factor to apply to re/debin each image.
#         The default is 10.
#     alignment_box : bool, optional
#         half base of the square box centered at the center of the tile employed by the allignment process to allign the images.
#         The box is constructed as alignment_box-x,x+alignment_box and alignment_box-y,y+alignment_box, where x,y are the
#         coordinate of the center of the tile.
#         The default is 2.
#     step : int, optional
#         DESCRIPTION. The default is 1.
#     Python_origin : bool
#         Choose to specify the origin of the xy input coordinates. For exmaple python array star counting from 0,
#         so a position obtained on a python image will have 0 as first pixel.
#         On the other hand, normal catalogs start counting from 1 (see coordinate on ds9 for example)
#         so we need to subtract 1 to make them compatible when we use those coordinates on python
#         The default is True
#     metod : str, optional
#         choose between median image or mean image. The default is median.
#     kill : bool, optional
#         choose to kill bad pixels instead of using the median of the neighbouring pixels. The default is False.
#     kill_plots:
#         choose to kill all plots created. The default is False.
#
#     Returns
#     -------
#     None.
#
#     '''
#     if xy_dmax!=None: xy_dmax=xy_dmax*zfactor
#
#     if len(target_images)>1:
#         rotated_images,rotated_angles=rotate_images(target_images,rot_angles=rot_angles,zfactor=zfactor)
#         shifted_images,shift_list=shift_images(rotated_images,zfactor=zfactor,alignment_box=alignment_box,shift_list_in=shift_list)
#     else:
#         rotated_images, rotated_angles = rotate_images(target_images, rot_angles=rot_angles, zfactor=1)
#         shifted_images = rotated_images.copy()
#
#     for elno in range(len(shifted_images)):
#         shifted_images[elno][shifted_images[elno] == 0] = np.nan
#
#     if method == 'median':
#         image = np.nanmedian(shifted_images, axis=0)
#     elif method == 'mean':
#         image = np.nanmean(shifted_images, axis=0)
#     else:
#         raise ValueError('method MUST be either median or mean.')
#
#     x=int((image.shape[1]-1)/2)
#     y=int((image.shape[0]-1)/2)
#
#     tile=Tile(data=image,x=x,y=y,tile_base=tile_base,inst=inst,Python_origin=Python_origin)
#
#     if np.all(np.isnan(image)): tile.mk_tile(fig=fig,ax=ax,verbose=verbose,cmap=cmap,showplot=showplot,keep_size=False,title=title,step=step,kill=kill,kill_plots=kill_plots)
#     else: tile.mk_tile(fig=fig,ax=ax,verbose=verbose,min_percent=min_percent,max_percent=max_percent,power=power,log=log,simplenorm=simplenorm,cmap=cmap,xy_m=xy_m,xy_cen=xy_cen,legend=legend,showplot=showplot,keep_size=False,xy_dmax=xy_dmax,cbar=cbar,title=title,step=step,kill=kill,kill_plots=kill_plots)
#     return(tile,shift_list)

def merge_images(rotated_images,method='median',tile_base=15,inst='WFC3',fig=None,ax=None,cmap='Greys_r',min_percent=0,max_percent=100,power=1,log=1000,simplenorm='linear',xy_m=True,xy_cen=False,legend=False,showplot=True,verbose=False,cbar=True,title='',step=1,Python_origin=True,xy_dmax=None,kill=False,kill_plots=True):
    '''
    Align a list of rotated images by taking the median or mean of the images
    Parameters
    ----------
    rotated_images : numpy ndarray
        list of rotated images.
    method : str, optional
        choose between median image or mean image. The default is 'median'.
    tile_base : int, optional
        side of the square tile. MUST be odd. The default is 15.
    inst : str, optional
        instrument related to the data input. The default is 'WFC3'.
    fig : matplotlib figure, optional
        figure for plot. The default is None.
    ax : matplotlib axis, optional
        axis for plot. The default is None.
    cmap : str, optional
        color map. The default is 'Greys_r'.
    min_percent: float, optional
        The percentile value used to determine the pixel value of minimum cut level.
        The default is 0.0. min_percent overrides percent.
    max_percent: float, optional
        The percentile value used to determine the pixel value of maximum cut level.
        The default is 100.0. max_percent overrides percent.
    power: float, optional
        The power index for stretch='power'. The default is 1.0.
    log: float, optional
        The log index for stretch='log'. The default is 1000.
    simplenorm : str, optional
        use simplenorm normalization. The default is 'linear'.
    xy_m : bool, optional
        choose to look for the position of the maximum in the tile.
        The default is True.
    xy_cen : bool, optional
        choose to look for the position of the centroid in the tile.
        The default is False.
    legend : bool, optional
        choose to show legends in plots. The default is False.
    showplot : bool, optional
        choose to show plots. The default is True.
    verbose : bool, optional
        choose to show prints. The default is False.
    cbar : bool, optional
        choose to show color bar in the plot. The default is True.
    title : str, optional
        title for the plot. The default is ''.
    step : int, optional
        step for the plot. The default is 1.
    Python_origin : bool
        Choose to specify the origin of the xy input coordinates. For exmaple python array star counting from 0,
        so a position obtained on a python image will have 0 as first pixel.
        On the other hand, normal catalogs start counting from 1 (see coordinate on ds9 for example)
        so we need to subtract 1 to make them compatible when we use those coordinates on python
        The default is True
    xy_dmax: int, optional
        distance from the center to look for maximum. The default is None.
    kill : bool, optional
        choose to kill bad pixels instead of using the median of the neighbouring pixels. The default is False.
    kill_plots:
        choose to kill all plots created. The default is True.
    Returns
    -------
    tile : Tile
    '''
    if method == 'median':
        image = np.nanmedian(rotated_images, axis=0)
    elif method == 'mean':
        image = np.nanmean(rotated_images, axis=0)
    else:
        raise ValueError('method MUST be either median or mean.')

    x=int((image.shape[1]-1)/2)
    y=int((image.shape[0]-1)/2)

    tile=Tile(data=image,x=x,y=y,tile_base=tile_base,inst=inst,Python_origin=Python_origin)

    if np.all(np.isnan(image)): tile.mk_tile(fig=fig,ax=ax,verbose=verbose,cmap=cmap,showplot=showplot,keep_size=False,title=title,step=step,kill=kill,kill_plots=kill_plots)
    else: tile.mk_tile(fig=fig,ax=ax,verbose=verbose,min_percent=min_percent,max_percent=max_percent,power=power,log=log,simplenorm=simplenorm,cmap=cmap,xy_m=xy_m,xy_cen=xy_cen,legend=legend,showplot=showplot,keep_size=False,xy_dmax=xy_dmax,cbar=cbar,title=title,step=step,kill=kill,kill_plots=kill_plots)
    return(tile)

def flatten_tile_axes(array):
    """
    returns the array with the final two axes - assumed to be the image pixels - flattened
    """
    if not isinstance(array,np.ndarray):array=np.array(array)
    shape = array.shape
    # imshape = shape[-2:]
    newshape = [i for i in list(shape[:-2])]

    newshape += [reduce(lambda x,y: x*y, shape[-2:])]
    return array.reshape(newshape)

def load_image(image,filter,fig=None,ax=None,cmap='Greys_r',title='',tile_base=15,inst='WFC3',simplenorm='linear',min_percent=0,max_percent=100,power=1,log=1000,xy_m=True,xy_cen=False,legend=False,cbar=True,showplot=True,Python_origin=True):
    '''
    Load and show target image tile

    Parameters
    ----------
    image : numpy ndarray
        target input image tile.
    filter : str
        filter related to the data input.
    fig : matplotlib, optional
        outside figure. The default is None
    ax : matplotlib, optional
        outside axis. The default is None.
    cmap : str, optional
        color map. The default is 'Greys_r'.
    title : str, optional
        title for the plot. The default is ''.
    tile_base : TYPE
        side of the square tile. MUST be odd
    inst : str
        instrument related to the data input.
    simplenorm : TYPE, optional
        use simplenorm normalization. The default is None.
    min_percent: float, optional
        The percentile value used to determine the pixel value of minimum cut level. 
        The default is 0.0. min_percent overrides percent.
    max_percent: float, optional
        The percentile value used to determine the pixel value of maximum cut level. 
        The default is 100.0. max_percent overrides percent.
    power: float, optional
        The power index for stretch='power'. The default is 1.0.
    log: float, optional
        The log index for stretch='log'. The default is 1000.
    xy_m : bool, optional
        choose to look for the position of the maximum in the tile. 
        The default is True.
    xy_cen : bool, optional
        choose to look for the position of the centroid in the tile. 
        The default is False.
    legend : bool, optional
        choose to show legends in plots. The default is False.
    cbar : bool, optional
        choose to show color bar in the plot. The default is True.
    showplot : bool, optional
        choose to show plots. The default is True.
    Python_origin : bool
        Choose to specify the origin of the xy input coordinates. For exmaple python array star counting from 0, 
        so a position obtained on a python image will have 0 as first pixel. 
        On the other hand, normal catalogs start counting from 1 (see coordinate on ds9 for example) 
        so we need to subtract 1 to make them compatible when we use those coordinates on python
        The default is True

    Returns
    -------
    None.

    '''
    # print(filter)
    tile=Tile(data=image,x=int((tile_base-1)/2),y=int((tile_base-1)/2),tile_base=tile_base,inst=inst,Python_origin=Python_origin)
    tile.mk_tile(fig=fig,ax=ax,pad_data=False,log=log,simplenorm=simplenorm,max_percent=max_percent,min_percent=min_percent,power=power,cmap=cmap,xy_m=xy_m,xy_cen=xy_cen,legend=legend,showplot=showplot,keep_size=True,xy_dmax=3,cbar=cbar,title=title)
    return(tile)

def make_tile_from_flat(flat, indices=None, shape=None, squeeze=True):
    """
    put the flattened region back into an image. if no indices or shape are specified, assumes that
    the region of N pixels is a square with Nx = Ny = sqrt(N). Only operates on the last axis.
    Input:
        flat: [Ni,[Nj,[Nk...]]] x Npix array (any shape as long as the last dim is the pixels)
        indices: [None] Npix array of flattened pixel coordinates 
                 corresponding to the region
        shape: [None] image shape
        squeeze [True]: gets rid of extra axes 
    Returns:
        img: an image (or array of) with dims `shape` and with nan's in 
            whatever indices are not explicitly set
    """
    # sometimes you get just a number, e.g. if the image is a NaN
    if np.ndim(flat) == 0:
        return flat
    oldshape = flat.shape[:]
    if shape is None:
        # assume that you have been given the full square imae
        Npix = oldshape[-1]
        Nside = int(np.sqrt(Npix))
        indices = np.array(range(Npix))
        shape = (Nside, Nside)
        return flat.reshape(oldshape[:-1]+shape)

    img = np.ravel(np.zeros(shape))*np.nan
    # this is super memory inefficient
    # handle the case of region being a 2D array by extending the img axes
    if flat.ndim > 1:
        # assume last dimension is the pixel
        flat = np.reshape(flat, (reduce(lambda x,y: x*y, oldshape[:-1]), oldshape[-1]))
        img = np.tile(img, (flat.shape[0], 1))
    else:
        img = img[None,:]
    # fill in the image
    img[:,indices] = flat
    # reshape and get rid of extra axes, if any
    img = img.reshape(list(oldshape[:-1])+list(shape))
    if squeeze == True:
        img = np.squeeze(img)
    return img

def setup_DATASET(DF, id, filter, d):
    filename = DF.path2out + f'/mvs_tiles/{filter}/tile_ID{id}.fits'
    hdulist = pyfits.open(filename)
    data = hdulist['SCI'].data
    data[data < 0] = 0
    centers = [int((data.shape[1] - 1) / 2), int((data.shape[0] - 1) / 2)]

    # now let's generate a dataset to reduce for KLIP. This contains data at both roll angles
    dataset = GenericData([data], [centers], filenames=[DF.path2out + f'/mvs_tiles/{filter}/tile_ID52.fits'])

    return(dataset)

def perform_PSF_subtraction(targ_tiles,ref_tiles,filename,psfnames='',psflib=None,kmodes=[],maxnumbasis=None,outputdir='./',prefix='parallel'):
    '''
    Perform KLIP subtraction on all the stamps for one star. Since stamps
    of the same star share the same references, this computes the Z_k's for
    a star and then projects all the star stamps onto them.

    Parameters
    ----------
    targ_tiles : numpy ndarray
        tile on which perform PSF subtraction.
    ref_tiles : numpy ndarray
        reference tiles to use for the PSF library.
    kmodes : list, optional
        list of KLIP modes to use in the PSF subtraction. If empty, use all The default is [].
    filename: path to the target tile
    psfnames: path to the target tiles used as psfs
    outputdir: path to the temporary dir to save klip output

    Returns
    -------
    (residuals,psf_models).

    '''
    centers = [int((targ_tiles.shape[1] - 1) / 2), int((targ_tiles.shape[0] - 1) / 2)]
    # now let's generate a dataset to reduce for KLIP. This contains data at both roll angles
    dataset = GenericData([targ_tiles], [centers], filenames=[filename])

    if psflib is None:
        # make the PSF library
        # we need to compute the correlation matrix of all images vs each other since we haven't computed it before
        psflib = rdi.PSFLibrary(ref_tiles, centers, np.array(psfnames), compute_correlation=True)

    # now we need to prepare the PSF library to reduce this dataset
    # what this does is tell the PSF library to not use files from this star in the PSF library for RDI
    psflib.prepare_library(dataset)

    if maxnumbasis is None:
        maxnumbasis = len(ref_tiles)

    parallelized.klip_dataset(dataset,
                              mode='RDI',
                              outputdir=outputdir,
                              fileprefix=prefix,
                              annuli=1,
                              subsections=1,
                              movement=0.,
                              numbasis=kmodes,
                              maxnumbasis=maxnumbasis,
                              aligned_center=dataset._centers[0],
                              psf_library=psflib,
                              corr_smooth=0,
                              verbose=False)

    hdulist = fits.open(f'{outputdir}/{prefix}-KLmodes-all.fits')
    residuals = hdulist[0].data
    hdulist.close()
    psf_models = np.array([targ_tiles - res for res in residuals])
    if os.path.exists(f'{outputdir}/{prefix}-KLmodes-all.fits'):
        os.remove(f'{outputdir}/{prefix}-KLmodes-all.fits')

    return(residuals,psf_models)


def psf_tile_from_basis(target, kl_basis, numbasis=None):
    """
    Generate a model PSF from the KLIP basis vectors. See Soummer et al 2012.

    Parameters
    ----------
    target : np.array
      the target PSF, 1-D
    kl_basis : np.array
      Nklip x Npix array
    numbasis : int or np.array [None]
      number of Kklips to use. Default is None, meaning use all the KL vectors

    Output
    ------
    psf_model : np.array
      Nklip x Nrows x Ncols array of the model PSFs. Dimensionality depends on the value
      of num_basis
    """
    # make sure numbasis is an integer array
    if numbasis is None:
        numbasis = len(kl_basis)
    if isinstance(numbasis, int):
        numbasis = np.array([numbasis])
    numbasis = numbasis.astype(int)

    coeffs = np.inner(target, kl_basis)
    psf_model = kl_basis * np.expand_dims(coeffs, [i+1 for i in range(kl_basis.ndim-1)])
    psf_model = np.array([np.sum(psf_model[:k], axis=0) for k in numbasis])

    return np.squeeze(psf_model)



def rotate_images(input_images,rot_angles=[0],zfactor=10):
    '''
    Rotate target tile by a given angle

    Parameters
    ----------
    input_images : numpy ndarray
        list of input target tiles.
    rot_angles : float, optional
        list of angles for each target input. The default is [0].
    zfactor : int, optional
        zoom factor to apply to rebin each image befor rotate. The default is 10.

    Returns
    -------
    (list of rotated images).

    '''
    elno=0
    rotated_images=[]
    rotated_angles=[]
    for image in input_images:
        rebinned_image=zoom(image,zfactor,order=0)
        rotated_image=rotate(rebinned_image,-rot_angles[elno],reshape=False,mode='nearest')
        rotated_image[np.isnan(rotated_image)]=0
        rotated_images.append(rotated_image)
        rotated_angles.append(rot_angles[elno])
        elno+=1
    return(np.array(rotated_images),np.array(rotated_angles))

def shift_target_image(reference,target,target_box=None,shift=None,zfactor=10):
    '''
    evaluate shift between a reference and a target image and then shift the target image 

    Parameters
    ----------
    reference : numpy ndarray
        reference image.
    target : numpy ndarray
        target image.
    target_box : list
        list of xmin,xmax,ymin,ymax values to confin the alignment between 
        the reference and the target. If non use the whole tile. Default is None.
    shift : float, optional
        input shift. If not None use this instead of evaluating a new one. 
        The default is None.
    zfactor : int, optional
        zoom factor to apply to debin each image after the shift is applied. 
        The default is 10.

    Returns
    -------
    (shifted debinned target image, shift).

    '''
    if not isinstance(shift,(list,np.ndarray)): 
        if isinstance(target_box,(list,np.ndarray)): 
            xmin,xmax,ymin,ymax=target_box
            shift, _, _ = phase_cross_correlation(reference[xmin:xmax,ymin:ymax], target[xmin:xmax,ymin:ymax])
        else: shift, _, _ = phase_cross_correlation(reference, target)
    shifted_image = fourier_shift(np.fft.fftn(target), shift)
    shifted_image = np.fft.ifftn(shifted_image)
    shifted_image=shifted_image.real
    shifted_image[np.isnan(shifted_image)]=0
    debinned_image=zoom(shifted_image,1/zfactor,order=0)
    return(debinned_image,shift)

def shift_images(input_images,zfactor=10,alignment_box=0,shift_list_in=None):
    '''
    Shift each image in list taking the first as reference

    Parameters
    ----------
    input_images : numpy ndarray
        list of input images.
    zfactor : int, optional
        zoom factor to apply to re/debin each image. 
        The default is 10.
    alignment_box : bool, optional
        half base of the square box centered at the center of the tile employed by the allignment process to allign the images. 
        The box is constructed as alignment_box-x,x+alignment_box and alignment_box-y,y+alignment_box, where x,y are the 
        coordinate of the center of the tile.
        The default is 0.
    shift_list_in : TYPE, optional
        input list of shifts. If not None use these instead of evaluating new one. 
        The default is None

    Returns
    -------
    None.

    '''
    shift_list=[0]
    image0=input_images[0]
    debinned_image0=zoom(image0,1/zfactor,order=0)
    shifted_images=[debinned_image0]
    if alignment_box >0 :
        xmin=int(image0.shape[0]/2)-int(alignment_box/2*zfactor)
        xmax=int(image0.shape[0]/2)+int(alignment_box/2*zfactor)
        ymin=int(image0.shape[1]/2)-int(alignment_box/2*zfactor)
        ymax=int(image0.shape[1]/2)+int(alignment_box/2*zfactor)

    for elno in range(1,len(input_images)):
        image=input_images[elno]
        if alignment_box >0 :
            image_box=[xmin,xmax,ymin,ymax]
        else:
            image_box=None
        if isinstance(shift_list_in, (list,np.ndarray)):
            shifted_image,shift=shift_target_image(image0,image,target_box=image_box,shift=shift_list_in[elno],zfactor=zfactor)
        else:
            shifted_image,shift=shift_target_image(image0,image,target_box=image_box,shift=None,zfactor=zfactor)
        shifted_images.append(shifted_image)
        shift_list.append(shift)

    return(np.array(shifted_images),shift_list)

def show_binary_PA(binary_df,DF=None,path2dir='',path2fits='',tag_label=None,label_id_p='unq_ids_p',label_id_c='unq_ids_c',label_dict_p='crclean_data',label_dict_c='companion',label_dict=None,primary_tile_label=None,KLIP_tile_label=None,Tag='ID',path2tile=None,xtile=None,ytile=None,tile_base=None,inst=None,tile_name=None,ytext=5,dim=3,ncols=5,row_count=0,column_count=0,skip_filters=['F658N'],no_ids=False,cmap='Greys',color='w',save_name=None,path2savedir='./',max_row=-1,show_more_text=False,show_extra_text=False,percent=99.,percent2=99.,simplenorm=None,simplenorm2=None,norm=None,norm2=None,fontsize=20,truncate=False,tinf=0.1,tsup=1,vmin=None,vmin2=None,vmax=None,vmax2=None,force_wide_KLIP=False,wide=False):
    nrows=int(binary_df[label_id_p].nunique()/ncols)
    if nrows < binary_df[label_id_p].nunique()/ncols: nrows+=1
    nrows=nrows*2
    if max_row!=-1: nrows=max_row
    fig,axes=plt.subplots(nrows,ncols,figsize=(ncols*dim,nrows*dim),squeeze=False,sharex=False,sharey=False, gridspec_kw={'hspace': 0,'wspace': 0})
    n_row=0
    if not isinstance(label_dict, dict) :label_dict={'data':1,'crclean_data':2,'companion':3}
    if truncate==True:
        cmap = plt.get_cmap(cmap)
        cmap = truncate_colormap(cmap, tinf,tsup)
    
    for unq_ids_p in binary_df[label_id_p].unique():
        if (primary_tile_label ==None and KLIP_tile_label ==None):
            if DF!=None:
                if not wide:filters_sel_list=np.array(DF.filters_list)[[i not in skip_filters for i in DF.filters_list]][~(DF.unq_candidates_df.loc[(DF.unq_candidates_df.unq_ids==unq_ids_p),['e%s'%i[1:4] for i in DF.filters_list if i not in skip_filters]].isna()).values[0]]
                else:filters_sel_list=np.array(DF.filters_list)[[i not in skip_filters for i in DF.filters_list]][~(DF.unq_targets_df.loc[(DF.unq_targets_df.unq_ids==unq_ids_p),['e%s'%i[1:4] for i in DF.filters_list if i not in skip_filters]].isna()).values[0]]
                q=np.where(DF.unq_targets_df.loc[(DF.unq_targets_df.unq_ids==unq_ids_p),['spx%s'%i[1:4] for i in filters_sel_list if i not in skip_filters]].values[0]==np.nanmin(DF.unq_targets_df.loc[(DF.unq_targets_df.unq_ids==unq_ids_p),['spx%s'%i[1:4] for i in filters_sel_list if i not in skip_filters]].values[0]))[0][-1]
                filter=filters_sel_list[q]
                if path2tile==None: path2tile_temp='%s/%s/%s/%s/median_tiles/%s/tile_ID%i.fits'%(path2data,DF.project,DF.target,DF.inst,filter,unq_ids_p)
                else:path2tile_temp=path2tile+tile_name+'%i.fits'%unq_ids_p
                xtile,ytile=[int((DF.tile_base-1)/2),int((DF.tile_base-1)/2)]
                tile_base=DF.tile_base
                inst=DF.inst
            else:
                path2tile_temp=path2tile+tile_name+'%i.fits'%unq_ids_p

        if primary_tile_label ==None:
            DATA=Tile(x=xtile,y=ytile,tile_base=tile_base,inst=inst)
            DATA.load_tile(path2tile_temp,ext=label_dict[label_dict_p],verbose=False,return_Datacube=True)
            DATA.mk_tile(fig=fig,ax=axes[row_count][column_count],showplot=False,keep_size=True,simplenorm='sqrt',return_tile=False)
        else:
            data=binary_df.loc[binary_df[label_id_p]==unq_ids_p,primary_tile_label].values[0]
            DATA=Tile(x=xtile,y=ytile,tile_base=tile_base,inst=inst,data=data)
            DATA.mk_tile(fig=fig,ax=axes[row_count][column_count],showplot=False,keep_size=True,simplenorm='sqrt',return_tile=False)

        if tag_label!=None:tag_label_temp=binary_df.loc[(binary_df[label_id_p]==unq_ids_p),tag_label].values[0]
        else: tag_label_temp=unq_ids_p
        if no_ids==False:axes[row_count][column_count].text(1,ytext,'%s%i'%(Tag,tag_label_temp),fontsize=fontsize,color=color)
        
        axes[row_count][column_count].axis('off')
        column_count+=1
        if (np.isnan(binary_df.loc[(binary_df[label_id_p]==unq_ids_p)][label_id_c].values[0]) or (binary_df.loc[(binary_df[label_id_p]==unq_ids_p)][label_id_c].values[0]==-1)) or force_wide_KLIP:
            if KLIP_tile_label ==None:
                DATA=Tile(x=xtile,y=ytile,tile_base=tile_base,inst=inst)
                DATA.load_tile(path2tile_temp,ext=label_dict[label_dict_c],verbose=False,return_Datacube=True)
                DATA.mk_tile(fig=fig,ax=axes[row_count][column_count],showplot=False,keep_size=True,simplenorm='sqrt',return_tile=False)
                if force_wide_KLIP and not np.all(binary_df.loc[(binary_df[label_id_p]==unq_ids_p)][label_id_c].isna()):
                    # Create a Rectangle patch
                    rect = patches.Rectangle((0,0), DATA.tile_base-1, DATA.tile_base-1, linewidth=5, edgecolor='r', facecolor='none')
                    # Add the patch to the Axes
                    axes[row_count][column_count].add_patch(rect)
            else:
                data=binary_df.loc[binary_df[label_id_p]==unq_ids_p,KLIP_tile_label].values[0]
                DATA=Tile(x=xtile,y=ytile,tile_base=tile_base,inst=inst,data=data)
                DATA.mk_tile(fig=fig,ax=axes[row_count][column_count],showplot=False,keep_size=True,simplenorm='sqrt',return_tile=False)

            if no_ids==False:axes[row_count][column_count].text(1,ytext,'%s%i'%(Tag,tag_label_temp),fontsize=fontsize,color=color)            
        
        axes[row_count][column_count].axis('off')
        column_count+=1
        
        if column_count==ncols:
            row_count+=1
            column_count=0
        if row_count == max_row:
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.025, hspace=0.05)
            if save_name!=None: print('Save %s_%s.pdf in %s'%(save_name.split('.pdf')[0],n_row,path2savedir))
            if n_row+1 >= int(binary_df[label_id_p].nunique()/(ncols*max_row)):
                nrows=math.ceil(nrows*abs(int(binary_df[label_id_p].nunique()/(ncols*max_row))-float(binary_df[label_id_p].nunique()/(ncols*max_row))))
            if save_name!=None: 
                fig.savefig('%s_%s.pdf'%(path2savedir+save_name.split('.pdf')[0],n_row),bbox_inches='tight')
                plt.close('all')
            else: plt.show()
            fig,axes=plt.subplots(nrows,ncols,figsize=(ncols*dim,nrows*dim),squeeze=False)
            row_count=0
            column_count=0
            n_row+=1
            
    for elno1 in np.arange(nrows): 
        for elno2 in np.arange(ncols):
            if (elno1 == row_count and elno2 >= column_count) | (elno1 > row_count): 
                try:fig.delaxes(axes[elno1][elno2])
                except: pass
            
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=0.05)
    if save_name!=None:
        if row_count == -1:
            print('Save %s in %s'%(save_name,path2savedir))
            fig.savefig(path2savedir+save_name,bbox_inches='tight')
        else:
            print('Save %s_%s.pdf'%(path2savedir+save_name.split('.pdf')[0],n_row))
            fig.savefig('%s_%s.pdf'%(path2savedir+save_name.split('.pdf')[0],n_row),bbox_inches='tight')
            plt.close('all')
    else: plt.show()

def small_tiles(DF,path2fits, path2tiles, filters, dict={},nrows=10, ncols=10, figsize=None, crossmatch_ids_df=None,ext='_flc', fitsroot = 'fitsroot'):
    if not os.path.exists(path2tiles+'/targets_tiles'):
        os.makedirs(path2tiles+'/targets_tiles')
        getLogger(__name__).info(f'Making {path2tiles}/targets_tiles directory')

    for filter in filters:
        getLogger(__name__).debug(f'Making targets_tiles images for filter {filter}')
        elno = 0
        elno1 = 0
        c=0
        if figsize is None:
            figsize=(ncols,nrows)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        for idx, row in DF.loc[~DF[f'x_{filter}'.lower()].isna()].iterrows():
            if crossmatch_ids_df is None:
                id = int(row.unq_ids)
            else:
                id = crossmatch_ids_df.loc[crossmatch_ids_df.mvs_ids==row.mvs_ids].unq_ids.unique()
            fitsname = row[fitsroot.lower()+f'_{filter}'] + f'{ext}.fits'
            getLogger(__name__).debug(f'Loading {fitsname} for mvs_ids {row.mvs_ids}')
            hdul = fits.open(path2fits+'/'+fitsname)
            SCI = hdul[1].data
            hdul.close()
            x, y = DF.loc[idx, [f'x_{filter}'.lower(), f'y_{filter}'.lower()]].values
            DATA = Tile(data=SCI, x=x, y=y, tile_base=11, delta=0, inst='WFC3', Python_origin=False)
            DATA.mk_tile(pad_data=True, legend=False, showplot=False, verbose=False, kill_plots=True, cbar=True,
                         return_tile=False)
            norm = simple_norm(DATA.data, 'sqrt')

            axes[elno][elno1].imshow(DATA.data, cmap='gray', origin='lower', norm=norm)
            axes[elno][elno1].set_title(f'{id}/{idx}', pad=-4, fontdict={'fontsize': 8})

            if len(dict) > 0:
                if idx in dict[f'bad_{filter}']:
                    rect = patches.Rectangle((-0.25, -0.25), 10.5, 10.5, linewidth=3, edgecolor='r', facecolor='none')
                    axes[elno][elno1].add_patch(rect)
                    DF.loc[idx, f'flag_{filter}'.lower()] = 'rejected'
                elif idx in dict[f'good_{filter}']:
                    DF.loc[idx, f'flag_{filter}'.lower()] = 'good_target'
                    rect = patches.Rectangle((-0.25, -0.25), 10.5, 10.5, linewidth=3, edgecolor='g', facecolor='none')
                    axes[elno][elno1].add_patch(rect)
                elif idx in dict[f'kd_{filter}']:
                    DF.loc[idx, f'flag_{filter}'.lower()] = 'known_double'
                    rect = patches.Rectangle((-0.25, -0.25), 10.5, 10.5, linewidth=3, edgecolor='g', facecolor='none')
                    axes[elno][elno1].add_patch(rect)
                else:
                    DF.loc[idx, f'flag_{filter}'.lower()] = 'good_psf'

            if elno1 >= ncols-1:
                elno1 = 0
                elno += 1
            else:
                elno1 += 1

            if elno >= nrows:
                elno = 0
                elno1 = 0
                [ax.axis('off') for ax in axes.flatten()]
                plt.tight_layout(pad=0.0, w_pad=0.1, h_pad=0.1)
                c+=1
                fig.savefig(path2tiles+ f'/targets_tiles/targets_tiles_{filter}_{c}.png')
                getLogger(__name__).debug('Saved %s'%(path2tiles + f'/targets_tiles/targets_tiles_{filter}_{c}.png'))
                plt.close()
                fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        [ax.axis('off') for ax in axes.flatten()]
        plt.tight_layout(pad=0.0, w_pad=0.1, h_pad=0.1)
        c += 1
        fig.savefig(path2tiles + f'/targets_tiles/targets_tiles_{filter}_{c}.png')
        getLogger(__name__).debug('Saved %s' % (path2tiles + f'/targets_tiles/targets_tiles_{filter}_{c}.png'))
        plt.close()
    # if len(dict) > 0:
    return(DF)