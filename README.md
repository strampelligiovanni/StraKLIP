# StraKLIP

We present a new pipeline developed to detect and characterize faint astronomical companions at small angular 
separation from the host star using sets of wide-field imaging observations not specifically designed for 
High Contrast Imaging analysis. The core of the pipeline relies on Karhunen-Lo`eve trun- cated transformation 
of the reference PSF library to perform PSF subtraction and identify candidates. Tests of reliability of detections 
and characterization of companions are made through simulation of binaries and generation of Receiver Operating 
Characteristic curves for false positive/true positive analysis. The algorithm has been successfully tested on 
large HST /ACS and WFC3 datasets acquired for two HST Treasury Programs on the Orion Nebula Cluster. 
Based on these extensive numerical experiments we find that, despite being based on methods designed for 
observations of single star at a time, our pipeline performs very well on mosaic space based data. 
In fact, we are able to detect brown dwarf-mass companions almost down to the planetary mass limit. 
The pipeline is able to re- liably detect signals at separations as close as ≳ 0.1′′ with a completeness 
'of ≳ 10%, or ∼ 0.2′′ with a completeness of ∼ 30%. This approach can potentially be applied to a wide variety 
of space based imaging surveys, starting with data in the existing HST archive, near-future JWST mosaics, 
and future wide-field Roman images.

Source: Strampelli, G.~M., Pueyo, L., Aguilar, J., et al.\ 2022, \aj, 164, 147. doi:10.3847/1538-3881/ac879e

NOTE: This paper, while describing the same steps this last version of the StraKLIP pipeline goes though, 
still refer to the an older version of the pipeline. While still valid in concept, the actual structure of the pipeline 
might be slightly different from what described there.

# Pipeline Quick Start Guide

## Pipeline Setup
Create a pipeline environment using the `straklipenv.yaml` in `src/StraKLIP`.

Move to the directory where you want to work, and create the following yaml files:
1. `data.yaml` - This is the dataset configuration file.
2. `pipe.yaml` - This is the pipeline global configuration file.

You'll need to redefine both to your actual needs, in particular the data.yaml
Examples of both files are provided in the `src/StraKLIP/straklip/template` directory.

The `data.yaml` file host three major sections:
```
target: this section host the information about the target that can be applied to all the soruces in the catalog as whole
mvs_table: this section host the convection use by the provided imput catalog for each column
unq_table: this section host the convection use by the provided imput catalog for each column
```

The `pipe.yaml` file host 13 major sections:
```
instrument: here we record information about the instument used to acquire the data (for example the pixelscale, the name of the PAM, the array dimension etc.)
ncpu: number of CPU to be utilized running the pipelinme
redo: wheter overwrit output of existing steps
debug: wheter promt more verbose modes and plots
flow: enable which step the pipeline will go thorugh
    - buildhdf: build the dataframe where to store the output of the pipeline
    - mktiles:  create small tiles, one for each source in the catalog (both for the multivisit and the average)
    - mkphotometry: perform aperture photometry on each tiles
    - fow2cells:  break the FOW dividing the source in groups of close spatially related targets in order to minimize distorsion when building the PSF library and performin PSF subtraction
    - psfsubtraction: perform PSF subtraction on each tile, and create a residual tile.
    - klipphotometry: perform photometry on each residual tile and check the presence of a candidate companion
    - analysis: extract companions, and asses contrast curves (raw and calibrated through throuput) and mass sensitivity curves 
paths: path mandatory for the pipeline
  pyklip: path to pyklip
  pam: path to Pixel Area Map for the instument
  data: path to the fits file. I in the data folder are genrally stored all the heavy files like the fits files
  database: path to catalogs and supplementary material
  out: path to the output directory for the pipeline
buildhdf: here we record all the specific option for this step of the pipeline. In particular, under default_mvs_table and default_unq_table, we record the default name the pipeline will look for twith the matched name provided by the input catalogs for that specifc column
mktiles: here we record all the specific option for this step of the pipeline
mkphotometry: here we record all the specific option for this step of the pipeline
fow2cells: here we record all the specific option for this step of the pipeline
psfsubtraction: here we record all the specific option for this step of the pipeline
klipphotometry: here we record all the specific option for this step of the pipeline
analysis: here we record all the specific option for this step of the pipeline
```

Before running the pipeline, you will need a series of HST `_flc` images, and a catalog recording the `x`, `y`
coordinates and a few additional information of each sources on these images. Having their photometry might help, 
but it's not mandatory. The pipeline will perform it's own aperture photometry.

The mandatory columns for the `mvs_table` catalog are the following:
1. unq_ids: ids for average catalog 
2. mvs_ids: ids for multivisit catalog 
3. vis: visit column name in catalog 
4. ext: extension column name in catalog that identify SCI in fits file (for HST if CCDCHIP = 1, EXT = 4, CCDCHIP = 2, EXT = 1)
5. x: filter wise x column name in catalog 
6. y: filter wise y column name in catalog
7. fitsroot:  filter wise fitsroot column name in catalog (it's the filename without the `_flc` extension)
8. exptime: filter wise exposure time for each source, i.e. header['EXPTIME']
9. pav3: filter wise the HST V3 position angle, i.e. header['PA_V3']
10. rota: filter wise HST orientation, i.e. header['ORIENTAT']

You can use the section `mvs_table` in the `data.yaml` to tell the pipeline how to match your columns name with the 
pipeline default (stored in the `pipe.yaml`).

An `average` catalog recording the `ra`, `dec` and `type` of each `unique` source is also need. 
Having their photometry might help, but it's not mandatory. The pipeline will evaluate this as well.

The mandatory columns for the `unq_table` catalog are the following:
1. unq_ids: ids for average catalog 
2. ra: ra column name in catalog 
3. dec: dec column name in catalog 
4. type: filter wise type column name in catalog (see below)
5. mag: filter wise mag column name in catalog
6. emag: filter wise error mag column name in catalog

```
type    Explanation
0       a target that must be rejected by the pipeline (bad detection/photometry).
1       a good target for the pipeline
2       unresolved double
3       known double
n       user defined flag
```

NOTE: the pipeline will select only type 1 sources to build the base of it's PSF subtraction library. Source of type
1, 2 or n (with n > 3) will be processed by the pipeline instead.

An example of out to build these catalog is presented in the `join_catalogues.ipynb` notebook 
in `/Users/gstrampelli/PycharmProjects/Giovanni/src/StraKLIP/straklip/template`.

## Running the Pipeline

To run the pipeline, run the `skpipe.py` script in `src/StraKLIP/script`. use the `-p` option to point to the
`pipe.yaml` and `-d` to point to the `data.yaml` if not running the pipeline from the directory where they are stored.

The `flow` section of the `pipe.yaml` lists all the steps that will be executed when running the pipeline. 
Here you may comment out or delete all steps you do not wish to run. To fully function, the pipeline should run through 
each of the following at least once.

```
flow: 
########## first part of the pipeline ##########
- buildhdf
- mktiles
- mkphotometry
- fow2cells
- psfsubtraction
- klipphotometry
########## second part of the pipeline ##########
- buildfphdf
- mkcompleteness
- fpanalysis
```

  
To generate all necessary directories as specified in the `paths` section of the `pipe.yaml`, run the pipeline with 
the `--make-dir` option enable.

NOTE: The default values for these `paths` will need to be changed in the `pipe.yaml` to point to the appropriate 
location for your computer. 
