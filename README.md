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

# Pipeline quick start guide

Create a pipeline environment using the `straklipenv.yaml` in `src/StraKLIP`.

Move to the directory where you want to work, and create the following yaml files:
1. `pipe.yaml` - This is the pipeline global configuration file.
2. `data.yaml` - This is the dataset configuration file. 

You'll need to redefine both to your actual needs, in particular the data.yaml
Examples of both files are provided in the `src/StraKLIP/straklip/template` directory.

To run the pipeline, run the `skpipe.py` script in `src/StraKLIP/script`. use the `-p` option to point to the
`pipe.yaml` and `-d` to point to the `data.yaml` if not running the pipeline from the directory where they are stored.

The `flow` section of the `pipe.yaml` lists all the steps that will be executed when running the pipeline. 
Here you may comment out or delete all steps you do not wish to run. To fully function, the pipeline should run through 
each of the following at least once.

```
flow: 
- buildhdf
- mktiles
- mkphotometry
- fow2cells
- psfsubtraction
- klipphotometry
- fpanalysis
```

  
To generate all necessary directories as specified in the `paths` section of the `pipe.yaml`, run the pipeline with 
the `--make-dir` option enable.

NOTE: The default values for these `paths` will need to be changed in the `pipe.yaml` to point to the appropriate 
location for your computer. 
