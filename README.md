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