==========================
Pipeline Quick Start Guide
==========================

---------------
Pipeline Setup
---------------
Create a pipeline environment using the instructions in `here <https://straklip.readthedocs.io/latest/installation.html>`_.

Move to the directory where you want to work, and create the following yaml files:

    - `data.yaml` - This is the dataset configuration file.
    - `pipe.yaml` - This is the pipeline global configuration file.

You'll need to redefine both to your actual needs, in particular the data.yaml
Examples of both files are provided in the `src/StraKLIP/straklip/template` directory.

The `data.yaml` file host three major sections:

    - target: this section host the information about the target that can be applied to all the sources in the catalog as whole
    - mvs_table: this section host the convection use by the provided input catalog for each column
    - unq_table: this section host the convection use by the provided input catalog for each column

The `pipe.yaml` file host 13 major sections:

    - instrument: here we record information about the instument used to acquire the data (for example the pixelscale, the name of the PAM, the array dimension etc.)
    - ncpu: number of CPU to be utilized running the pipelinme
    - redo: wheter overwrit output of existing steps
    - debug: wheter promt more verbose modes and plots
    - flow: enable which step the pipeline will go thorugh
        - buildhdf: build the dataframe where to store the output of the pipeline
        - mktiles:  create small tiles, one for each source in the catalog (both for the multivisit and the average)
        - mkphotometry: perform aperture photometry on each tiles
        - fow2cells:  break the FOW dividing the source in groups of close spatially related targets in order to minimize distorsion when building the PSF library and performin PSF subtraction
        - psfsubtraction: perform PSF subtraction on each tile, and create a residual tile.
        - klipphotometry: perform photometry on each residual tile and check the presence of a candidate companion
        - analysis: extract companions, and asses contrast curves (raw and calibrated through throuput) and mass sensitivity curves
    - paths: path mandatory for the pipeline
        - pyklip: path to pyklip
        - pam: path to Pixel Area Map for the instument
        - data: path to the fits file. I in the data folder are genrally stored all the heavy files like the fits files
        - database: path to catalogs and supplementary material
        - out: path to the output directory for the pipeline
    - buildhdf: here we record all the specific option for this step of the pipeline. In particular, under default_mvs_table and default_unq_table, we record the default name the pipeline will look for twith the matched name provided by the input catalogs for that specifc column
    - mktiles: here we record all the specific option for this step of the pipeline
    - mkphotometry: here we record all the specific option for this step of the pipeline
    - fow2cells: here we record all the specific option for this step of the pipeline
    - psfsubtraction: here we record all the specific option for this step of the pipeline
    - klipphotometry: here we record all the specific option for this step of the pipeline
    - analysis: here we record all the specific option for this step of the pipeline


Before running the pipeline, you will need a series of HST `_flc` or `_flt_` images, and a catalog recording the `x`, `y`
coordinates and a few additional information of each sources on these images. Having their photometry might help,
but it's not mandatory. If needed, the pipeline will perform its own aperture photometry.

The mandatory columns for the `mvs_table` catalog are the following:

    - unq_ids: ids for average catalog
    - mvs_ids: ids for multivisit catalog
    - vis: visit column name in catalog
    - ext: extension column name in catalog that identify SCI in fits file (for HST if CCDCHIP = 1, EXT = 4, CCDCHIP = 2, EXT = 1)
    - x: filter wise x column name in catalog
    - y: filter wise y column name in catalog
    - fitsroot:  filter wise fitsroot column name in catalog (it's the filename without the `_flc` extension)
    - exptime: filter wise exposure time for each source, i.e. header['EXPTIME']
    - pav3: filter wise the HST V3 position angle, i.e. header['PA_V3']
    - rota: filter wise HST orientation, i.e. header['ORIENTAT']

You can use the section `mvs_table` in the `data.yaml` to tell the pipeline how to match your columns name with the
pipeline default (stored in the `pipe.yaml`).

An `unique` catalog recording the `ra`, `dec` and `type` of each `unique` source is also need.
Having their photometry might help, but it's not mandatory. The pipeline will evaluate this as well.

The mandatory columns for the `unq_table` catalog are the following:

    - unq_ids: ids for average catalog
    - ra: ra column name in catalog
    - dec: dec column name in catalog
    - type: filter wise type column name in catalog (see below)
    - mag: filter wise mag column name in catalog
    - emag: filter wise error mag column name in catalog

+------+--------------------------------------------------------------------------------+
|type  |  Explanation                                                                   |
+======+================================================================================+
|0     |  a target that must be rejected by the pipeline (bad detection/photometry).    |
+------+--------------------------------------------------------------------------------+
|1     |  a good target for the pipeline                                                |
+------+--------------------------------------------------------------------------------+
|2     |  unresolved double                                                             |
+------+--------------------------------------------------------------------------------+
|3     |  known double                                                                  |
+------+--------------------------------------------------------------------------------+
|n     |  user defined flag                                                             |
+------+--------------------------------------------------------------------------------+


NOTE: the pipeline will select only type 1 sources to build the base of it's PSF subtraction library. Source of type
1, 2 or n (with n > 3) will be processed by the pipeline instead.

An example of out to build these catalog is presented `here <https://straklip.readthedocs.io/latest/tutorials/tutorial_join_catalogues_DRC.html>`_.

--------------------
Running the Pipeline
--------------------
The pipeline can be assemble to fit the specific user needs, combining its different building block. An example of how
to run the pipeline is presented in the `tutorials <https://straklip.readthedocs.io/latest/tutorials.html>`_
section.

A 'default' script is provided by the `skpipe.py` routine in `./script` directory. Use the `-p` option to point to the
`pipe.yaml` and `-d` to point to the `data.yaml` if not running the pipeline from the directory where they are stored.

The `flow` section of the `pipe.yaml` lists all the steps that will be executed when running the pipeline.
Here you may comment out or delete all steps you do not wish to run. To fully function, the pipeline should run through
each of the following at least once.

flow:
    - buildhdf
    - mktiles
    - mkphotometry
    - fow2cells
    - psfsubtraction
    - klipphotometry
    - analysis

To generate all necessary directories as specified in the `paths` section of the `pipe.yaml`, run the pipeline with
the `--make-dir` option enable.

NOTE: The default values for these `paths` will need to be changed in the `pipe.yaml` to point to the appropriate
location for your computer.