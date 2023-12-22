from straklip import config,input_tables
from datetime import datetime
from dataframe import DataFrame
from mktiles import make_mvs_tiles
from astropy.io import fits
from astropy.visualization import simple_norm
from glob import glob
import matplotlib.pyplot as plt

pipe_cfg='/Users/gstrampelli/PycharmProjects/Giovanni/work/pipeline_logs/FFP/pipe.yaml'
data_cfg='/Users/gstrampelli/PycharmProjects/Giovanni/work/pipeline_logs/FFP/data.yaml'
pipe_cfg = config.configure_pipeline(pipe_cfg,pipe_cfg=pipe_cfg,data_cfg=data_cfg,dt_string=datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
data_cfg = config.configure_data(data_cfg,pipe_cfg)

dataset = input_tables.Tables(data_cfg, pipe_cfg)
DF = DataFrame(path2out=pipe_cfg.paths['out'])
DF.load_dataframe()

filter='f814w'
make_mvs_tiles(DF,filter,dataset.pipe_cfg,
                avg_ids_test_list=[49],
                redo=True,
                xy_m=dataset.pipe_cfg.mktiles['xy_m'],
                workers=int(dataset.pipe_cfg.ncpu),
                cr_remove=dataset.pipe_cfg.mktiles['cr_remove'],
                la_cr_remove=dataset.pipe_cfg.mktiles['la_cr_remove'],
                parallel_runs=False,
                Python_origin=dataset.pipe_cfg.mktiles['python_origin'],
                look4duplicants=dataset.pipe_cfg.mktiles['look4duplicants'],
                multiply_by_exptime=dataset.pipe_cfg.mktiles['multiply_by_exptime'],
                multiply_by_PAM=dataset.pipe_cfg.mktiles['multiply_by_PAM'],
                multiply_by_gain=dataset.pipe_cfg.mktiles['multiply_by_gain'],
                cr_radius=dataset.pipe_cfg.mktiles['cr_radius'] / dataset.pipe_cfg.instrument['pixelscale'],
                xy_dmax=dataset.pipe_cfg.mktiles['xy_dmax'],
                verbose=True)

# fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(5, 2),squeeze=False)
# fitsnames = glob(f'/Users/gstrampelli/PycharmProjects/Giovanni/work/out/FFP/mvs_tiles/{filter}/*.fits')
# elno = 0
# elno1 = 0
#
# for fitsname in fitsnames:
#     idx = int(fitsname.split('ID')[-1].split('.')[0])
#     id = DF.crossmatch_ids_df.loc[DF.crossmatch_ids_df.mvs_ids == idx].avg_ids.unique()
#     hdul=fits.open(fitsname)
#     SCI = hdul[1].data
#     hdul.close()
#
#     norm = simple_norm(SCI, 'sqrt')
#     axes[elno][elno1].imshow(SCI, cmap='gray', origin='lower', norm=norm)
#     axes[elno][elno1].plot((SCI.shape[1]-1)/2,(SCI.shape[0]-1)/2, 'or',ms=1)
#     axes[elno][elno1].set_title(f'{id}/{idx}', pad=-4, fontdict={'fontsize': 8})
#     if elno1 >= 9:
#         elno1 = 0
#         elno += 1
#     else:
#         elno1 += 1
#
#     [ax.axis('off') for ax in axes.flatten()]
# plt.tight_layout(pad=0.0, w_pad=0.1, h_pad=0.1)
# plt.show()
# print()