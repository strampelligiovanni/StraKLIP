from straklip import config,input_tables
from datetime import datetime
from dataframe import DataFrame
from mkphotometry import get_ee_df, make_mvs_photometry

pipe_cfg='/Users/gstrampelli/PycharmProjects/Giovanni/work/pipeline_logs/FFP/pipe.yaml'
data_cfg='/Users/gstrampelli/PycharmProjects/Giovanni/work/pipeline_logs/FFP/data.yaml'
pipe_cfg = config.configure_pipeline(pipe_cfg,pipe_cfg=pipe_cfg,data_cfg=data_cfg,dt_string=datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
data_cfg = config.configure_data(data_cfg,pipe_cfg)

dataset = input_tables.Tables(data_cfg, pipe_cfg)
DF = DataFrame(path2out=pipe_cfg.paths['out'])
DF.load_dataframe()

zpt = DF.zpt
ee_dict=get_ee_df(dataset)
for filter in ['f814w']:
    make_mvs_photometry(DF, filter,
                        mvs_ids_test_list=[199],
                        ee_dict=ee_dict,
                        workers=1,
                        parallel_runs=False,
                        la_cr_remove=dataset.pipe_cfg.mktiles['la_cr_remove'],
                        cr_radius=dataset.pipe_cfg.mktiles['cr_radius'],
                        multiply_by_exptime=dataset.pipe_cfg.mktiles['multiply_by_exptime'],
                        multiply_by_gain=dataset.pipe_cfg.mktiles['multiply_by_gain'],
                        multiply_by_PAM=dataset.pipe_cfg.mktiles['multiply_by_PAM'],
                        zpt=zpt,
                        radius_in=dataset.pipe_cfg.mkphotometry['radius_in']/DF.pixscale,
                        radius1_in=dataset.pipe_cfg.mkphotometry['radius1_in']/DF.pixscale,
                        radius2_in=dataset.pipe_cfg.mkphotometry['radius2_in']/DF.pixscale,
                        kill_plots=not dataset.pipe_cfg.mkphotometry['debug'],
                        grow_curves=dataset.pipe_cfg.mkphotometry['grow_curves'],
                        p=dataset.pipe_cfg.mkphotometry['p'],
                        gstep=dataset.pipe_cfg.mkphotometry['gstep'],
                        bpx_list=dataset.pipe_cfg.mkphotometry['bad_pixel_flags'],
                        spx_list=dataset.pipe_cfg.mkphotometry['sat_pixel_flags'],
                        skip_flags=dataset.pipe_cfg.mkphotometry['skip_flags'],
                        path2savefile=dataset.pipe_cfg.paths['database']+f'/targets_photometry_tiles/{filter}')
