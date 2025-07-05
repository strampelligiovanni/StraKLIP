import numpy as np
from straklip.stralog import getLogger
from straklip.utils.utils_fpanalysis import update_candidates_photometry

def run_false_positive_analysis(DF, avg_ids_list=[], label='crclean_data', aptype='square', verbose=False,
                                sigma=2.5, DF_fk=None, delta=5, skip_filters=[], sat_thr=np.inf,
                                suffix='',showplot=True,oversubtraction=False):

    if oversubtraction and redo:
        getLogger(__name__).info(f'Correcting companion photometry for over subtraction.')

        DF=update_candidates_photometry(DF,
                                        avg_ids_list=avg_ids_list,
                                        label=label,
                                        aptype=aptype,
                                        verbose=verbose,
                                        sigma=sigma,
                                        DF_fk=DF_fk,
                                        delta=delta,
                                        skip_filters=skip_filters,
                                        sat_thr=sat_thr,
                                        suffix=suffix)

    # DF=false_positive_analysis(avg_ids_list=[], showplot=showplot, verbose=verbose)

    return(DF)

def run(packet):
    DF = packet['DF']
    dataset = packet['dataset']

    DF=run_false_positive_analysis(DF,
                                    avg_ids_list=dataset.pipe_cfg.fpanalysis['avg_ids_list'],
                                    label=dataset.pipe_cfg.klipphotometry['label'],
                                    aptype=dataset.pipe_cfg.klipphotometry['aptype'],
                                    verbose=dataset.pipe_cfg.fpanalysis['verbose'],
                                    sigma=dataset.pipe_cfg.fpanalysis['sigma'],
                                    delta=dataset.pipe_cfg.klipphotometry['delta'],
                                    skip_filters=dataset.pipe_cfg.klipphotometry['skip_filters'],
                                    sat_thr=dataset.pipe_cfg.klipphotometry['sat_thr'],
                                    suffix=dataset.pipe_cfg.klipphotometry['suffix'],
                                    showplot=dataset.pipe_cfg.fpanalysis['showplot'],
                                    oversubtraction=dataset.pipe_cfg.fpanalysis['oversubtraction']
                                   )

    # DF.save_dataframes(__name__)