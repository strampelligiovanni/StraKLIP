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
from scipy import stats

def load_data(DF, id, filter, numbasis):
    filename = DF.path2out + f'/mvs_tiles/{filter}/tile_ID{id}.fits'
    hdulist = pyfits.open(filename)
    # data = hdulist['SCI'].data
    # data[data < 0] = 0
    # centers = [int((data.shape[1] - 1) / 2), int((data.shape[0] - 1) / 2)]
    #
    # # now let's generate a dataset to reduce for KLIP. This contains data at both roll angles
    # dataset = GenericData([data], [centers], filenames=[filename])
    # PSF = get_MODEL_from_data(hdulist[f'MODEL{KL}'].data, centers, d)
    residuals=np.array([hdulist[f'KMODE{nb}'].data for nb in numbasis])

    return(residuals)

if __name__ == "__main__":
    id = 1
    d = 7
    filter = 'f850lp'
    # filter = 'f814w'
    KL=3


    pipe_cfg = '/Users/gstrampelli/PycharmProjects/Giovanni/work/pipeline_logs/FFP_drc/pipe.yaml'
    data_cfg = '/Users/gstrampelli/PycharmProjects/Giovanni/work/pipeline_logs/FFP_drc/data.yaml'
    pipe_cfg = config.configure_pipeline(pipe_cfg, pipe_cfg=pipe_cfg, data_cfg=data_cfg,
                                         dt_string=datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    data_cfg = config.configure_data(data_cfg, pipe_cfg)
    numbasis = np.array(pipe_cfg.psfsubtraction['kmodes'])  # KL basis cutoffs you want to try

    dataset = input_tables.Tables(data_cfg, pipe_cfg)
    DF = config.configure_dataframe(dataset, load=True)

    fig, ax = plt.subplots()
    for id in DF.mvs_targets_df.mvs_ids.unique():
        shapiro_test_list=[]
        residuals = load_data(DF, id, filter, numbasis)

        for residual in residuals:
            shapiro_test = stats.shapiro(residual)
            shapiro_test_list.append(shapiro_test.statistic)
        ax.plot(DF.kmodes, shapiro_test_list, lw=1)
    ax.axvline(6)
    ax.minorticks_on()
    ax.set_ylim(0, 1)
    plt.show()

