import logging, yaml, os
import logging.config
import logging.handlers
from multiprocessing_logging import install_mp_handler
import multiprocessing, logging

LEVELS = [logging.DEBUG, logging.INFO, logging.WARNING,
          logging.ERROR, logging.CRITICAL]
def getLogger(args,**kwargs):
    setup = kwargs.pop('setup', False)
    logfile = kwargs.pop('logfile', None)
    configfile = kwargs.pop('configfile', None)
    if setup:
        setupLogger(configfile,logfile)
    log = logging.getLogger(args)
    if not setup:
        log.addHandler(logging.NullHandler())
    return log

def setupLogger(configfile,logfile):
    with open(configfile, 'r') as f:
        config = yaml.safe_load(f.read())
    if logfile:
        config['handlers']['file']['filename'] = logfile
    logging.config.dictConfig(config)
    try:
        coloredlogs.install()
    except NameError:
        pass
    logging.captureWarnings(True)
    install_mp_handler()


class MakeFileHandler(logging.FileHandler):
    def __init__(self, filename, mode='a', encoding=None, delay=0):
        mkdir_p(os.path.dirname(filename))
        logging.FileHandler.__init__(self, filename, mode, encoding, delay)

def mkdir_p(path):
    """http://stackoverflow.com/a/600612/190597 (tzot)"""
    if not path:
        return
    try:
        os.makedirs(path, exist_ok=True)  # Python>3.2
    except TypeError:
        try:
            os.makedirs(path)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                raise
