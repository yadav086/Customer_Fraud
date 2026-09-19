import pandas as pd
import logging
from load_config import load_config

logger = logging.getLogger(__name__)


def load_csv():
    try:
        config = load_config()
        path = config['data']['file_path']
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logger.exception('Error in load csv')
        raise