from sklearn.model_selection import train_test_split
from load_config import load_config
import logging
logger = logging.getLogger(__name__)
def load_spilt_data(X,y):
    try : 

        config = load_config()
        random_state =  config['data']['random_state']
        test_size =  config['data']['test_size']
        
        
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size, random_state=random_state, stratify=y)

        return X_train,X_test,y_train,y_test
    except Exception as e:
         logger.exception('error in load_spilt_data')
         raise