from sklearn.ensemble import IsolationForest
import logging
logger = logging.getLogger(__name__)

def remove_outlier(preprocess, X_train,y_train):
    try:
        X_train_preprocess = preprocess.fit_transform(X_train)
        iso = IsolationForest(contamination=0.01,random_state=42, n_estimators=1000, n_jobs=-1)
        label = iso.fit_predict(X_train_preprocess)

        mask= label ==1

        X_train_clean = X_train[mask]
        y_train_clean = y_train[mask]
        
        return X_train_clean, y_train_clean
    except Exception as e:
        logger.exception('error in remove_outlier')
        raise
        