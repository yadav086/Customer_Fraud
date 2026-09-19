from load_config import load_config
from lightgbm import LGBMClassifier
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

def load_model(preprocess):
    
    config = load_config()
        
    n_estimators= config['LightGBM']['n_estimators']
    learning_rate= config['LightGBM']['learning_rate']
    num_leaves= config['LightGBM']['num_leaves']
    max_depth= config['LightGBM']['max_depth']
    scale_pos_weight= config['LightGBM']['scale_pos_weight']
    subsample= config['LightGBM']['subsample']
    colsample_bytree= config['LightGBM']['colsample_bytree']
    random_state =  config['data']['random_state']
    n_jobs =  config['data']['n_jobs']
    cv  = config['data']['cv']
    min_features_to_select  = config['data']['min_features_to_select']
    n_estimators_rf =config['RandomForestClassifier']['n_estimators']
    criterion =config['RandomForestClassifier']['criterion']
    max_depth =config['RandomForestClassifier']['max_depth']                 # Caps depth to avoid memorizing individual fraud rows
    min_samples_split =config['RandomForestClassifier']['min_samples_split']         # Prevents splitting on tiny clusters of data
    min_samples_leaf =config['RandomForestClassifier']['min_samples_leaf']          # Ensures leaf nodes have generalized rules
    max_features =config['RandomForestClassifier']['max_features']
    class_weight =config['RandomForestClassifier']['class_weight']

    
    
    model_LGBMClassifier = LGBMClassifier(
                                            n_estimators=n_estimators,
                                            learning_rate=learning_rate,
                                            num_leaves=num_leaves,
                                            max_depth=max_depth,
                                            scale_pos_weight=scale_pos_weight,      # Adjust this to match your exact fraud ratio
                                            subsample=subsample,
                                            colsample_bytree=colsample_bytree,
                                            random_state=random_state,
                                            n_jobs=n_jobs
                                        )
    select_RFECV = RFECV(estimator=model_LGBMClassifier, cv =cv, 
                         min_features_to_select=min_features_to_select,n_jobs=n_jobs, 
                         scoring='f1', verbose=True)

    rf_model = RandomForestClassifier(
                                        n_estimators=n_estimators_rf,
                                        criterion=criterion,
                                        max_depth=max_depth,                 # Caps depth to avoid memorizing individual fraud rows
                                        min_samples_split=min_samples_split,         # Prevents splitting on tiny clusters of data
                                        min_samples_leaf=min_samples_leaf,           # Ensures leaf nodes have generalized rules
                                        max_features=max_features,
                                        class_weight=class_weight,      # Crucial: Weights the fraud class higher
                                        random_state=random_state,
                                        n_jobs=n_jobs           # Uses all available CPU cores for speed
                                    )

    model = Pipeline(steps = [('preprocess',preprocess),
                          ('smote',SMOTE()),
                          ('select_RFECV',select_RFECV),
                          ('rf',rf_model)])

    return model
