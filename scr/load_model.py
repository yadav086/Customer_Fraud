import logging
from load_config import load_config
from lightgbm import LGBMClassifier
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# Initialize the module-level logger
logger = logging.getLogger(__name__)

def load_model(preprocess):
    """
    Loads runtime configurations, builds the feature selection wrappers,
    and returns a robust combined imblearn pipeline (Preprocess -> SMOTE -> RFECV -> Random Forest).
    """
    try:
        logger.info("Step 4: Loading configurations and constructing model pipeline components...")
        config = load_config()
        
        # 1. Parse LightGBM configurations for feature selection
        logger.debug("Parsing LightGBM and global environment parameters from configuration...")
        n_estimators = config['LightGBM']['n_estimators']
        learning_rate = config['LightGBM']['learning_rate']
        num_leaves = config['LightGBM']['num_leaves']
        max_depth_lgb = config['LightGBM']['max_depth']
        scale_pos_weight = config['LightGBM']['scale_pos_weight']
        subsample = config['LightGBM']['subsample']
        colsample_bytree = config['LightGBM']['colsample_bytree']
        
        # Global Data configurations
        random_state = config['data']['random_state']
        n_jobs = config['data']['n_jobs']
        cv = config['data']['cv']
        min_features_to_select = config['data']['min_features_to_select']
        
        # 2. Parse Random Forest configurations for the final estimator
        logger.debug("Parsing RandomForestClassifier parameters from configuration...")
        n_estimators_rf = config['RandomForestClassifier']['n_estimators']
        criterion = config['RandomForestClassifier']['criterion']
        max_depth_rf = config['RandomForestClassifier']['max_depth']                 
        min_samples_split = config['RandomForestClassifier']['min_samples_split']         
        min_samples_leaf = config['RandomForestClassifier']['min_samples_leaf']          
        max_features = config['RandomForestClassifier']['max_features']
        class_weight = config['RandomForestClassifier']['class_weight']

        # 3. Instantiate LightGBM and the RFECV Feature Selector
        logger.info("Initializing LightGBM and building RFECV feature selection framework...")
        model_LGBMClassifier = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth_lgb,
            scale_pos_weight=scale_pos_weight,      
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=n_jobs
        )
        
        select_RFECV = RFECV(
            estimator=model_LGBMClassifier, 
            cv=cv, 
            min_features_to_select=min_features_to_select,
            n_jobs=n_jobs, 
            scoring='f1', 
            verbose=True
        )

        # 4. Instantiate the final Random Forest Classifier
        logger.info("Initializing the final RandomForestClassifier estimator...")
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators_rf,
            criterion=criterion,
            max_depth=max_depth_rf,                 
            min_samples_split=min_samples_split,         
            min_samples_leaf=min_samples_leaf,           
            max_features=max_features,
            class_weight=class_weight,      
            random_state=random_state,
            n_jobs=n_jobs           
        )

        # 5. Assemble the final imblearn pipeline
        logger.info("Assembling the composite imblearn execution pipeline...")
        model = Pipeline(steps=[
            ('preprocess', preprocess),
            ('smote', SMOTE(random_state=random_state)),
            ('select_RFECV', select_RFECV),
            ('rf', rf_model)
        ])

        logger.info("Model pipeline setup successfully assembled and ready for fitting.")
        return model

    except KeyError as ke:
        logger.error(f"Configuration key parsing failure! Verify your config.yaml layout structure: {ke}")
        raise ke
    except Exception as e:
        logger.exception("A critical exception occurred while setting up the model pipeline layout.")
        raise e
