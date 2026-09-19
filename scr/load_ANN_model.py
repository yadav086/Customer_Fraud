import logging
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFECV
from load_config import load_config
from lightgbm import LGBMClassifier

# Initialize module-level logger
logger = logging.getLogger(__name__)

# --- 1. ONLY define the Neural Network structure here ---
def create_nn_architecture(meta):
    """
    Builds and compiles the TensorFlow/Keras deep learning graph topology.
    """
    try:
        logger.debug(f"Initializing Keras Sequential model structure. Input feature width: {meta['n_features_in_']}")
        
        model_seq = Sequential([
            tf.keras.layers.Input(shape=(meta["n_features_in_"],)),
            Dense(32, activation='relu', kernel_initializer='he_normal'),
            Dropout(0.30),
            Dense(16, activation='relu', kernel_initializer='he_normal'),
            Dropout(0.20),
            Dense(1, activation='sigmoid')
        ])
        
        logger.debug("Compiling model graph with fraud-focused evaluation metrics...")
        model_seq.compile(
            optimizer='adam',  
            loss='binary_crossentropy', 
            metrics=[
                tf.keras.metrics.AUC(name='auc'),                
                tf.keras.metrics.Precision(name='precision'),    
                tf.keras.metrics.Recall(name='recall')          
            ]
        )
        logger.info("TensorFlow/Keras model structure compiled successfully.")
        return model_seq

    except Exception as e:
        logger.exception("Failed to build or compile the neural network graph topology.")
        raise e

# --- 2. Build the full pipeline workflow here ---
def create_model(preprocess):
    """
    Parses configuration files, sets up automated feature elimination, 
    and wraps the neural network model into an out-of-sample safe imblearn pipeline.
    """
    try:
        logger.info("Step 4 (ANN): Initializing neural network execution pipeline workspace...")
        config = load_config()
                
        # 1. Parse operational parameters from configuration
        logger.debug("Parsing LightGBM and runtime environment profiles from config.yaml...")
        n_estimators = config['LightGBM']['n_estimators']
        learning_rate = config['LightGBM']['learning_rate']
        num_leaves = config['LightGBM']['num_leaves']
        max_depth = config['LightGBM']['max_depth']
        scale_pos_weight = config['LightGBM']['scale_pos_weight']
        subsample = config['LightGBM']['subsample']
        colsample_bytree = config['LightGBM']['colsample_bytree']
        
        random_state = config['data']['random_state']
        n_jobs = config['data']['n_jobs']
        cv = config['data']['cv']
        min_features_to_select = config['data']['min_features_to_select']
        
        # 2. Instantiate cross-validated feature elimination mechanics
        logger.info("Building LightGBM background ranker and RFECV core components...")
        model_LGBMClassifier = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
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
        
        # 3. Setup structural training callbacks for the neural network
        logger.debug("Configuring Keras training save tracking callbacks...")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath='best_fraud_model.keras',   
            monitor='val_recall',                   
            mode='max',                          
            save_best_only=True,                 
            verbose=1                            
        )
        
        # 4. Wrap the model architecture via SciKeras
        logger.info("Wrapping neural network architecture into SciKeras estimator interface...")
        nn_model = KerasClassifier(
            model=create_nn_architecture, 
            epochs=50, 
            batch_size=16,
            shuffle=True,
            validation_split=0.2,         
            callbacks=[checkpoint_callback]
        )

        # 5. Assemble the standalone imblearn workflow
        logger.info("Assembling pipeline layers (Preprocess -> SMOTE -> RFECV -> NN)...")
        model = Pipeline(steps=[
            ('preprocess', preprocess),
            ('smote', SMOTE(random_state=random_state)),
            ('select_RFECV', select_RFECV),
            ('nn', nn_model)
        ])
        
        logger.info("Composite deep learning pipeline successfully constructed and verified.")
        return model

    except KeyError as ke:
        logger.error(f"Configuration profile parsing failure! Key element missing inside config.yaml: {ke}")
        raise ke
    except Exception as e:
        logger.exception("A critical exception halted the deep learning pipeline layout configuration.")
        raise e
