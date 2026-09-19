import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Sequential
from scikeras.wrappers import KerasClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFECV
from load_config import load_config
from lightgbm import LGBMClassifier

# --- 1. ONLY define the Neural Network structure here ---
def create_nn_architecture(meta):
    model_seq = Sequential([
        tf.keras.layers.Input(shape=(meta["n_features_in_"],)),
        Dense(32, activation='relu', kernel_initializer='he_normal'),
        Dropout(0.30),
        Dense(16, activation='relu', kernel_initializer='he_normal'),
        Dropout(0.20),
        Dense(1, activation='sigmoid')
    ])
    
    model_seq.compile(
        optimizer='adam',  
        loss='binary_crossentropy', 
        metrics=[
            tf.keras.metrics.AUC(name='auc'),                
            tf.keras.metrics.Precision(name='precision'),    
            tf.keras.metrics.Recall(name='recall')          
        ]
    )
    return model_seq

# --- 2. Build the full pipeline workflow here ---
def create_model(preprocess):
    # Load config parameters
    config = load_config()
            
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
    
    # Configure LightGBM for RFECV feature selection
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
    
    # Define Keras Checkpoint callback
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='best_fraud_model.keras',   
        monitor='val_recall',                   
        mode='max',                          
        save_best_only=True,                 
        verbose=1                            
    )
    
    # Wrap the Neural Network Architecture
    nn_model = KerasClassifier(
        model=create_nn_architecture, # Points ONLY to the architecture function
        epochs=50, 
        batch_size=16,
        shuffle=True,
        validation_split=0.2,         # Required so 'val_recall' actually exists
        callbacks=[checkpoint_callback]
    )

    # Assemble the final imblearn pipeline
    model = Pipeline(steps=[
        ('preprocess', preprocess),
        ('smote', SMOTE(random_state=random_state)),
        ('select_RFECV', select_RFECV),
        ('nn', nn_model)
    ])
    
    return model
