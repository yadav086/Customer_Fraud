import logging
from load_config import load_config
from load_csv import load_csv
from load_features import load_feature
from load_preprocess import load_preprocess
from load_model import load_model
from load_remove_outlier import remove_outlier
from load_ANN_model import create_model
from load_spilt_data import load_spilt_data
from load_metrics import evaluate_fraud_metrics  

# Configure logging format to display timestamp, level, and messages clearly
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Step 1: Starting the main fraud detection execution pipeline...")
        
        # 1. Load data
        df = load_csv()
        logger.info(f"Successfully loaded CSV data. Shape: {df.shape}")
        
        # Optional Outlier Removal Step (Add if needed: df = remove_outlier(df))
        
        # 2. Extract features and targets
        df_num, df_cat, X, y = load_feature(df)
        logger.info(f"Features extracted. Numeric columns: {df_num.shape[1]}, Categorical columns: {df_cat.shape[1]}")
        
        # 3. Create preprocessing pipeline
        preprocess = load_preprocess(df_num, df_cat)
        logger.info("Preprocessing configurations initialized.")
        
        # 4. Stratified splitting of imbalanced data
        X_train_val, X_test_val, y_train_val, y_test_val = load_spilt_data(X, y)
        logger.info(f"Data split executed successfully.")
        logger.info(f"Training Set Size: {X_train_val.shape[0]} | Test Set Size: {X_test_val.shape[0]}")
        logger.info(f"Training Fraud Prevalence: {y_train_val.mean():.4%}")
        
        # 5. Build full Machine Learning pipeline (Preprocess -> SMOTE -> RFECV -> ANN)
        logger.info("Constructing the imblearn Pipeline components...")
        model = create_model(preprocess)
        
        # 6. Fit the model pipeline
        logger.info("Starting model training pipeline (Preprocessing, SMOTE Resampling, Feature Selection, ANN Fitting)...")
        model.fit(X_train_val, y_train_val)
        logger.info("Pipeline training completed successfully!")
        
        #7. Metrics Eval
        metrics = evaluate_fraud_metrics(model, X_test_val, y_test_val)
        return None
        
    except Exception as e:
        logger.exception('A critical error occurred in the main pipeline execution loop.')

if __name__ == '__main__':
    main()
