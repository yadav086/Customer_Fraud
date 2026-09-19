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

# Configure logging format to display clean timestamps and tracking levels
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Initializing fraud detection pipeline execution...")
        
        # 1. Load data
        df = load_csv()
        logger.info(f"Step 1 Complete: Successfully loaded source CSV. Raw shape: {df.shape}")
        
        # 2. Extract features and targets
        df_num, df_cat, X, y = load_feature(df)
        logger.info(f"Step 2 Complete: Features extracted. Numeric columns: {df_num.shape}, Categorical columns: {df_cat.shape}")
        
        # 3. Create preprocessing transformations
        preprocess = load_preprocess(df_num, df_cat)
        logger.info("Step 3 Complete: Preprocessing configurations successfully created.")
        
        # 4. Initialize model pipeline setup
        model = load_model(preprocess)
        logger.info("Step 4 Complete: Model pipeline setup defined.")
        
        # 5. Stratified splitting of imbalanced dataset
        X_train_val, X_test_val, y_train_val, y_test_val = load_spilt_data(X, y)
        logger.info("Step 5 Complete: Data successfully split into training and test chunks.")
        logger.info(f" -> Training Data Size: {X_train_val.shape} | Test Data Size: {X_test_val.shape}")
        logger.info(f" -> Fraud Prevalence in Training Set: {y_train_val.mean():.4%}")
        
        # 6. Fit the model pipeline
        logger.info("Step 6: Launching model pipeline training (Preprocess -> Resample -> Train)...")
        model.fit(X_train_val, y_train_val)
        logger.info("Pipeline pipeline training completed successfully!")

								#7. Metrics Eval
        metrics = evaluate_fraud_metrics(model, X_test_val, y_test_val)			
        
        return None

    except Exception as e:
        logger.exception("A critical error halted the main pipeline execution loop.")

if __name__ == '__main__':
    main()
