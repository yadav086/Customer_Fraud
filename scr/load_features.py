import logging

# Configure the logger instance for this specific module
logger = logging.getLogger(__name__)

def load_feature(df):

    try:
        logger.info("Step 2: Starting feature extraction and target partitioning...")
        
        # 1. Separate feature types and drop metadata/target columns if present
        # Using errors='ignore' ensures the script won't crash if these columns aren't in the df
        cols_to_drop = ['transaction_id', 'is_fraud']
        df_num = df.select_dtypes(include='number').drop(columns=cols_to_drop, errors='ignore')
        df_cat = df.select_dtypes(exclude='number')

        logger.debug(f"Identified {len(df_num.columns)} numeric and {len(df_cat.columns)} categorical features.")

        # 2. Combine columns to form the feature matrix X, and isolate target y
        feature_columns = df_num.columns.to_list() + df_cat.columns.to_list()
        X = df[feature_columns]
        y = df['is_fraud']

        logger.info("Successfully extracted features and target vector.")
        logger.info(f" -> Matrix X Shape: {X.shape} | Target y Shape: {y.shape}")

        return df_num, df_cat, X, y

    except KeyError as ke:
        logger.error(f"Target column 'is_fraud' was not found in the dataset: {ke}")
        raise ke
    except Exception as e:
        logger.exception("A critical exception occurred during the feature separation step.")
        raise e
