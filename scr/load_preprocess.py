from imblearn.pipeline import  Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PowerTransformer
import logging
logger = logging.getLogger(__name__)

def load_preprocess(df_num, df_cat):
    try: 
        tranform_num = Pipeline(steps=[('nums', PowerTransformer(method='yeo-johnson'))])
        tranform_cat = Pipeline(steps = [('cat',OneHotEncoder(drop='first',sparse_output=False, handle_unknown='ignore'))])
        tranform_cat_ordinal = Pipeline(steps = [('cat_ord', OrdinalEncoder())])

        preprocess = ColumnTransformer(transformers=[('num',tranform_num,df_num.columns.to_list()),
                                                    ('cat',tranform_cat, df_cat[['merchant_category', 'card_type', 'auth_method', 'channel','device_type']].columns.to_list()),
                                                    ('cat_ord',tranform_cat_ordinal , df_cat[['is_foreign_transaction', 'is_new_merchant', 'used_vpn','ip_country_mismatch', 'billing_shipping_mismatch','is_ai_generated_scam_attempt']].columns.to_list())                                                                                                                                       
                                                    ])

        return preprocess
    except Exception as e:
        logger.exception('Error in load_preprocess')
        raise
    