def load_feature(df):
    
    df_num =df.select_dtypes(include ='number').drop(['transaction_id','is_fraud'],axis =1 )
    df_cat =df.select_dtypes(exclude ='number')

    X = df[ df_num.columns.to_list() +df_cat.columns.to_list() ]
    y = df['is_fraud']

    return df_num,df_cat, X,y