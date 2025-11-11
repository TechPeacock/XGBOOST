# feature_engineering.py
class SwingFeatureEngineer:
    def __init__(self):
        self.feature_columns = []
        
    def create_price_features(self, df):
        """Create price-based features"""
        df['price_change'] = df['CLOSE_PRICE'] - df['PREV_CLOSE']
        df['price_change_pct'] = (df['price_change'] / df['PREV_CLOSE']) * 100
        df['high_low_spread'] = df['HIGH_PRICE'] - df['LOW_PRICE']
        df['high_low_spread_pct'] = (df['high_low_spread'] / df['PREV_CLOSE']) * 100
        df['close_to_high'] = (df['HIGH_PRICE'] - df['CLOSE_PRICE']) / df['HIGH_PRICE'] * 100
        df['open_close_spread'] = abs(df['OPEN_PRICE'] - df['CLOSE_PRICE'])
        df['price_position'] = (df['CLOSE_PRICE'] - df['LOW_PRICE']) / (df['HIGH_PRICE'] - df['LOW_PRICE'])
        
        return df
    
    def create_volume_features(self, df):
        """Create volume-based features"""
        df['volume_value_ratio'] = df['TTL_TRD_QNTY'] / (df['TURNOVER_LACS'] * 100000 + 1)
        df['avg_trade_size'] = df['TTL_TRD_QNTY'] / (df['NO_OF_TRADES'] + 1)
        df['delivery_strength'] = df['DELIV_QTY'] / (df['TTL_TRD_QNTY'] + 1)
        df['non_delivery_pct'] = 100 - df['DELIV_PER']
        
        return df
    
    def create_technical_features(self, df, volatility_df):
        """Create technical indicators"""
        if not volatility_df.empty:
            df = df.merge(volatility_df[['Symbol', 'Annual_Vol', 'Log_Returns']], 
                         left_on='SYMBOL', right_on='Symbol', how='left')
        
        return df
    
    def create_52week_features(self, df, week52_df):
        """Create 52-week position features"""
        if not week52_df.empty:
            df = df.merge(week52_df[['SYMBOL', 'Adjusted_52_Week_High', 'Adjusted_52_Week_Low']], 
                         on='SYMBOL', how='left')
            
            df['distance_from_52w_high'] = ((df['Adjusted_52_Week_High'] - df['CLOSE_PRICE']) / 
                                           df['Adjusted_52_Week_High'] * 100)
            df['distance_from_52w_low'] = ((df['CLOSE_PRICE'] - df['Adjusted_52_Week_Low']) / 
                                          df['Adjusted_52_Week_Low'] * 100)
            df['52w_range_position'] = ((df['CLOSE_PRICE'] - df['Adjusted_52_Week_Low']) / 
                                       (df['Adjusted_52_Week_High'] - df['Adjusted_52_Week_Low']) * 100)
        
        return df
    
    def create_surveillance_features(self, df, surveillance_df):
        """Create risk/surveillance features"""
        if not surveillance_df.empty:
            # Create risk score (higher = riskier)
            risk_cols = ['GSM', 'Long_Term_Additional_Surveillance_Measure', 
                        'Short_Term_Additional_Surveillance_Measure']
            
            surveillance_df['risk_score'] = 0
            for col in risk_cols:
                if col in surveillance_df.columns:
                    surveillance_df['risk_score'] += (100 - surveillance_df[col])
            
            df = df.merge(surveillance_df[['Symbol', 'risk_score']], 
                         left_on='SYMBOL', right_on='Symbol', how='left')
            df['risk_score'].fillna(0, inplace=True)
        
        return df
    
    def create_multi_day_features(self, df_list):
        """Create features from multiple days of data"""
        # Calculate rolling averages and trends
        combined_features = pd.DataFrame()
        
        for i, df in enumerate(df_list):
            df[f'delivery_day_{i+1}'] = df['DELIV_PER']
            df[f'volume_day_{i+1}'] = df['TTL_TRD_QNTY']
            df[f'price_day_{i+1}'] = df['CLOSE_PRICE']
        
        # Merge all days
        base_df = df_list[0]
        for i in range(1, len(df_list)):
            base_df = base_df.merge(df_list[i][['SYMBOL', f'delivery_day_{i+1}', 
                                                f'volume_day_{i+1}', f'price_day_{i+1}']], 
                                   on='SYMBOL', how='left')
        
        # Calculate trends
        base_df['delivery_trend'] = base_df[[f'delivery_day_{i+1}' for i in range(len(df_list))]].mean(axis=1)
        base_df['volume_trend'] = base_df[[f'volume_day_{i+1}' for i in range(len(df_list))]].mean(axis=1)
        
        return base_df