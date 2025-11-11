# data_processor.py
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class NSEDataProcessor:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.equity_path = self.base_path / "equity"
        self.fo_path = self.base_path / "fo"
        self.debt_path = self.base_path / "debt"
        
    def load_equity_data(self, date_str):
        """Load and clean equity bhavdata"""
        try:
            file_path = self.equity_path / f"sec_bhavdata_full_{date_str}.csv"
            df = pd.read_csv(file_path)
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Convert data types
            df['DATE1'] = pd.to_datetime(df['DATE1'].str.strip(), format='%d-%b-%Y')
            df['DELIV_QTY'] = pd.to_numeric(df['DELIV_QTY'].str.replace(',', '').str.strip(), errors='coerce')
            df['DELIV_PER'] = pd.to_numeric(df['DELIV_PER'].str.strip(), errors='coerce')
            
            return df
        except Exception as e:
            print(f"Error loading equity data: {e}")
            return pd.DataFrame()
    
    def load_volatility_data(self, date_str):
        """Load volatility data"""
        try:
            file_path = self.equity_path / f"CMVOLT_{date_str}.CSV"
            df = pd.read_csv(file_path)
            df.columns = ['Date', 'Symbol', 'Close_Price', 'Prev_Close', 
                         'Log_Returns', 'Prev_Volatility', 'Daily_Vol', 'Annual_Vol']
            return df
        except:
            return pd.DataFrame()
    
    def load_pe_data(self, date_str):
        """Load PE ratio data"""
        try:
            file_path = self.equity_path / f"PE_{date_str[:6]}.csv"
            df = pd.read_csv(file_path)
            return df
        except:
            return pd.DataFrame()
    
    def load_52week_data(self, date_str):
        """Load 52-week high/low data"""
        try:
            file_path = self.equity_path / f"CM_52_wk_High_low_{date_str}.csv"
            # Handle parsing errors
            df = pd.read_csv(file_path, error_bad_lines=False, warn_bad_lines=False)
            return df
        except:
            return pd.DataFrame()
    
    def load_fo_participant_data(self, date_str):
        """Load F&O participant OI data"""
        try:
            file_path = self.fo_path / f"fao_participant_oi_{date_str}.csv"
            df = pd.read_csv(file_path, skiprows=1)
            return df
        except:
            return pd.DataFrame()
    
    def load_surveillance_data(self, date_str):
        """Load surveillance/regulatory data"""
        try:
            file_path = self.equity_path / f"REG_IND{date_str[:6]}.csv"
            df = pd.read_csv(file_path)
            return df
        except:
            return pd.DataFrame()