import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmartMeterProcessor:
    def __init__(self, raw_data_list):
        if not raw_data_list:
            raise ValueError("Input data list is empty.")
        self.raw_data_list = raw_data_list

    def _to_dataframe(self):
        df = pd.DataFrame(self.raw_data_list)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by='timestamp').reset_index(drop=True)
        return df

    def clean_and_process(self):
        try:
            df = self._to_dataframe()
            logger.info("Converting data to DataFrame and sorting completed.")

            df['consumption'] = df['consumption'].interpolate(method='linear')
            df['consumption'] = df['consumption'].ffill().bfill()
            logger.info("Missing values handling completed.")

            rolling_window = df.rolling('5min', on='timestamp')
            rolling_mean = rolling_window['consumption'].mean()
            rolling_std = rolling_window['consumption'].std()

            outlier_condition = np.abs(df['consumption'] - rolling_mean) > (3 * rolling_std)
            df.loc[outlier_condition.fillna(False), 'consumption'] = np.nan
            
            df['consumption'] = df['consumption'].interpolate(method='linear').ffill().bfill()
            logger.info("Outlier removal and re-interpolation completed.")
            
            return df
        
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            raise e

    def calculate_peak_demand(self, processed_df):
        df_indexed = processed_df.set_index('timestamp')
        hourly_summary = df_indexed['consumption'].resample('1H').mean()
        
        peak_hour = hourly_summary.idxmax()
        peak_value = hourly_summary.max()
        
        return {
            'peak_timestamp': peak_hour,
            'peak_consumption_kwh': round(peak_value, 3),
            'hourly_average': round(hourly_summary.mean(), 3)
        }
