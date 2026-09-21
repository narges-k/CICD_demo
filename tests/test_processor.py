import pytest
import pandas as pd
from src.data_processor import SmartMeterProcessor

@pytest.fixture
def clean_data():
    return [
        {'timestamp': '2026-09-21T10:00:00', 'consumption': 1.0},
        {'timestamp': '2026-09-21T10:01:00', 'consumption': 1.1},
        {'timestamp': '2026-09-21T10:02:00', 'consumption': 1.2},
    ]

def test_missing_values_interpolation(clean_data):
    test_data = clean_data.copy()
    test_data[1]['consumption'] = None
    
    processor = SmartMeterProcessor(test_data)
    df = processor.clean_and_process()
    
    assert pd.notna(df.iloc[1]['consumption'])
    assert df.iloc[1]['consumption'] == 1.1

def test_outlier_removal_and_processing(clean_data):
    test_data = clean_data.copy()
    test_data[1]['consumption'] = 50.0
    
    processor = SmartMeterProcessor(test_data)
    df = processor.clean_and_process()
    
    assert df.iloc[1]['consumption'] < 10.0
    assert round(df.iloc[1]['consumption'], 1) == 1.1

def test_peak_demand_calculation():
    times = pd.date_range('2026-09-21T10:00:00', periods=120, freq='1min')
    consumption = [1.0] * 60 + [5.0] * 60
    data = [{'timestamp': t.isoformat(), 'consumption': c} for t, c in zip(times, consumption)]
    
    processor = SmartMeterProcessor(data)
    processed_df = processor.clean_and_process()
    peak_stats = processor.calculate_peak_demand(processed_df)
    
    assert peak_stats['peak_timestamp'].hour == 11
    assert peak_stats['peak_consumption_kwh'] == 5.0

def test_empty_data_raises_error():
    with pytest.raises(ValueError, match="Input data list is empty"):
        SmartMeterProcessor([])
