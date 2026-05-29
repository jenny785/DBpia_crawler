"""
Time Series Analysis Module
Based on TS-Unity framework

Provides time series forecasting, anomaly detection, and classification
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesConfig:
    """Configuration for time series analysis"""
    window_size: int = 12
    forecast_horizon: int = 1
    threshold_std: float = 3.0
    model_type: str = 'transformer'


class TimeSeriesAnalyzer:
    """
    Time series analyzer for DBpia crawler data

    Use cases:
    - Forecast paper publication volume over time
    - Detect anomalies in crawling patterns
    - Classify research trends
    """

    def __init__(self, config: Optional[TimeSeriesConfig] = None):
        self.config = config or TimeSeriesConfig()
        self.logger = logger

    def analyze_publication_trends(self, dates: List[str], volumes: List[int]) -> dict:
        """
        Analyze publication volume trends over time

        Args:
            dates: List of dates
            volumes: List of publication volumes

        Returns:
            Dictionary with trend analysis results
        """
        df = pd.DataFrame({'date': pd.to_datetime(dates), 'volume': volumes})
        df = df.sort_values('date')

        result = {
            'mean': float(np.mean(volumes)),
            'std': float(np.std(volumes)),
            'min': float(np.min(volumes)),
            'max': float(np.max(volumes)),
            'trend': self._calculate_trend(volumes),
            'seasonality': self._detect_seasonality(volumes),
        }

        self.logger.info(f"Publication trend analysis completed: mean={result['mean']:.2f}")
        return result

    def detect_anomalies(self, time_series: List[float]) -> List[int]:
        """
        Detect anomalies using statistical method (3-sigma rule)

        Args:
            time_series: Input time series data

        Returns:
            Indices of detected anomalies
        """
        arr = np.array(time_series)
        mean = np.mean(arr)
        std = np.std(arr)
        threshold = self.config.threshold_std

        anomalies = np.where(np.abs(arr - mean) > threshold * std)[0].tolist()
        self.logger.info(f"Detected {len(anomalies)} anomalies")
        return anomalies

    def forecast_next(self, time_series: List[float], steps: int = 1) -> np.ndarray:
        """
        Simple exponential smoothing forecast

        Args:
            time_series: Historical data
            steps: Number of steps to forecast

        Returns:
            Forecasted values
        """
        arr = np.array(time_series)
        alpha = 0.3

        result = [arr[-1]]
        for _ in range(steps):
            next_val = alpha * arr[-1] + (1 - alpha) * result[-1]
            result.append(next_val)

        self.logger.info(f"Forecasted {steps} steps ahead")
        return np.array(result[1:])

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return 'insufficient_data'

        first_half = np.mean(values[:len(values)//2])
        second_half = np.mean(values[len(values)//2:])

        if second_half > first_half * 1.05:
            return 'upward'
        elif second_half < first_half * 0.95:
            return 'downward'
        else:
            return 'stable'

    def _detect_seasonality(self, values: List[float]) -> bool:
        """Detect seasonal pattern"""
        if len(values) < 24:
            return False

        # Simple autocorrelation check
        arr = np.array(values)
        lag = min(12, len(values) // 2)
        autocorr = np.corrcoef(arr[:-lag], arr[lag:])[0, 1]

        return autocorr > 0.5
