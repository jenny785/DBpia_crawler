"""
DBpia Crawler ML Modules
Integrated machine learning modules from DSBA Lab

Complete ML solution with real-time integration, REST API, and web dashboard.
"""

from .time_series_analyzer import TimeSeriesAnalyzer, TimeSeriesConfig
from .log_anomaly_detector import LogAnomalyDetector, AnomalyScore
from .text_classifier import FakeNewsDetector
from .crawler_integration import CrawlerMLIntegration, PaperValidationResult
from .advanced_text_analyzer import AdvancedTextAnalyzer, AdvancedClassificationResult

__all__ = [
    'TimeSeriesAnalyzer',
    'TimeSeriesConfig',
    'LogAnomalyDetector',
    'AnomalyScore',
    'FakeNewsDetector',
    'CrawlerMLIntegration',
    'PaperValidationResult',
    'AdvancedTextAnalyzer',
    'AdvancedClassificationResult',
]

__version__ = '1.0.0'
__author__ = 'DSBA Lab Integration'
