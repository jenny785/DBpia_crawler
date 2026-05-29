"""
Test suite for ML modules

Tests the integration of DSBA Lab ML components
"""

import unittest
from time_series_analyzer import TimeSeriesAnalyzer, TimeSeriesConfig
from log_anomaly_detector import LogAnomalyDetector
from text_classifier import FakeNewsDetector


class TestTimeSeriesAnalyzer(unittest.TestCase):
    """Tests for time series analyzer"""

    def setUp(self):
        self.analyzer = TimeSeriesAnalyzer()

    def test_publication_trends(self):
        """Test publication trend analysis"""
        dates = ['2024-01-01', '2024-01-02', '2024-01-03']
        volumes = [10, 15, 20]

        result = self.analyzer.analyze_publication_trends(dates, volumes)

        self.assertEqual(result['mean'], 15.0)
        self.assertEqual(result['trend'], 'upward')
        self.assertIn('seasonality', result)

    def test_anomaly_detection(self):
        """Test anomaly detection"""
        time_series = [1, 1, 1, 1, 100, 1, 1, 1]  # 100 is anomaly

        anomalies = self.analyzer.detect_anomalies(time_series)

        self.assertIn(4, anomalies)

    def test_forecasting(self):
        """Test forecasting"""
        history = [1, 2, 3, 4, 5]

        forecast = self.analyzer.forecast_next(history, steps=2)

        self.assertEqual(len(forecast), 2)
        self.assertTrue(all(f >= 0 for f in forecast))


class TestLogAnomalyDetector(unittest.TestCase):
    """Tests for log anomaly detector"""

    def setUp(self):
        self.detector = LogAnomalyDetector(threshold=0.7)

    def test_error_detection(self):
        """Test error detection in logs"""
        logs = [
            "[2024-01-01 10:00:00] Starting crawler",
            "[2024-01-01 10:00:01] ERROR: Connection timeout",
            "[2024-01-01 10:00:02] Processing complete",
        ]

        results = self.detector.analyze_logs(logs)

        self.assertEqual(len(results), 3)
        self.assertTrue(results[1].is_anomaly)  # Error log

    def test_anomaly_summary(self):
        """Test anomaly summary generation"""
        logs = [
            "[2024-01-01 10:00:00] Normal operation",
            "[2024-01-01 10:00:01] ERROR: Failed connection",
        ]

        results = self.detector.analyze_logs(logs)
        summary = self.detector.get_anomaly_summary(results)

        self.assertIn('total_anomalies', summary)
        self.assertIn('high_severity_count', summary)


class TestFakeNewsDetector(unittest.TestCase):
    """Tests for text classification"""

    def setUp(self):
        self.detector = FakeNewsDetector()

    def test_clickbait_detection(self):
        """Test clickbait title detection"""
        title = "You Won't Believe What Scientists Discovered!"

        result = self.detector.classify_title(title)

        self.assertEqual(result.class_label, 'suspicious')
        self.assertGreater(result.confidence, 0.5)

    def test_reliable_title(self):
        """Test reliable title classification"""
        title = "A Study of Machine Learning Algorithms in Natural Language Processing"

        result = self.detector.classify_title(title)

        self.assertNotEqual(result.class_label, 'suspicious')

    def test_batch_classification(self):
        """Test batch classification"""
        titles = [
            "Normal Research Paper Title",
            "Amazing Discovery That Doctors Hide From You!",
            "Comparative Analysis of Deep Learning Methods",
        ]

        results = self.detector.batch_classify(titles, mode='title')

        self.assertEqual(len(results), 3)
        # Second title should be flagged as suspicious
        self.assertGreater(results[1].confidence, results[0].confidence)

    def test_abstract_classification(self):
        """Test abstract classification"""
        abstract = (
            "This study examines the effectiveness of machine learning models "
            "in text classification tasks. Using a dataset of 10,000 documents, "
            "we found that transformer-based models achieved 95% accuracy. "
            "Results were validated using cross-validation techniques."
        )

        result = self.detector.classify_abstract(abstract)

        self.assertIn(result.class_label, ['credible', 'neutral', 'questionable'])
        self.assertGreater(result.confidence, 0)


class IntegrationTest(unittest.TestCase):
    """Integration tests for all modules"""

    def test_workflow(self):
        """Test complete analysis workflow"""
        # Setup analyzers
        ts_analyzer = TimeSeriesAnalyzer()
        log_detector = LogAnomalyDetector()
        text_detector = FakeNewsDetector()

        # Time series analysis
        dates = ['2024-01-' + str(i).zfill(2) for i in range(1, 11)]
        volumes = [10, 12, 11, 13, 50, 12, 11, 10, 11, 12]
        ts_result = ts_analyzer.analyze_publication_trends(dates, volumes)
        self.assertIsNotNone(ts_result)

        # Log analysis
        logs = ["[2024-01-01 10:00:00] Process started", "[2024-01-01 10:00:01] ERROR: Issue found"]
        log_results = log_detector.analyze_logs(logs)
        self.assertEqual(len(log_results), 2)

        # Text analysis
        title = "New Research Findings in AI"
        text_result = text_detector.classify_title(title)
        self.assertIsNotNone(text_result)


if __name__ == '__main__':
    unittest.main()
