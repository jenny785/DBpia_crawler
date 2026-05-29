"""
Log Anomaly Detection Module
Based on RAPID (Training-free Retrieval-based Log Anomaly Detection)

Detects anomalies in crawler execution logs using pre-trained language models
"""

import re
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AnomalyScore:
    """Anomaly detection result"""
    log_entry: str
    timestamp: str
    anomaly_score: float
    is_anomaly: bool
    severity: str  # 'low', 'medium', 'high'


class LogAnomalyDetector:
    """
    Detects anomalies in DBpia crawler logs

    Use cases:
    - Monitor crawler execution patterns
    - Alert on unusual behavior
    - Identify system failures early
    """

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.logger = logger
        self.log_patterns = {
            'error': re.compile(r'ERROR|Failed|Exception|Traceback'),
            'warning': re.compile(r'WARNING|Warn|Deprecated'),
            'timeout': re.compile(r'timeout|Time out|Connection timeout'),
            'network': re.compile(r'Connection refused|Network unreachable|DNS'),
            'database': re.compile(r'Database|SQL|Query failed'),
        }

    def analyze_logs(self, log_entries: List[str]) -> List[AnomalyScore]:
        """
        Analyze log entries for anomalies

        Args:
            log_entries: List of log lines

        Returns:
            List of anomaly scores
        """
        results = []

        for i, log in enumerate(log_entries):
            timestamp = self._extract_timestamp(log)
            anomaly_score = self._compute_anomaly_score(log)
            is_anomaly = anomaly_score > self.threshold
            severity = self._determine_severity(anomaly_score)

            result = AnomalyScore(
                log_entry=log,
                timestamp=timestamp,
                anomaly_score=anomaly_score,
                is_anomaly=is_anomaly,
                severity=severity
            )
            results.append(result)

        self.logger.info(
            f"Analyzed {len(log_entries)} logs, "
            f"found {sum(1 for r in results if r.is_anomaly)} anomalies"
        )
        return results

    def _extract_timestamp(self, log: str) -> str:
        """Extract timestamp from log entry"""
        timestamp_pattern = re.compile(
            r'(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})|'
            r'(\d{2}/\w+/\d{4}\s\d{2}:\d{2}:\d{2})'
        )
        match = timestamp_pattern.search(log)
        return match.group(0) if match else 'unknown'

    def _compute_anomaly_score(self, log: str) -> float:
        """
        Compute anomaly score using pattern matching

        Simple rule-based approach for now, can be replaced with PLM-based scoring
        """
        score = 0.0

        for pattern_type, pattern in self.log_patterns.items():
            if pattern.search(log):
                pattern_scores = {
                    'error': 0.9,
                    'warning': 0.6,
                    'timeout': 0.8,
                    'network': 0.8,
                    'database': 0.7,
                }
                score = max(score, pattern_scores.get(pattern_type, 0.5))

        # Check for repeated errors (indicates issue escalation)
        if '[' in log and ']' in log:
            count_match = re.search(r'\[(\d+)x?\]', log)
            if count_match:
                count = int(count_match.group(1))
                if count > 3:
                    score = min(1.0, score + 0.2)

        return score

    def _determine_severity(self, score: float) -> str:
        """Determine severity level from score"""
        if score >= 0.85:
            return 'high'
        elif score >= 0.70:
            return 'medium'
        else:
            return 'low'

    def get_anomaly_summary(self, anomalies: List[AnomalyScore]) -> dict:
        """
        Generate summary statistics for anomalies

        Args:
            anomalies: List of anomaly scores

        Returns:
            Summary dictionary
        """
        high_severity = [a for a in anomalies if a.severity == 'high']
        medium_severity = [a for a in anomalies if a.severity == 'medium']

        return {
            'total_anomalies': len([a for a in anomalies if a.is_anomaly]),
            'high_severity_count': len(high_severity),
            'medium_severity_count': len(medium_severity),
            'anomaly_rate': len([a for a in anomalies if a.is_anomaly]) / len(anomalies)
            if anomalies else 0,
            'average_score': sum(a.anomaly_score for a in anomalies) / len(anomalies)
            if anomalies else 0,
        }
