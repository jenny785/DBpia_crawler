"""
Crawler Integration Module
Integrates ML modules with DBpia crawler for real-time analysis
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

from time_series_analyzer import TimeSeriesAnalyzer
from log_anomaly_detector import LogAnomalyDetector
from text_classifier import FakeNewsDetector

logger = logging.getLogger(__name__)


@dataclass
class PaperValidationResult:
    """Comprehensive validation result for a paper"""
    title: str
    authors: List[str]
    journal: str
    date: str

    # Validation results
    title_credibility: float
    abstract_credibility: float
    quality_score: float

    is_valid: bool
    warnings: List[str]
    timestamp: str


class CrawlerMLIntegration:
    """
    Integrates ML modules with DBpia crawler for real-time validation and analysis
    """

    def __init__(self):
        self.text_classifier = FakeNewsDetector()
        self.log_detector = LogAnomalyDetector(threshold=0.7)
        self.ts_analyzer = TimeSeriesAnalyzer()

        self.logger = logger
        self.collected_papers = []
        self.crawler_logs = []
        self.publication_dates = []
        self.publication_volumes = []

    def validate_paper(self, paper: Dict) -> PaperValidationResult:
        """
        Validate a paper collected by crawler

        Args:
            paper: Paper dictionary with title, abstract, authors, journal, date

        Returns:
            Validation result with credibility scores
        """
        warnings = []

        # Validate title
        title_result = self.text_classifier.classify_title(paper.get('title', ''))
        title_credibility = 1.0 - title_result.confidence

        if title_result.class_label == 'suspicious':
            warnings.append(f"Suspicious title: {title_result.explanation}")

        # Validate abstract (if available)
        abstract_credibility = 1.0
        if paper.get('abstract'):
            abstract_result = self.text_classifier.classify_abstract(paper['abstract'])
            abstract_credibility = 1.0 - abstract_result.confidence

            if abstract_result.class_label == 'questionable':
                warnings.append(f"Abstract credibility: {abstract_result.explanation}")

        # Calculate overall quality score
        quality_score = (title_credibility + abstract_credibility) / 2

        # Determine if paper is valid
        is_valid = quality_score > 0.6 and title_result.class_label != 'suspicious'

        result = PaperValidationResult(
            title=paper.get('title', ''),
            authors=paper.get('authors', []),
            journal=paper.get('journal', ''),
            date=paper.get('date', ''),
            title_credibility=title_credibility,
            abstract_credibility=abstract_credibility,
            quality_score=quality_score,
            is_valid=is_valid,
            warnings=warnings,
            timestamp=datetime.now().isoformat()
        )

        self.collected_papers.append(result)
        self.logger.info(
            f"Validated paper: {paper.get('title', 'Unknown')} "
            f"(score: {quality_score:.2f}, valid: {is_valid})"
        )

        return result

    def validate_batch(self, papers: List[Dict]) -> List[PaperValidationResult]:
        """
        Validate multiple papers

        Args:
            papers: List of paper dictionaries

        Returns:
            List of validation results
        """
        results = []
        for paper in papers:
            result = self.validate_paper(paper)
            results.append(result)

        self.logger.info(f"Validated {len(papers)} papers")
        return results

    def analyze_crawler_logs(self, logs: List[str]) -> dict:
        """
        Analyze crawler execution logs for anomalies

        Args:
            logs: List of log entries

        Returns:
            Anomaly analysis summary
        """
        self.crawler_logs.extend(logs)

        anomaly_results = self.log_detector.analyze_logs(logs)
        summary = self.log_detector.get_anomaly_summary(anomaly_results)

        self.logger.info(
            f"Log analysis: {summary['total_anomalies']} anomalies detected, "
            f"anomaly rate: {summary['anomaly_rate']:.2%}"
        )

        return {
            'summary': summary,
            'results': anomaly_results,
            'critical_issues': [
                r for r in anomaly_results if r.severity == 'high'
            ]
        }

    def analyze_publication_trends(
        self,
        dates: Optional[List[str]] = None,
        volumes: Optional[List[int]] = None
    ) -> dict:
        """
        Analyze publication trends over time

        Args:
            dates: List of dates (or None to use collected dates)
            volumes: List of volumes (or None to use collected volumes)

        Returns:
            Trend analysis results
        """
        if dates is None:
            dates = self.publication_dates
        if volumes is None:
            volumes = self.publication_volumes

        if not dates or not volumes:
            self.logger.warning("Insufficient data for trend analysis")
            return {}

        result = self.ts_analyzer.analyze_publication_trends(dates, volumes)

        # Detect anomalies in publication patterns
        anomalies = self.ts_analyzer.detect_anomalies(volumes)

        self.logger.info(
            f"Trend analysis: {result['trend']} trend detected, "
            f"{len(anomalies)} anomalies in publication pattern"
        )

        return {
            'trend_analysis': result,
            'anomalies': anomalies,
            'forecast': self.ts_analyzer.forecast_next(volumes, steps=7).tolist()
        }

    def track_paper_date(self, date: str) -> None:
        """Track publication date for trend analysis"""
        self.publication_dates.append(date)
        self.publication_volumes.append(
            sum(1 for d in self.publication_dates if d == date)
        )

    def get_validation_report(self) -> dict:
        """
        Generate comprehensive validation report

        Returns:
            Report with statistics and warnings
        """
        if not self.collected_papers:
            return {'status': 'no_data'}

        total = len(self.collected_papers)
        valid = sum(1 for p in self.collected_papers if p.is_valid)
        suspicious = sum(
            1 for p in self.collected_papers
            if p.title_credibility < 0.5
        )

        avg_quality = sum(p.quality_score for p in self.collected_papers) / total

        return {
            'total_papers': total,
            'valid_papers': valid,
            'suspicious_papers': suspicious,
            'validation_rate': valid / total,
            'average_quality_score': avg_quality,
            'papers_with_warnings': [
                p for p in self.collected_papers if p.warnings
            ]
        }

    def get_system_health(self) -> dict:
        """
        Get overall system health based on crawler logs and validation

        Returns:
            Health status with recommendations
        """
        if not self.crawler_logs:
            return {'status': 'no_logs'}

        log_analysis = self.analyze_crawler_logs(self.crawler_logs)
        validation_report = self.get_validation_report()

        health_score = (
            (1 - log_analysis['summary'].get('anomaly_rate', 0)) * 0.5 +
            validation_report.get('validation_rate', 0) * 0.5
        )

        recommendations = []
        if log_analysis['summary'].get('high_severity_count', 0) > 0:
            recommendations.append("Fix critical log errors")
        if validation_report.get('suspicious_papers', 0) > total_papers * 0.1:
            recommendations.append("Review suspicious papers")

        return {
            'health_score': health_score,
            'status': 'healthy' if health_score > 0.8 else 'needs_attention' if health_score > 0.5 else 'critical',
            'recommendations': recommendations,
            'details': {
                'log_analysis': log_analysis['summary'],
                'validation_report': validation_report
            }
        }
