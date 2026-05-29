"""
DBpia Crawler ML API
FastAPI-based REST API for ML modules
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import logging
from datetime import datetime

import sys
sys.path.insert(0, '../ml_modules')

from ml_modules.crawler_integration import CrawlerMLIntegration, PaperValidationResult
from ml_modules.time_series_analyzer import TimeSeriesAnalyzer
from ml_modules.log_anomaly_detector import LogAnomalyDetector
from ml_modules.text_classifier import FakeNewsDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DBpia Crawler ML API",
    description="Machine learning integration API for DBpia crawler",
    version="1.0.0"
)

# Global integration instance
integration = CrawlerMLIntegration()


# Pydantic Models
class PaperInput(BaseModel):
    """Paper data input model"""
    title: str
    abstract: Optional[str] = None
    authors: Optional[List[str]] = []
    journal: str
    date: str
    url: Optional[str] = None


class BulkPaperInput(BaseModel):
    """Bulk paper validation input"""
    papers: List[PaperInput]


class LogEntry(BaseModel):
    """Log entry input model"""
    timestamp: str
    level: str
    message: str


class TimeSeriesInput(BaseModel):
    """Time series data input"""
    dates: List[str]
    volumes: List[int]


class ValidationResponse(BaseModel):
    """Validation response model"""
    title: str
    quality_score: float
    is_valid: bool
    warnings: List[str]
    timestamp: str


# Health check endpoint
@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "healthy",
        "service": "DBpia Crawler ML API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Detailed health status"""
    health_status = integration.get_system_health()
    return health_status


# Paper Validation Endpoints
@app.post("/validate/paper")
async def validate_paper(paper: PaperInput):
    """
    Validate a single paper

    Args:
        paper: Paper data

    Returns:
        Validation result
    """
    try:
        result = integration.validate_paper(paper.dict())
        return {
            "title": result.title,
            "quality_score": result.quality_score,
            "is_valid": result.is_valid,
            "title_credibility": result.title_credibility,
            "abstract_credibility": result.abstract_credibility,
            "warnings": result.warnings,
            "timestamp": result.timestamp
        }
    except Exception as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/validate/batch")
async def validate_batch(request: BulkPaperInput, background_tasks: BackgroundTasks):
    """
    Validate multiple papers

    Args:
        request: Bulk validation request
        background_tasks: Background task handler

    Returns:
        Bulk validation results
    """
    try:
        papers = [p.dict() for p in request.papers]
        results = integration.validate_batch(papers)

        return {
            "total_papers": len(results),
            "valid_papers": sum(1 for r in results if r.is_valid),
            "average_quality": sum(r.quality_score for r in results) / len(results),
            "results": [
                {
                    "title": r.title,
                    "quality_score": r.quality_score,
                    "is_valid": r.is_valid,
                    "warnings": r.warnings
                }
                for r in results
            ]
        }
    except Exception as e:
        logger.error(f"Batch validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/validation/report")
async def get_validation_report():
    """Get validation summary report"""
    try:
        report = integration.get_validation_report()
        return report
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# Log Analysis Endpoints
@app.post("/logs/analyze")
async def analyze_logs(logs: List[LogEntry]):
    """
    Analyze crawler logs for anomalies

    Args:
        logs: List of log entries

    Returns:
        Anomaly analysis results
    """
    try:
        log_strings = [
            f"[{log.timestamp}] {log.level}: {log.message}"
            for log in logs
        ]
        results = integration.analyze_crawler_logs(log_strings)

        return {
            "total_logs": len(logs),
            "anomalies": results['summary']['total_anomalies'],
            "high_severity": results['summary']['high_severity_count'],
            "anomaly_rate": results['summary']['anomaly_rate'],
            "average_score": results['summary']['average_score'],
            "critical_issues": [
                {
                    "timestamp": r.timestamp,
                    "severity": r.severity,
                    "anomaly_score": r.anomaly_score,
                    "log": r.log_entry[:100] + "..." if len(r.log_entry) > 100 else r.log_entry
                }
                for r in results['critical_issues']
            ]
        }
    except Exception as e:
        logger.error(f"Log analysis error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# Time Series Endpoints
@app.post("/timeseries/analyze")
async def analyze_timeseries(data: TimeSeriesInput):
    """
    Analyze publication trends

    Args:
        data: Time series data (dates and volumes)

    Returns:
        Trend analysis and forecast
    """
    try:
        result = integration.ts_analyzer.analyze_publication_trends(
            data.dates, data.volumes
        )

        # Get forecast
        forecast = integration.ts_analyzer.forecast_next(data.volumes, steps=7)

        # Detect anomalies
        anomalies = integration.ts_analyzer.detect_anomalies(data.volumes)

        return {
            "trend": result['trend'],
            "mean": result['mean'],
            "std": result['std'],
            "min": result['min'],
            "max": result['max'],
            "seasonality": result['seasonality'],
            "forecast_next_7_days": forecast.tolist(),
            "anomalies": [
                {
                    "index": i,
                    "value": float(data.volumes[i]),
                    "date": data.dates[i]
                }
                for i in anomalies
            ]
        }
    except Exception as e:
        logger.error(f"Time series analysis error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/timeseries/forecast")
async def forecast(data: TimeSeriesInput, steps: int = 7):
    """
    Forecast future values

    Args:
        data: Historical time series data
        steps: Number of steps to forecast

    Returns:
        Forecast values
    """
    try:
        forecast = integration.ts_analyzer.forecast_next(data.volumes, steps=steps)
        return {
            "forecast_steps": steps,
            "values": forecast.tolist(),
            "last_historical_value": float(data.volumes[-1]),
            "last_historical_date": data.dates[-1]
        }
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# Text Classification Endpoints
@app.post("/classify/title")
async def classify_title(title: str):
    """
    Classify paper title credibility

    Args:
        title: Paper title

    Returns:
        Classification result
    """
    try:
        result = integration.text_classifier.classify_title(title)
        return {
            "text": result.text,
            "class": result.class_label,
            "confidence": result.confidence,
            "explanation": result.explanation
        }
    except Exception as e:
        logger.error(f"Title classification error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/classify/abstract")
async def classify_abstract(abstract: str):
    """
    Classify abstract credibility

    Args:
        abstract: Paper abstract

    Returns:
        Classification result
    """
    try:
        result = integration.text_classifier.classify_abstract(abstract)
        return {
            "text": result.text,
            "class": result.class_label,
            "confidence": result.confidence,
            "explanation": result.explanation
        }
    except Exception as e:
        logger.error(f"Abstract classification error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# Statistics Endpoints
@app.get("/stats/papers")
async def get_paper_stats():
    """Get paper validation statistics"""
    try:
        report = integration.get_validation_report()
        return report
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/stats/logs")
async def get_log_stats():
    """Get crawler log statistics"""
    try:
        if not integration.crawler_logs:
            return {"status": "no_logs"}

        analysis = integration.analyze_crawler_logs(integration.crawler_logs)
        return analysis['summary']
    except Exception as e:
        logger.error(f"Log stats error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
