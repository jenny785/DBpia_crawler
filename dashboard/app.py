"""
DBpia Crawler ML Dashboard
Real-time monitoring and analysis dashboard using Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys

sys.path.insert(0, '../ml_modules')

from ml_modules.crawler_integration import CrawlerMLIntegration
from ml_modules.time_series_analyzer import TimeSeriesAnalyzer
from ml_modules.log_anomaly_detector import LogAnomalyDetector
from ml_modules.text_classifier import FakeNewsDetector

# Page configuration
st.set_page_config(
    page_title="DBpia Crawler ML Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'integration' not in st.session_state:
    st.session_state.integration = CrawlerMLIntegration()

integration = st.session_state.integration

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success-metric {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .warning-metric {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .danger-metric {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("📊 DBpia Crawler ML Dashboard")
st.markdown("Real-time monitoring and analysis of paper collection and validation")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["📈 Overview", "📄 Paper Validation", "📋 Log Analysis", "📊 Trends", "⚙️ Settings"]
)

# ===== PAGE 1: OVERVIEW =====
if page == "📈 Overview":
    st.header("System Overview")

    # System Health
    col1, col2, col3, col4 = st.columns(4)

    health = integration.get_system_health()

    with col1:
        st.metric(
            "System Health",
            f"{health.get('health_score', 0):.1%}",
            "Healthy" if health.get('status') == 'healthy' else "Needs Attention",
            delta=None
        )

    with col2:
        report = integration.get_validation_report()
        st.metric(
            "Papers Validated",
            report.get('total_papers', 0),
            f"{report.get('validation_rate', 0):.1%} Valid"
        )

    with col3:
        st.metric(
            "Average Quality",
            f"{report.get('average_quality_score', 0):.2f}/1.0",
            "Score"
        )

    with col4:
        suspicious = report.get('suspicious_papers', 0)
        st.metric(
            "Suspicious Papers",
            suspicious,
            "⚠️ Alert" if suspicious > 0 else "✓ None"
        )

    st.divider()

    # Key Metrics
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Validation Statistics")
        if report.get('total_papers', 0) > 0:
            stats_df = pd.DataFrame({
                'Status': ['Valid', 'Suspicious', 'Warning'],
                'Count': [
                    report.get('valid_papers', 0),
                    report.get('suspicious_papers', 0),
                    len(report.get('papers_with_warnings', []))
                ]
            })

            fig = px.pie(
                stats_df, values='Count', names='Status',
                color_discrete_map={
                    'Valid': '#28a745',
                    'Suspicious': '#dc3545',
                    'Warning': '#ffc107'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No papers validated yet")

    with col2:
        st.subheader("System Health Details")
        if 'details' in health:
            health_details = health['details']
            col_a, col_b = st.columns(2)

            with col_a:
                st.write("**Crawler Logs**")
                log_summary = health_details.get('log_analysis', {})
                st.write(f"Total Anomalies: {log_summary.get('total_anomalies', 0)}")
                st.write(f"High Severity: {log_summary.get('high_severity_count', 0)}")
                st.write(f"Anomaly Rate: {log_summary.get('anomaly_rate', 0):.2%}")

            with col_b:
                st.write("**Paper Validation**")
                val_summary = health_details.get('validation_report', {})
                st.write(f"Total Papers: {val_summary.get('total_papers', 0)}")
                st.write(f"Valid: {val_summary.get('valid_papers', 0)}")
                st.write(f"Validation Rate: {val_summary.get('validation_rate', 0):.2%}")

    st.divider()

    # Recommendations
    st.subheader("📌 Recommendations")
    if health.get('recommendations'):
        for i, rec in enumerate(health['recommendations'], 1):
            st.warning(f"{i}. {rec}")
    else:
        st.success("✓ No critical issues detected")


# ===== PAGE 2: PAPER VALIDATION =====
elif page == "📄 Paper Validation":
    st.header("Paper Validation")

    tabs = st.tabs(["Single Paper", "Batch Validation", "Results"])

    with tabs[0]:
        st.subheader("Validate Single Paper")

        col1, col2 = st.columns(2)
        with col1:
            title = st.text_input("Paper Title", placeholder="Enter paper title...")
        with col2:
            journal = st.text_input("Journal/Conference", placeholder="Enter journal name...")

        abstract = st.text_area(
            "Abstract (Optional)",
            placeholder="Enter paper abstract...",
            height=100
        )

        col1, col2 = st.columns(2)
        with col1:
            authors = st.text_input("Authors (comma-separated)", placeholder="Author1, Author2...")
        with col2:
            date = st.date_input("Publication Date")

        if st.button("Validate Paper", type="primary"):
            paper_data = {
                'title': title,
                'abstract': abstract,
                'journal': journal,
                'authors': [a.strip() for a in authors.split(',') if a.strip()],
                'date': date.isoformat()
            }

            result = integration.validate_paper(paper_data)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Quality Score", f"{result.quality_score:.2f}", "/ 1.0")
            with col2:
                st.metric("Title Credibility", f"{result.title_credibility:.2f}")
            with col3:
                st.metric("Abstract Credibility", f"{result.abstract_credibility:.2f}")

            st.divider()

            col1, col2 = st.columns(2)
            with col1:
                if result.is_valid:
                    st.success("✓ Paper is VALID")
                else:
                    st.error("✗ Paper validation FAILED")

            with col2:
                if result.warnings:
                    st.warning("⚠️ Warnings detected")
                    for warning in result.warnings:
                        st.caption(f"• {warning}")

    with tabs[1]:
        st.subheader("Batch Validation")

        uploaded_file = st.file_uploader(
            "Upload CSV file (columns: title, abstract, journal, authors, date)",
            type="csv"
        )

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            if st.button("Validate All Papers", type="primary"):
                papers = df.to_dict('records')
                results = integration.validate_batch(papers)

                st.success(f"✓ Validated {len(results)} papers")

                # Results summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total", len(results))
                with col2:
                    valid = sum(1 for r in results if r.is_valid)
                    st.metric("Valid", valid)
                with col3:
                    invalid = len(results) - valid
                    st.metric("Invalid", invalid)

    with tabs[2]:
        st.subheader("Validation Results")
        report = integration.get_validation_report()

        if report.get('total_papers', 0) == 0:
            st.info("No papers have been validated yet")
        else:
            # Summary cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Papers", report['total_papers'])
            with col2:
                st.metric("Valid Papers", report['valid_papers'])
            with col3:
                st.metric("Suspicious", report['suspicious_papers'])
            with col4:
                st.metric("Avg Quality", f"{report['average_quality_score']:.2f}")

            # Papers with warnings
            if report['papers_with_warnings']:
                st.subheader("Papers with Warnings")
                warning_df = pd.DataFrame([
                    {
                        'Title': p.title[:50] + '...' if len(p.title) > 50 else p.title,
                        'Quality Score': f"{p.quality_score:.2f}",
                        'Warnings': ', '.join(p.warnings[:1])  # First warning
                    }
                    for p in report['papers_with_warnings']
                ])
                st.dataframe(warning_df, use_container_width=True)


# ===== PAGE 3: LOG ANALYSIS =====
elif page == "📋 Log Analysis":
    st.header("Crawler Log Analysis")

    # Input logs
    st.subheader("Input Logs")

    log_text = st.text_area(
        "Paste crawler logs here (one per line, format: [timestamp] LEVEL: message)",
        placeholder="[2024-05-29 10:00:00] ERROR: Connection timeout\n[2024-05-29 10:00:01] INFO: Processing complete",
        height=150
    )

    if st.button("Analyze Logs", type="primary"):
        if log_text.strip():
            logs = [log.strip() for log in log_text.split('\n') if log.strip()]
            analysis = integration.analyze_crawler_logs(logs)

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            summary = analysis['summary']

            with col1:
                st.metric("Total Logs", len(logs))
            with col2:
                st.metric("Anomalies", summary.get('total_anomalies', 0))
            with col3:
                st.metric("High Severity", summary.get('high_severity_count', 0))
            with col4:
                st.metric("Anomaly Rate", f"{summary.get('anomaly_rate', 0):.2%}")

            st.divider()

            # Critical issues
            if analysis['critical_issues']:
                st.subheader("🚨 Critical Issues")
                for issue in analysis['critical_issues']:
                    st.error(
                        f"[{issue.timestamp}] Score: {issue.anomaly_score:.2f} - "
                        f"{issue.log_entry[:80]}..."
                    )
        else:
            st.warning("Please enter some logs to analyze")


# ===== PAGE 4: TRENDS =====
elif page == "📊 Trends":
    st.header("Publication Trends")

    # Generate sample data
    dates = [(datetime.now() - timedelta(days=x)).strftime('%Y-%m-%d') for x in range(30, 0, -1)]
    volumes = [np.random.poisson(lam=15) for _ in range(30)]

    # Add some pattern
    volumes = [v + (i % 7) * 2 for i, v in enumerate(volumes)]

    if st.button("Analyze Trends", type="primary"):
        analysis = integration.analyze_publication_trends(dates, volumes)

        # Trend summary
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Trend Direction", analysis['trend_analysis']['trend'].upper())
        with col2:
            st.metric("Average Volume", f"{analysis['trend_analysis']['mean']:.1f}")
        with col3:
            st.metric("Seasonality", "Detected" if analysis['trend_analysis']['seasonality'] else "None")

        st.divider()

        # Trend visualization
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Publication Volume Over Time")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=volumes, mode='lines+markers', name='Volume'))

            # Highlight anomalies
            if analysis['anomalies']:
                anomaly_dates = [dates[i] for i in analysis['anomalies']]
                anomaly_vols = [volumes[i] for i in analysis['anomalies']]
                fig.add_trace(go.Scatter(
                    x=anomaly_dates, y=anomaly_vols,
                    mode='markers', name='Anomaly',
                    marker=dict(size=10, color='red')
                ))

            fig.update_layout(hovermode='x unified', height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Forecast (Next 7 Days)")
            forecast = analysis['forecast']
            forecast_dates = [(datetime.now() + timedelta(days=x)).strftime('%Y-%m-%d') for x in range(1, 8)]

            fig = go.Figure()
            fig.add_trace(go.Bar(x=forecast_dates, y=forecast, name='Forecasted Volume'))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)


# ===== PAGE 5: SETTINGS =====
elif page == "⚙️ Settings":
    st.header("Settings & Configuration")

    st.subheader("API Configuration")

    col1, col2 = st.columns(2)

    with col1:
        api_host = st.text_input("API Host", value="localhost", disabled=True)
        api_port = st.number_input("API Port", value=8000, disabled=True)

    with col2:
        st.write("")
        st.write("")
        st.info(f"API running at: http://{api_host}:{api_port}")

    st.divider()

    st.subheader("Text Classification Settings")

    col1, col2 = st.columns(2)

    with col1:
        min_credibility = st.slider(
            "Minimum Credibility Score",
            0.0, 1.0, 0.6,
            help="Papers below this score are marked as suspicious"
        )

    with col2:
        st.write("Current threshold:", f"{min_credibility:.2f}")

    st.divider()

    st.subheader("Log Analysis Settings")

    anomaly_threshold = st.slider(
        "Anomaly Detection Threshold",
        0.0, 1.0, 0.7,
        help="Higher = stricter anomaly detection"
    )

    st.divider()

    st.subheader("System Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Papers Processed", len(integration.collected_papers))
    with col2:
        st.metric("Logs Analyzed", len(integration.crawler_logs))
    with col3:
        st.metric("Dashboard Uptime", "24h 30m")

    st.divider()

    st.subheader("Clear Data")

    if st.button("Clear All Data", type="secondary"):
        st.session_state.integration = CrawlerMLIntegration()
        st.success("✓ All data cleared")
        st.rerun()

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: gray; margin-top: 20px;">
    <p>DBpia Crawler ML Dashboard | DSBA Lab Integration</p>
    <p>Built with Streamlit, FastAPI, and PyTorch</p>
</div>
""", unsafe_allow_html=True)
