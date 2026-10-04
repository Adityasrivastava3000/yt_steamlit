import logging
from prometheus_client import Gauge, Counter

logger = logging.getLogger("services.metrics_service")

# ------------------------------------------------------------------
# Metric Definitions
# ------------------------------------------------------------------

# Gauges to track current jobs status counts
JOBS_TOTAL = Gauge(
    "lecture_extractor_jobs_total",
    "Total number of slide extraction jobs tracked by status",
    ["status"]
)

# Counter to count total failures
FAILURES_TOTAL = Counter(
    "lecture_extractor_failures_total",
    "Total number of failed slide extraction jobs"
)

# Gauges to track average execution times in seconds (for completed jobs)
PROCESSING_TIME = Gauge(
    "lecture_extractor_processing_time_seconds",
    "Average total processing time of completed jobs in seconds"
)

OCR_TIME = Gauge(
    "lecture_extractor_ocr_time_seconds",
    "Average OCR processing duration of completed jobs in seconds"
)

PPT_TIME = Gauge(
    "lecture_extractor_ppt_time_seconds",
    "Average PowerPoint generation duration of completed jobs in seconds"
)

PDF_TIME = Gauge(
    "lecture_extractor_pdf_time_seconds",
    "Average PDF conversion duration of completed jobs in seconds"
)

# Helper to initialize status labels
for status in ["pending", "running", "completed", "failed", "cancelled"]:
    JOBS_TOTAL.labels(status=status).set(0)

# ------------------------------------------------------------------
# Metrics Update Function
# ------------------------------------------------------------------

def update_metrics():
    """Query Redis to aggregate jobs metrics and update Prometheus values."""
    try:
        from services.job_manager import JobManager
        jm = JobManager()
        
        # 1. Fetch all jobs
        jobs = jm.list_jobs()
        
        # Reset counters map
        counts = {"pending": 0, "running": 0, "completed": 0, "failed": 0, "cancelled": 0}
        
        for job in jobs:
            status = job.get("status", "unknown").lower()
            if status in counts:
                counts[status] += 1
                
        # Update JOBS_TOTAL gauge labels
        for status, count in counts.items():
            JOBS_TOTAL.labels(status=status).set(count)
            
        # Update FAILURES_TOTAL counter (Prometheus counter increases, we set value directly here)
        # Note: In prometheus_client Counter, setting value directly is done via private _value.set()
        FAILURES_TOTAL._value.set(counts["failed"])
        
        # 2. Compute average step durations for completed jobs
        completed_jobs = [j for j in jobs if j.get("status") == "completed"]
        
        if completed_jobs:
            total_duration = 0.0
            total_ocr = 0.0
            total_ppt = 0.0
            total_pdf = 0.0
            
            ocr_count = 0
            ppt_count = 0
            pdf_count = 0
            duration_count = 0

            for j in completed_jobs:
                key = f"job:{j['job_id']}"
                hdata = jm.r.hgetall(key)
                
                # Extract and parse values
                td = hdata.get("total_duration_seconds")
                od = hdata.get("ocr_duration_seconds")
                pt = hdata.get("ppt_duration_seconds")
                pd = hdata.get("pdf_duration_seconds")
                
                if td:
                    total_duration += float(td)
                    duration_count += 1
                if od:
                    total_ocr += float(od)
                    ocr_count += 1
                if pt:
                    total_ppt += float(pt)
                    ppt_count += 1
                if pd:
                    total_pdf += float(pd)
                    pdf_count += 1
            
            # Set Gauges to average durations
            if duration_count > 0:
                PROCESSING_TIME.set(round(total_duration / duration_count, 2))
            if ocr_count > 0:
                OCR_TIME.set(round(total_ocr / ocr_count, 2))
            if ppt_count > 0:
                PPT_TIME.set(round(total_ppt / ppt_count, 2))
            if pdf_count > 0:
                PDF_TIME.set(round(total_pdf / pdf_count, 2))
        else:
            # Set to 0 if no completed jobs exist
            PROCESSING_TIME.set(0)
            OCR_TIME.set(0)
            PPT_TIME.set(0)
            PDF_TIME.set(0)

    except Exception as e:
        logger.error(f"Failed to update Prometheus metrics: {e}")
