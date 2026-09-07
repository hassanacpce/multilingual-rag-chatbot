"""
scheduler.py

Runs knowledge_base_manager.update_vector_store() on a periodic interval.

Two ways to use it:

1. Standalone process (recommended for production):
       python scheduler.py
   Keeps running and updates the knowledge base in the background on the
   interval configured in sources_config.json, independent of whether the
   Streamlit app is open.

2. In-process, started from the Streamlit app (start_scheduler()) so a
   single running `streamlit run main.py` process also keeps itself fresh.
   start_scheduler() is idempotent -- safe to call on every Streamlit rerun.
"""

import time
import threading

from apscheduler.schedulers.background import BackgroundScheduler

from knowledge_base_manager import load_config, update_vector_store

_scheduler = None
_lock = threading.Lock()
_JOB_ID = "kb_update_job"


def start_scheduler():
    """Start the background job if it isn't already running. Idempotent."""
    global _scheduler
    with _lock:
        if _scheduler is not None and _scheduler.running:
            return _scheduler

        config = load_config()
        interval = config.get("schedule", {}).get("interval_minutes", 60)

        _scheduler = BackgroundScheduler()
        _scheduler.add_job(
            update_vector_store,
            "interval",
            minutes=interval,
            id=_JOB_ID,
            replace_existing=True,
        )
        _scheduler.start()
        return _scheduler


def stop_scheduler():
    global _scheduler
    with _lock:
        if _scheduler is not None:
            _scheduler.shutdown(wait=False)
            _scheduler = None


def reschedule(interval_minutes: int):
    """Call after changing the interval so the running job picks it up."""
    global _scheduler
    with _lock:
        if _scheduler is not None and _scheduler.running:
            _scheduler.reschedule_job(
                _JOB_ID, trigger="interval", minutes=interval_minutes
            )


def is_running():
    return _scheduler is not None and _scheduler.running


if __name__ == "__main__":
    sched = start_scheduler()
    cfg = load_config()
    mins = cfg.get("schedule", {}).get("interval_minutes", 60)
    print(f"Knowledge base scheduler started, updating every {mins} minute(s).")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_scheduler()
        print("Scheduler stopped.")
