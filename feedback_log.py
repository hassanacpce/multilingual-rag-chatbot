"""
feedback_log.py

Logs per-turn sentiment + explicit user feedback (thumbs up/down) to a local
CSV, and computes aggregate stats from it. This directly supports the
assignment's third evaluation criterion -- "impact on customer satisfaction"
-- which can't be measured from sentiment detection accuracy alone; it needs
an actual feedback signal correlated with what sentiment was detected and
whether the tone-adjusted response landed well.

Deliberately simple (CSV, not a database) and fully local -- no new
external dependency, consistent with the project's existing pattern of
deterministic components (sentiment analysis, retrieval, extraction) being
testable offline.
"""

import csv
import os
from datetime import datetime, timezone
from typing import Optional

LOG_PATH = "feedback_log.csv"
FIELDNAMES = ["timestamp", "question", "sentiment_label", "sentiment_compound", "response", "feedback"]


def log_feedback(
    question: str,
    sentiment_label: str,
    sentiment_compound: float,
    response: str,
    feedback: str,  # "up" | "down"
    log_path: str = LOG_PATH,
) -> None:
    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "question": question,
                "sentiment_label": sentiment_label,
                "sentiment_compound": sentiment_compound,
                "response": response[:500],  # keep the log readable/bounded
                "feedback": feedback,
            }
        )


def get_feedback_stats(log_path: str = LOG_PATH) -> Optional[dict]:
    """
    Returns aggregate satisfaction stats, or None if no feedback logged yet.

    Breaks down thumbs-up rate BY detected sentiment label -- this is the
    number that actually answers "does the sentiment-aware tone adjustment
    help": if thumbs-up rate on negative-sentiment turns is meaningfully
    lower than on neutral/positive turns, that's a real signal the tone
    handling needs work, not just an aggregate "X% satisfied" number that
    hides where the problem is.
    """
    if not os.path.exists(log_path):
        return None

    rows = []
    with open(log_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None

    total = len(rows)
    up = sum(1 for r in rows if r["feedback"] == "up")

    by_sentiment = {}
    for r in rows:
        label = r["sentiment_label"]
        by_sentiment.setdefault(label, {"total": 0, "up": 0})
        by_sentiment[label]["total"] += 1
        if r["feedback"] == "up":
            by_sentiment[label]["up"] += 1

    for label, counts in by_sentiment.items():
        counts["satisfaction_rate"] = counts["up"] / counts["total"] if counts["total"] else 0.0

    return {
        "total": total,
        "up": up,
        "down": total - up,
        "overall_satisfaction_rate": up / total if total else 0.0,
        "by_sentiment": by_sentiment,
    }


if __name__ == "__main__":
    # Quick self-test with synthetic feedback events
    test_log = "test_feedback_log.csv"
    if os.path.exists(test_log):
        os.remove(test_log)

    log_feedback("This is terrible!", "strongly_negative", -0.85, "I'm sorry for the trouble...", "down", log_path=test_log)
    log_feedback("This is terrible!", "strongly_negative", -0.85, "I'm sorry for the trouble...", "up", log_path=test_log)
    log_feedback("What are your hours?", "neutral", 0.0, "We're open 9-5.", "up", log_path=test_log)
    log_feedback("Thanks!", "positive", 0.6, "You're welcome!", "up", log_path=test_log)

    stats = get_feedback_stats(log_path=test_log)
    print("Overall:", stats["total"], "total,", f"{stats['overall_satisfaction_rate']*100:.0f}% satisfaction")
    for label, counts in stats["by_sentiment"].items():
        print(f"  {label:17s}: {counts['up']}/{counts['total']} = {counts['satisfaction_rate']*100:.0f}%")

    os.remove(test_log)
