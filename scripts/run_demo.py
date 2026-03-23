#!/usr/bin/env python3
"""
Interactive 3-Month Skincare Journey Demo.

Runs the real CV pipeline (YOLO acne + wrinkle UNet) on 3 face images,
saves analysis results to the database with backdated timestamps, gets
agent product recommendations, auto-records purchases, and ends with a
free-chat loop so you can ask the agent anything about the journey.

Usage:
    cd /home/vanessa/project
    test-venv/bin/python scripts/run_demo.py
"""

import asyncio
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from test_all import run_inference, CONCERNS
import db.crud as crud
from db.database import get_db, init_db
from agent.graph import invoke_agent

DEMO_IMAGES = [
    PROJECT_ROOT / "agentictest" / "1.jpg",
    PROJECT_ROOT / "agentictest" / "2.PNG",
    PROJECT_ROOT / "agentictest" / "3.jpg",
]

MONTH_LABELS = ["Month 1", "Month 2", "Month 3"]

BACKDATE_DAYS = [60, 30, 0]


class InferenceArgs:
    """Mirrors the Args class in test_all.py but with configurable paths."""

    def __init__(self, image_path: str, skin_type: str, output_dir: Path, month: int):
        self.image = str(image_path)
        self.detection_model = str(
            PROJECT_ROOT / "acne_yolo_runs/roboflow_6classes/weights/best.pt"
        )
        self.severity_model = "/home/vanessa/acne_severity_runs/20251110_172153/best_model.pt"
        self.bisenet_model = str(PROJECT_ROOT / "79999_iter.pth")
        self.wrinkle_checkpoint = str(
            PROJECT_ROOT
            / "ffhq_wrinkle_data/pretrained_ckpt/stage2_wrinkle_finetune_unet/stage2_unet.pth"
        )
        self.wrinkle_network = "UNet"
        self.conf_thres = 0.1
        self.iou_thres = 0.5
        self.skin_type = skin_type
        self.budget = None
        self.rec_top_n = 5
        self.output = str(output_dir / f"month{month}_report.json")
        self.save_acne_vis = str(output_dir / f"month{month}_acne.png")
        self.save_wrinkle_vis = str(output_dir / f"month{month}_wrinkle.png")
        self.save_rec_vis = str(output_dir / f"month{month}_recommendations.png")


def _backdate(table: str, col: str, row_id: int, days_ago: int):
    """Update a timestamp column to N days in the past."""
    ts = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db()
    conn.execute(f"UPDATE {table} SET {col} = ? WHERE id = ?", (ts, row_id))
    conn.commit()
    conn.close()


def _backdate_user(user_id: str, days_ago: int):
    ts = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db()
    conn.execute("UPDATE users SET created_at = ? WHERE user_id = ?", (ts, user_id))
    conn.commit()
    conn.close()


def cleanup_demo_user(user_id: str):
    """Remove all data for a demo user so reruns are clean."""
    conn = get_db()
    conn.execute("DELETE FROM purchases WHERE user_id = ?", (user_id,))
    conn.execute("DELETE FROM recommendations WHERE user_id = ?", (user_id,))
    conn.execute("DELETE FROM analyses WHERE user_id = ?", (user_id,))
    conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()


def print_concern_vector(concern_vector: list[float]):
    """Pretty-print the 7-dim concern vector."""
    for name, val in zip(CONCERNS, concern_vector):
        bar = "\u2588" * int(val * 30)
        print(f"    {name:22s} {val:.3f}  {bar}")


def run_cv_pipeline(image_path: Path, skin_type: str, output_dir: Path, month: int) -> dict:
    """Run the full CV pipeline on one image, return the JSON report."""
    args = InferenceArgs(image_path, skin_type, output_dir, month)
    run_inference(args)
    with open(args.output) as f:
        return json.load(f)


def extract_db_fields(report: dict) -> tuple[list[float], dict, dict]:
    """Extract concern_vector, acne_summary, wrinkle_summary from a report."""
    concern_vector = [report["user_concern_vector"].get(c, 0.0) for c in CONCERNS]

    acne = report["acne"]
    acne_summary = {
        "total_detections": acne["total_detections"],
        "regions": acne["regions"],
    }

    wrinkle = report["wrinkle"]
    wrinkle_summary = {
        "severity": wrinkle["severity"],
        "severity_score": wrinkle["severity_score"],
        "wrinkle_pct": wrinkle["wrinkle_pct"],
    }

    return concern_vector, acne_summary, wrinkle_summary


def record_purchases(
    user_id: str, recommendations: dict, days_ago: int
) -> list[str]:
    """Save recommended products as purchases, return product titles."""
    titles = []
    for _cat, items in recommendations.items():
        for item in items[:2]:
            title = item.get("full_name", item.get("title", "Unknown"))
            pid = crud.save_purchase(
                user_id=user_id,
                product_url=item.get("product_url", ""),
                product_title=title,
                price=item.get("price_value"),
            )
            purchase_days = max(0, days_ago - 5)
            _backdate("purchases", "purchased_at", pid, purchase_days)
            titles.append(title)
    return titles


async def run_demo():
    print("\n" + "=" * 70)
    print("     INTERACTIVE DEMO: 3-Month Skincare Journey")
    print("     Real CV Analysis  |  LLM Agent  |  Product Recommendations")
    print("=" * 70)

    name = input("\n  Enter your name: ").strip() or "Demo User"
    print("  Skin types: oily, dry, sensitive, normal, combination")
    skin_type = input("  Enter your skin type: ").strip().lower() or "oily"

    user_id = f"demo_{name.lower().replace(' ', '_')}"

    init_db()
    cleanup_demo_user(user_id)
    crud.create_user(user_id, name=name, skin_type=skin_type)
    _backdate_user(user_id, BACKDATE_DAYS[0])

    output_dir = PROJECT_ROOT / "demo_output"
    output_dir.mkdir(exist_ok=True)

    print(f"\n  User ID:   {user_id}")
    print(f"  Skin type: {skin_type}")
    print(f"  Images:    {', '.join(p.name for p in DEMO_IMAGES)}")
    print(f"  Output:    {output_dir}/")

    reports: list[dict] = []

    for month_idx in range(3):
        month_num = month_idx + 1
        image_path = DEMO_IMAGES[month_idx]
        label = MONTH_LABELS[month_idx]
        days_ago = BACKDATE_DAYS[month_idx]

        print(f"\n\n{'=' * 70}")
        print(f"  {label} -- Analyzing {image_path.name}")
        print(f"{'=' * 70}\n")

        report = run_cv_pipeline(image_path, skin_type, output_dir, month_num)
        reports.append(report)

        concern_vector, acne_summary, wrinkle_summary = extract_db_fields(report)

        analysis_id = crud.save_analysis(
            user_id=user_id,
            image_path=str(image_path),
            concern_vector=concern_vector,
            acne_summary=acne_summary,
            wrinkle_summary=wrinkle_summary,
            full_report=report,
        )
        _backdate("analyses", "created_at", analysis_id, days_ago)

        print(f"\n  Analysis saved (id={analysis_id})")
        print(f"  Acne detections: {acne_summary['total_detections']}")
        print(f"  Wrinkle severity: {wrinkle_summary['severity']} "
              f"({wrinkle_summary['severity_score']}/3)")
        print(f"\n  Concern vector:")
        print_concern_vector(concern_vector)

        cv_str = ", ".join(f"{c}={v:.2f}" for c, v in zip(CONCERNS, concern_vector))

        if month_num == 1:
            prompt = (
                f"I just did my first skin analysis. My user ID is {user_id}. "
                f"My skin type is {skin_type}. "
                f"My concern scores are: {cv_str}. "
                f"I had {acne_summary['total_detections']} acne detections "
                f"and wrinkle severity is {wrinkle_summary['severity']}. "
                f"Please recommend the top 5 products for my skin concerns."
            )
        elif month_num == 2:
            prompt = (
                f"I just completed my second monthly skin analysis. "
                f"My user ID is {user_id}. "
                f"Please compare my latest analysis with my previous one to show "
                f"how my skin has changed. Then recommend products for my current needs."
            )
        else:
            prompt = (
                f"I just completed my third monthly skin analysis. "
                f"My user ID is {user_id}. "
                f"Please compare my skin progress over the past 3 months and give "
                f"me a full assessment. What has improved? What should I focus on next? "
                f"Also recommend any products that could help."
            )

        print(f"\n{'~' * 70}")
        print(f"  Agent Response ({label}):")
        print(f"{'~' * 70}")
        response = await invoke_agent(user_id=user_id, message=prompt)
        print(f"\n{response}")

        if month_num < 3:
            recs = report.get("recommendations", {})
            titles = record_purchases(user_id, recs, days_ago)
            if titles:
                print(f"\n  [{len(titles)} products auto-purchased:]")
                for t in titles:
                    print(f"    - {t}")

            input(f"\n  Press Enter to continue to {MONTH_LABELS[month_idx + 1]}...")

    # ── Free chat loop ───────────────────────────────────────────────────
    print(f"\n\n{'=' * 70}")
    print("  FREE CHAT -- Ask anything about your skincare journey")
    print("  Examples:")
    print(f'    "Show me my full profile"')
    print(f'    "How has my acne changed over 3 months?"')
    print(f'    "What products should I focus on now?"')
    print(f'    "Has anything gotten worse?"')
    print("  Type 'quit' to exit")
    print(f"{'=' * 70}")

    while True:
        try:
            user_input = input("\n  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue
        response = await invoke_agent(user_id=user_id, message=user_input)
        print(f"\n  Agent: {response}")

    print(f"\n  Demo complete!")
    print(f"  Visualizations saved to: {output_dir}/")
    print(f"  Database: {PROJECT_ROOT / 'data' / 'skincare.db'}")


if __name__ == "__main__":
    asyncio.run(run_demo())
