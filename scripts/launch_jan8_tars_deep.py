#!/Users/luke/local/miniconda3/envs/py311_sx/bin/python
"""Launch parallel jobs for the Jan 8 TARS deep-field catalog.

This launcher throttles to a max concurrency, sleeps between submissions,
and writes per-target logs plus a summary CSV.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, List, Optional, TextIO

import numpy as np
import pandas as pd

from sx_phot.tic_motion import normalize_tic_id


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = (
    ROOT
    / "sx_phot"
    / "data"
    / "targetlists"
    / "jan8_default_catalog_no_quality_flags_spherex_obs_deep.csv"
)
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "jan8_tars_deep"
DEFAULT_RUNNER = ROOT / "scripts" / "run_ticid_with_period.py"
DEFAULT_MAX_PARALLEL = 50
DEFAULT_LAUNCH_DELAY = 10.0
SUMMARY_NAME = "launcher_summary.csv"


def _log(message: str) -> None:
    """Log a timestamped message.

    Args:
        message: Message to print.
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch parallel runs for the Jan 8 TARS deep-field list."
    )
    parser.add_argument(
        "--input-csv",
        default=str(DEFAULT_INPUT_CSV),
        help="Input catalog CSV path.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root output directory for per-TIC products.",
    )
    parser.add_argument(
        "--runner",
        default=str(DEFAULT_RUNNER),
        help="Path to the single-target runner script.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for each job.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=DEFAULT_MAX_PARALLEL,
        help="Maximum number of concurrent jobs (default: 50).",
    )
    parser.add_argument(
        "--launch-delay",
        type=float,
        default=DEFAULT_LAUNCH_DELAY,
        help="Delay in seconds between job submissions (default: 10).",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=None,
        help="Optional maximum number of targets to process.",
    )
    parser.add_argument(
        "--ylim",
        default=None,
        help="Comma-separated y-limits for residuals (default in runner).",
    )
    parser.add_argument(
        "--do-photometry",
        action="store_true",
        help="Force recomputation of photometry even if caches exist.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip targets with a done marker or cached supplemented CSV.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Disable skipping of existing targets.",
    )
    parser.set_defaults(skip_existing=True)
    return parser.parse_args()


def _init_summary(path: Path) -> None:
    """Create the summary CSV with header if needed."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "tic_id",
                "period_days",
                "status",
                "return_code",
                "start_time",
                "end_time",
                "log_path",
                "target_dir",
            ]
        )


def _append_summary(path: Path, row: List[str]) -> None:
    """Append one row to the summary CSV."""
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(row)


def _should_skip(target_dir: Path) -> bool:
    """Return True if a target should be skipped based on cached outputs."""
    done_marker = target_dir / ".done"
    if done_marker.exists():
        return True
    cached = list(target_dir.glob("sxphot_cache_TIC_*_splsupp.csv"))
    return bool(cached)


def _launch_job(
    python_exec: str,
    runner: Path,
    tic_id: int,
    period_days: float,
    output_root: Path,
    ylim: Optional[str],
    do_photometry: bool,
    log_path: Path,
) -> tuple[subprocess.Popen, TextIO]:
    """Launch a single target job and return the process and log handle."""
    cmd = [
        python_exec,
        str(runner),
        "--tic-id",
        str(tic_id),
        "--period-days",
        str(period_days),
        "--output-root",
        str(output_root),
    ]
    if ylim:
        cmd.extend(["--ylim", ylim])
    if do_photometry:
        cmd.append("--do-photometry")
    cmd.append("--skip-existing")

    log_handle = log_path.open("a", encoding="utf-8")
    log_handle.write(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Launching: "
        f"{' '.join(cmd)}\n"
    )
    log_handle.flush()
    proc = subprocess.Popen(
        cmd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    return proc, log_handle


def main() -> None:
    """Launch parallel jobs for the Jan 8 TARS deep list."""
    args = _parse_args()
    input_csv = Path(args.input_csv)
    output_root = Path(args.output_root)
    runner = Path(args.runner)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if not runner.exists():
        raise FileNotFoundError(f"Runner script not found: {runner}")

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / SUMMARY_NAME
    _init_summary(summary_path)

    df = pd.read_csv(input_csv)
    required = {"TICID", "adopted_period"}
    missing = sorted(col for col in required if col not in df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    rows = df.to_dict(orient="records")
    if args.max_targets is not None:
        rows = rows[: args.max_targets]

    _log(f"Launching {len(rows)} targets from {input_csv}")

    running: List[Dict[str, object]] = []
    idx = 0
    while idx < len(rows) or running:
        # Launch new jobs if slots are available.
        while idx < len(rows) and len(running) < args.max_parallel:
            row = rows[idx]
            idx += 1

            try:
                tic_id = normalize_tic_id(row["TICID"])
            except Exception as exc:
                _log(f"Skipping invalid TIC: {exc}")
                _append_summary(
                    summary_path,
                    [
                        str(row.get("TICID", "")),
                        str(row.get("adopted_period", "")),
                        "invalid_tic",
                        "",
                        "",
                        "",
                        "",
                        "",
                    ],
                )
                continue

            period_days = float(row["adopted_period"])
            if not np.isfinite(period_days) or period_days <= 0:
                _log(f"TIC_{tic_id}: invalid period {period_days}")
                _append_summary(
                    summary_path,
                    [
                        str(tic_id),
                        str(period_days),
                        "invalid_period",
                        "",
                        "",
                        "",
                        "",
                        "",
                    ],
                )
                continue

            star_id = f"TIC_{tic_id}"
            target_dir = output_root / star_id
            target_dir.mkdir(parents=True, exist_ok=True)
            log_path = target_dir / f"run_{star_id}.log"

            if args.skip_existing and _should_skip(target_dir):
                _log(f"{star_id}: skipping existing outputs")
                _append_summary(
                    summary_path,
                    [
                        str(tic_id),
                        str(period_days),
                        "skipped",
                        "",
                        "",
                        "",
                        str(log_path),
                        str(target_dir),
                    ],
                )
                continue

            proc, log_handle = _launch_job(
                args.python,
                runner,
                tic_id,
                period_days,
                output_root,
                args.ylim,
                args.do_photometry,
                log_path,
            )
            running.append(
                {
                    "proc": proc,
                    "tic_id": tic_id,
                    "period_days": period_days,
                    "log_path": log_path,
                    "log_handle": log_handle,
                    "target_dir": target_dir,
                    "start_time": datetime.now(),
                }
            )
            _log(f"{star_id}: launched ({len(running)}/{args.max_parallel})")
            time.sleep(args.launch_delay)

        # Reap finished jobs.
        still_running: List[Dict[str, object]] = []
        for entry in running:
            proc = entry["proc"]
            retcode = proc.poll()
            if retcode is None:
                still_running.append(entry)
                continue

            end_time = datetime.now()
            status = "ok" if retcode == 0 else "failed"
            entry["log_handle"].close()
            _append_summary(
                summary_path,
                [
                    str(entry["tic_id"]),
                    str(entry["period_days"]),
                    status,
                    str(retcode),
                    entry["start_time"].strftime("%Y-%m-%d %H:%M:%S"),
                    end_time.strftime("%Y-%m-%d %H:%M:%S"),
                    str(entry["log_path"]),
                    str(entry["target_dir"]),
                ],
            )
            _log(f"TIC_{entry['tic_id']}: finished ({status})")

        running = still_running
        if running:
            time.sleep(5)

    _log("All jobs completed.")


if __name__ == "__main__":
    main()
