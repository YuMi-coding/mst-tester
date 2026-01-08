#!/usr/bin/env python3
import csv
import json
import os
import time
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Config
# -----------------------------
POLL_INTERVAL_SEC = 10  # change this
MAX_LOOP = 60
OUTDIR = "./out"
SNAPSHOT_CSV = os.path.join(OUTDIR, "port_counters_snapshot.csv")
DELTA_CSV = os.path.join(OUTDIR, "port_counters_delta.csv")

# If you want to fetch via command, set this.
# Otherwise, leave CMD=None and implement fetch_json() to call your code.
CMD: Optional[List[str]] = None
# Example (adjust to your actual binary + args):
# CMD = ["sudo", "./mstlink", "-d", "0000:01:00.0", "-p", "1", "-c", "--json"]
# /home/cumulus/mstflint-ser-install/bin/mstlink -d 0000:01:00.0 -p 1 -c --json
CMD = ["/home/cumulus/mstflint-ser-install/bin/mstlink", "-d", "0000:01:00.0", "-p", "1", "-c", "--json"]

# -----------------------------
# Helpers: parsing + normalization
# -----------------------------
def utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

def safe_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, int):
        return x
    s = str(x).strip()
    if s == "" or s.upper() == "N/A":
        return None
    # Handle scientific notation that should be int-ish? (Not expected for counters)
    try:
        return int(s, 10)
    except ValueError:
        # sometimes values are floats in strings; try float->int (still risky)
        try:
            f = float(s)
            return int(f)
        except Exception:
            return None

def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, float):
        return x
    if isinstance(x, int):
        return float(x)
    s = str(x).strip()
    if s == "" or s.upper() == "N/A":
        return None
    try:
        return float(s)
    except ValueError:
        # handle things like "15E-255"
        try:
            return float(s.replace("E", "e"))
        except Exception:
            return None

def get_path(d: Dict[str, Any], path: List[str]) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur

def parse_lane_values(node: Any) -> List[Optional[int]]:
    """
    Your JSON uses:
      "Raw Physical Errors Per Lane": {"values": ["1789", ...]}
    """
    if not isinstance(node, dict):
        return []
    vals = node.get("values", [])
    if not isinstance(vals, list):
        return []
    return [safe_int(v) for v in vals]


# -----------------------------
# Data model
# -----------------------------
@dataclass
class Sample:
    ts: float
    time_since_clear_ms: Optional[int]
    time_since_clear_min: Optional[float]

    phy_received_bits: Optional[int]
    phy_corrected_bits: Optional[int]

    eff_phy_errors: Optional[int]
    eff_phy_ber: Optional[float]

    raw_phy_errors_per_lane: List[Optional[int]]
    rs_fec_corr_sym_total: Optional[int]
    rs_fec_corr_blocks: Optional[int]
    rs_fec_uncorr_blocks: Optional[int]
    rs_fec_noerr_blocks: Optional[int]
    rs_fec_corr_sym_per_lane: List[Optional[int]]

    link_down_cnt: Optional[int]
    link_err_recovery_cnt: Optional[int]

    raw_phy_ber_per_lane: List[Optional[float]]
    raw_phy_ber: Optional[float]


def extract_sample(j: Dict[str, Any]) -> Sample:
    base = ["result", "output", "Physical Counters and BER Info"]

    ts = time.time()
    time_ms = safe_int(get_path(j, base + ["Time Since Last Clear [ms]"]))
    time_min = safe_float(get_path(j, base + ["Time Since Last Clear [Min]"]))

    phy_rx_bits = safe_int(get_path(j, base + ["PHY Received Bits"]))
    phy_corr_bits = safe_int(get_path(j, base + ["PHY Corrected Bits"]))

    eff_err = safe_int(get_path(j, base + ["Effective Physical Errors"]))
    eff_ber = safe_float(get_path(j, base + ["Effective Physical BER"]))

    raw_err_lane = parse_lane_values(get_path(j, base + ["Raw Physical Errors Per Lane"]))
    rs_corr_sym_total = safe_int(get_path(j, base + ["RS-FEC Corrected Symbols (Total)"]))
    rs_corr_blocks = safe_int(get_path(j, base + ["RS-FEC Corrected Blocks"]))
    rs_uncorr_blocks = safe_int(get_path(j, base + ["RS-FEC Uncorrectable Blocks"]))
    rs_noerr_blocks = safe_int(get_path(j, base + ["RS-FEC No-Error Blocks"]))
    rs_corr_sym_lane = parse_lane_values(get_path(j, base + ["RS-FEC Corrected Symbols Per Lane"]))

    link_down = safe_int(get_path(j, base + ["Link Down Counter"]))
    link_recov = safe_int(get_path(j, base + ["Link Error Recovery Counter"]))

    raw_ber_lane_node = get_path(j, base + ["Raw Physical BER Per Lane"])
    raw_ber_lane_vals: List[Optional[float]] = []
    if isinstance(raw_ber_lane_node, dict) and isinstance(raw_ber_lane_node.get("values"), list):
        raw_ber_lane_vals = [safe_float(v) for v in raw_ber_lane_node["values"]]

    raw_ber = safe_float(get_path(j, base + ["Raw Physical BER"]))

    return Sample(
        ts=ts,
        time_since_clear_ms=time_ms,
        time_since_clear_min=time_min,
        phy_received_bits=phy_rx_bits,
        phy_corrected_bits=phy_corr_bits,
        eff_phy_errors=eff_err,
        eff_phy_ber=eff_ber,
        raw_phy_errors_per_lane=raw_err_lane,
        rs_fec_corr_sym_total=rs_corr_sym_total,
        rs_fec_corr_blocks=rs_corr_blocks,
        rs_fec_uncorr_blocks=rs_uncorr_blocks,
        rs_fec_noerr_blocks=rs_noerr_blocks,
        rs_fec_corr_sym_per_lane=rs_corr_sym_lane,
        link_down_cnt=link_down,
        link_err_recovery_cnt=link_recov,
        raw_phy_ber_per_lane=raw_ber_lane_vals,
        raw_phy_ber=raw_ber,
    )


# -----------------------------
# Delta computation
# -----------------------------
def delta_int(cur: Optional[int], prev: Optional[int]) -> Optional[int]:
    if cur is None or prev is None:
        return None
    return cur - prev

def delta_lane(cur: List[Optional[int]], prev: List[Optional[int]], lanes: int = 8) -> List[Optional[int]]:
    out: List[Optional[int]] = []
    for i in range(lanes):
        c = cur[i] if i < len(cur) else None
        p = prev[i] if i < len(prev) else None
        out.append(delta_int(c, p))
    return out

def is_monotonic_ok(cur: Optional[int], prev: Optional[int]) -> bool:
    if cur is None or prev is None:
        return True
    return cur >= prev


# -----------------------------
# CSV writing
# -----------------------------
def ensure_outdir() -> None:
    os.makedirs(OUTDIR, exist_ok=True)

def write_row(csv_path: str, header: List[str], row: Dict[str, Any]) -> None:
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(row)

def snapshot_header(lanes: int = 8) -> List[str]:
    h = [
        "ts_utc",
        "time_since_clear_ms",
        "time_since_clear_min",
        "phy_received_bits",
        "phy_corrected_bits",
        "eff_phy_errors",
        "eff_phy_ber",
        "rs_fec_corr_sym_total",
        "rs_fec_corr_blocks",
        "rs_fec_uncorr_blocks",
        "rs_fec_noerr_blocks",
        "link_down_cnt",
        "link_err_recovery_cnt",
        "raw_phy_ber",
    ]
    h += [f"raw_phy_errors_lane_{i}" for i in range(lanes)]
    h += [f"rs_fec_corr_sym_lane_{i}" for i in range(lanes)]
    h += [f"raw_phy_ber_lane_{i}" for i in range(lanes)]
    return h

def delta_header(lanes: int = 8) -> List[str]:
    h = [
        "ts_utc",
        "dt_sec",
        "d_time_since_clear_ms",
        "d_phy_received_bits",
        "d_phy_corrected_bits",
        "d_eff_phy_errors",
        "d_rs_fec_corr_sym_total",
        "d_rs_fec_corr_blocks",
        "d_rs_fec_uncorr_blocks",
        "d_rs_fec_noerr_blocks",
        "d_link_down_cnt",
        "d_link_err_recovery_cnt",
    ]
    h += [f"d_raw_phy_errors_lane_{i}" for i in range(lanes)]
    h += [f"d_rs_fec_corr_sym_lane_{i}" for i in range(lanes)]
    return h


# -----------------------------
# Fetch JSON
# -----------------------------
def fetch_json_via_cmd(cmd: List[str]) -> Dict[str, Any]:
    # Assumes the command prints JSON to stdout.
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"command failed rc={p.returncode}: {p.stderr.strip()}")
    return json.loads(p.stdout)

def fetch_json() -> Dict[str, Any]:
    """
    Replace this if you already have an in-process function returning the json dict.
    """
    if CMD is None:
        raise RuntimeError("CMD is None. Set CMD or implement fetch_json() to call your existing code.")
    return fetch_json_via_cmd(CMD)


# -----------------------------
# Main loop
# -----------------------------
def sample_to_snapshot_row(s: Sample, lanes: int = 8) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "ts_utc": utc_iso(s.ts),
        "time_since_clear_ms": s.time_since_clear_ms,
        "time_since_clear_min": s.time_since_clear_min,
        "phy_received_bits": s.phy_received_bits,
        "phy_corrected_bits": s.phy_corrected_bits,
        "eff_phy_errors": s.eff_phy_errors,
        "eff_phy_ber": s.eff_phy_ber,
        "rs_fec_corr_sym_total": s.rs_fec_corr_sym_total,
        "rs_fec_corr_blocks": s.rs_fec_corr_blocks,
        "rs_fec_uncorr_blocks": s.rs_fec_uncorr_blocks,
        "rs_fec_noerr_blocks": s.rs_fec_noerr_blocks,
        "link_down_cnt": s.link_down_cnt,
        "link_err_recovery_cnt": s.link_err_recovery_cnt,
        "raw_phy_ber": s.raw_phy_ber,
    }

    for i in range(lanes):
        row[f"raw_phy_errors_lane_{i}"] = s.raw_phy_errors_per_lane[i] if i < len(s.raw_phy_errors_per_lane) else None
        row[f"rs_fec_corr_sym_lane_{i}"] = s.rs_fec_corr_sym_per_lane[i] if i < len(s.rs_fec_corr_sym_per_lane) else None
        row[f"raw_phy_ber_lane_{i}"] = s.raw_phy_ber_per_lane[i] if i < len(s.raw_phy_ber_per_lane) else None

    return row

def sample_pair_to_delta_row(cur: Sample, prev: Sample, lanes: int = 8) -> Dict[str, Any]:
    dt = cur.ts - prev.ts
    d_time_ms = delta_int(cur.time_since_clear_ms, prev.time_since_clear_ms)

    d_raw_err_lane = delta_lane(cur.raw_phy_errors_per_lane, prev.raw_phy_errors_per_lane, lanes=lanes)
    d_rs_sym_lane = delta_lane(cur.rs_fec_corr_sym_per_lane, prev.rs_fec_corr_sym_per_lane, lanes=lanes)

    row: Dict[str, Any] = {
        "ts_utc": utc_iso(cur.ts),
        "dt_sec": round(dt, 3),
        "d_time_since_clear_ms": d_time_ms,
        "d_phy_received_bits": delta_int(cur.phy_received_bits, prev.phy_received_bits),
        "d_phy_corrected_bits": delta_int(cur.phy_corrected_bits, prev.phy_corrected_bits),
        "d_eff_phy_errors": delta_int(cur.eff_phy_errors, prev.eff_phy_errors),
        "d_rs_fec_corr_sym_total": delta_int(cur.rs_fec_corr_sym_total, prev.rs_fec_corr_sym_total),
        "d_rs_fec_corr_blocks": delta_int(cur.rs_fec_corr_blocks, prev.rs_fec_corr_blocks),
        "d_rs_fec_uncorr_blocks": delta_int(cur.rs_fec_uncorr_blocks, prev.rs_fec_uncorr_blocks),
        "d_rs_fec_noerr_blocks": delta_int(cur.rs_fec_noerr_blocks, prev.rs_fec_noerr_blocks),
        "d_link_down_cnt": delta_int(cur.link_down_cnt, prev.link_down_cnt),
        "d_link_err_recovery_cnt": delta_int(cur.link_err_recovery_cnt, prev.link_err_recovery_cnt),
    }

    for i in range(lanes):
        row[f"d_raw_phy_errors_lane_{i}"] = d_raw_err_lane[i]
        row[f"d_rs_fec_corr_sym_lane_{i}"] = d_rs_sym_lane[i]

    return row

def print_delta_summary(delta_row: Dict[str, Any], lanes: int = 8) -> None:
    # Customize this to your taste; keep it compact so it’s readable in a live console.
    ts = delta_row["ts_utc"]
    dt = delta_row["dt_sec"]
    d_rx = delta_row.get("d_phy_received_bits")
    d_corr = delta_row.get("d_phy_corrected_bits")
    d_sym = delta_row.get("d_rs_fec_corr_sym_total")
    d_raw_lane = [delta_row.get(f"d_raw_phy_errors_lane_{i}") for i in range(lanes)]

    print(
        f"[{ts}] dt={dt}s "
        f"d_rx_bits={d_rx} d_corr_bits={d_corr} d_rs_corr_sym={d_sym} "
        f"d_raw_err_lane={d_raw_lane}"
    )

def main() -> None:
    ensure_outdir()
    lanes = 8

    prev: Optional[Sample] = None
    loop_count = 0

    while True:
        try:
            j = fetch_json()
            cur = extract_sample(j)

            # Always write snapshot
            write_row(SNAPSHOT_CSV, snapshot_header(lanes), sample_to_snapshot_row(cur, lanes))

            # If we have a previous sample, compute delta
            if prev is not None:
                # Basic sanity: if counters reset, skip delta (but keep snapshot)
                # You can extend this to detect link flap / clear event etc.
                if (
                    is_monotonic_ok(cur.time_since_clear_ms, prev.time_since_clear_ms)
                    and is_monotonic_ok(cur.phy_received_bits, prev.phy_received_bits)
                    and is_monotonic_ok(cur.rs_fec_noerr_blocks, prev.rs_fec_noerr_blocks)
                ):
                    drow = sample_pair_to_delta_row(cur, prev, lanes)
                    write_row(DELTA_CSV, delta_header(lanes), drow)
                    print_delta_summary(drow, lanes=lanes)
                else:
                    print(f"[{utc_iso(cur.ts)}] counters look reset/non-monotonic; skipping delta")

            prev = cur

        except Exception as e:
            print(f"[{utc_iso(time.time())}] ERROR: {e}")

        loop_count += 1
        if loop_count >= MAX_LOOP:
            print(f"Reached MAX_LOOP={MAX_LOOP}, exiting.")
            break
        time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main()
