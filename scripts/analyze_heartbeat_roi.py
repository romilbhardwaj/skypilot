#!/usr/bin/env python3
"""Analyze SkyPilot server heartbeat telemetry for GPU ROI metrics.

Fetches heartbeat data from Loki (via Grafana proxy), computes per-server
GPU allocation stats and Kueue borrowing ROI, and optionally generates
matplotlib plots and CSV reports.

Usage:
    python scripts/analyze_heartbeat_roi.py \
        --grafana-url http://34.212.129.117:3225 \
        --grafana-user admin --grafana-password skypilot-dev \
        --range 7d --plot-dir ./roi_plots --csv-output report.csv
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class GPUPriceTable:
    """GPU pricing for ROI calculations."""

    DEFAULT = {
        'H100': 3.50,
        'H200': 4.50,
        'A100': 2.50,
        'A100-80GB': 2.50,
        'L40S': 1.50,
        'L40': 1.50,
        'L4': 0.75,
        'T4': 0.35,
        'V100': 0.80,
    }

    def __init__(self, overrides: Optional[Dict[str, float]] = None):
        self.prices = dict(self.DEFAULT)
        if overrides:
            self.prices.update(overrides)

    def get_price(self, gpu_type: str) -> float:
        """Get price per GPU-hour. Falls back to H100 price if unknown."""
        return self.prices.get(gpu_type, self.prices.get('H100', 3.50))


class LokiClient:
    """Client for querying Loki via Grafana proxy."""

    QUERY_PATH = '/api/datasources/proxy/1/loki/api/v1/query_range'

    def __init__(self, grafana_url: str, user: str, password: str):
        self.grafana_url = grafana_url.rstrip('/')
        self.auth = (user, password)
        self.session = requests.Session()
        self.session.auth = self.auth

    def query_heartbeats(self,
                         range_str: str,
                         server_hash: Optional[str] = None,
                         limit: int = 5000) -> List[dict]:
        """Query heartbeat logs from Loki.

        Args:
            range_str: Time range like '1h', '24h', '7d', '30d'.
            server_hash: Optional filter for specific server.
            limit: Max log lines per request.

        Returns:
            List of parsed heartbeat JSON objects, sorted by timestamp.
        """
        end_ns = int(time.time() * 1e9)
        duration_seconds = _parse_range(range_str)
        start_ns = end_ns - int(duration_seconds * 1e9)

        query = '{type="server_heartbeat"}'

        all_heartbeats = []
        current_end = end_ns

        while current_end > start_ns:
            params = {
                'query': query,
                'start': str(start_ns),
                'end': str(current_end),
                'limit': str(limit),
                'direction': 'backward',
            }

            url = f'{self.grafana_url}{self.QUERY_PATH}'
            resp = self.session.get(url, params=params, timeout=60)
            resp.raise_for_status()

            data = resp.json()
            if data.get('status') != 'success':
                raise RuntimeError(
                    f'Loki query failed: {data.get("error", "unknown")}')

            results = data.get('data', {}).get('result', [])
            if not results:
                break

            batch = []
            min_ts = current_end
            for stream in results:
                for ts_ns, line in stream.get('values', []):
                    ts = int(ts_ns)
                    if ts < min_ts:
                        min_ts = ts
                    try:
                        hb = json.loads(line)
                        hb['_ts_ns'] = ts
                        batch.append(hb)
                    except json.JSONDecodeError:
                        continue

            if not batch:
                break

            all_heartbeats.extend(batch)

            # Move window back; stop if we got fewer than limit
            total_values = sum(
                len(s.get('values', [])) for s in results)
            if total_values < limit:
                break
            current_end = min_ts - 1

        # Deduplicate by (server_hash, send_time)
        seen = set()
        unique = []
        for hb in all_heartbeats:
            key = (hb.get('server_hash', ''), hb.get('send_time', 0))
            if key not in seen:
                seen.add(key)
                unique.append(hb)

        # Filter by server_hash if specified
        if server_hash:
            unique = [
                hb for hb in unique
                if hb.get('server_hash', '') == server_hash
            ]

        # Sort by send_time
        unique.sort(key=lambda hb: hb.get('send_time', 0))
        return unique


class HeartbeatAnalyzer:
    """Analyze heartbeat data for GPU allocation and Kueue ROI."""

    def __init__(self, heartbeats: List[dict],
                 price_table: GPUPriceTable):
        self.heartbeats = heartbeats
        self.price_table = price_table

    def servers(self) -> Dict[str, dict]:
        """Group heartbeats by server_hash.

        Returns:
            Dict mapping server_hash -> {
                hostname, release_name, sky_version, heartbeats
            }
        """
        groups: Dict[str, dict] = {}
        for hb in self.heartbeats:
            sh = hb.get('server_hash', 'unknown')
            if sh not in groups:
                groups[sh] = {
                    'hostname': hb.get('hostname', 'unknown'),
                    'release_name': hb.get('release_name'),
                    'sky_version': hb.get('sky_version', 'unknown'),
                    'heartbeats': [],
                }
            groups[sh]['heartbeats'].append(hb)
        return groups

    def gpu_allocation_timeseries(
            self, heartbeats: List[dict]
    ) -> List[Dict[str, Any]]:
        """Build GPU allocation time series from billing plugin.

        Returns list of records:
            {timestamp, context, gpu_type, total, allocated}
        """
        records = []
        for hb in heartbeats:
            ts = _ns_to_datetime(hb.get('send_time', 0))
            billing = hb.get('plugins', {}).get('billing', {})
            for ctx in billing.get('contexts', []):
                ctx_name = ctx.get('context_name', 'unknown')
                gpus = ctx.get('gpus', {})
                by_type = gpus.get('by_type', {})
                for gpu_type, counts in by_type.items():
                    records.append({
                        'timestamp': ts,
                        'context': ctx_name,
                        'gpu_type': gpu_type,
                        'total': counts.get('total', 0),
                        'allocated': counts.get('allocated', 0),
                    })
        return records

    def gpu_allocation_context_timeseries(
            self, heartbeats: List[dict]
    ) -> List[Dict[str, Any]]:
        """Build context-level GPU allocation time series.

        Returns list of records:
            {timestamp, context, total_gpus, allocated_gpus}
        """
        records = []
        for hb in heartbeats:
            ts = _ns_to_datetime(hb.get('send_time', 0))
            billing = hb.get('plugins', {}).get('billing', {})
            for ctx in billing.get('contexts', []):
                ctx_name = ctx.get('context_name', 'unknown')
                gpus = ctx.get('gpus', {})
                records.append({
                    'timestamp': ts,
                    'context': ctx_name,
                    'total_gpus': gpus.get('total', 0),
                    'allocated_gpus': gpus.get('allocated', 0),
                })
        return records

    def gpu_allocation_stats(
        self, heartbeats: List[dict]
    ) -> Dict[str, Dict[str, Any]]:
        """Compute per-context GPU allocation stats (across all types).

        Returns:
            {context: {total_gpus, allocated_avg, allocated_min,
                       allocated_max, util_pct_avg}}
        """
        # Collect per-context per-heartbeat totals
        ctx_data: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for hb in heartbeats:
            billing = hb.get('plugins', {}).get('billing', {})
            for ctx in billing.get('contexts', []):
                ctx_name = ctx.get('context_name', 'unknown')
                gpus = ctx.get('gpus', {})
                total = gpus.get('total', 0)
                allocated = gpus.get('allocated', 0)
                ctx_data[ctx_name].append((total, allocated))

        stats = {}
        for ctx_name, samples in ctx_data.items():
            totals = [s[0] for s in samples]
            allocs = [s[1] for s in samples]
            max_total = max(totals) if totals else 0
            avg_alloc = sum(allocs) / len(allocs) if allocs else 0
            min_alloc = min(allocs) if allocs else 0
            max_alloc = max(allocs) if allocs else 0
            util_pct = (avg_alloc / max_total *
                        100) if max_total > 0 else 0
            stats[ctx_name] = {
                'total_gpus': max_total,
                'allocated_avg': avg_alloc,
                'allocated_min': min_alloc,
                'allocated_max': max_alloc,
                'util_pct_avg': util_pct,
            }
        return stats

    def _context_gpu_types(
            self, heartbeats: List[dict]) -> Dict[str, str]:
        """Determine dominant GPU type per context from billing data."""
        ctx_gpu_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int))
        for hb in heartbeats:
            billing = hb.get('plugins', {}).get('billing', {})
            for ctx in billing.get('contexts', []):
                ctx_name = ctx.get('context_name', 'unknown')
                by_type = ctx.get('gpus', {}).get('by_type', {})
                for gpu_type, counts in by_type.items():
                    ctx_gpu_counts[ctx_name][gpu_type] += counts.get(
                        'total', 0)

        result = {}
        for ctx_name, gpu_types in ctx_gpu_counts.items():
            if gpu_types:
                # Use the most expensive GPU type present
                best = max(gpu_types.keys(),
                           key=lambda g: self.price_table.get_price(g))
                result[ctx_name] = best
        return result

    def kueue_borrowing_timeseries(
            self, heartbeats: List[dict]
    ) -> List[Dict[str, Any]]:
        """Build Kueue borrowing time series.

        Returns list of records:
            {timestamp, context, queue, borrowed_gpus, gpu_type, roi_value}
        """
        ctx_gpu_types = self._context_gpu_types(heartbeats)
        records = []

        for hb in heartbeats:
            ts = _ns_to_datetime(hb.get('send_time', 0))
            interval = hb.get('interval_seconds', 600)
            kueue = hb.get('plugins', {}).get('kueue', {})

            for ctx in kueue.get('contexts', []):
                ctx_name = ctx.get('context', 'unknown')
                gpu_type = ctx_gpu_types.get(ctx_name, 'H100')
                price = self.price_table.get_price(gpu_type)

                for queue in ctx.get('cluster_queues', []):
                    queue_name = queue.get('name', 'unknown')
                    borrowed = 0

                    for flavor in queue.get('flavors_reservation', []):
                        for res in flavor.get('resources', []):
                            if 'gpu' in res.get('name', '').lower():
                                borrowed += int(
                                    res.get('borrowed', '0'))

                    roi = (borrowed * price *
                           (interval / 3600)) if borrowed > 0 else 0

                    records.append({
                        'timestamp': ts,
                        'context': ctx_name,
                        'queue': queue_name,
                        'borrowed_gpus': borrowed,
                        'gpu_type': gpu_type,
                        'roi_value': roi,
                        'admitted_workloads':
                            queue.get('admitted_workloads', 0),
                        'pending_workloads':
                            queue.get('num_pending_workloads',
                                      queue.get('pending_workloads', 0)),
                    })
        return records

    def kueue_borrowing_stats(
        self, heartbeats: List[dict]
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Compute per-context, per-queue Kueue borrowing stats.

        Returns:
            {context: {queue: {borrowed_avg, borrowed_min, borrowed_max,
                               gpu_hours, roi_total, gpu_type}}}
        """
        records = self.kueue_borrowing_timeseries(heartbeats)

        # Group by (context, queue)
        groups: Dict[Tuple[str, str],
                      List[Dict[str, Any]]] = defaultdict(list)
        for r in records:
            groups[(r['context'], r['queue'])].append(r)

        stats: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        for (ctx, queue), recs in groups.items():
            borrowed_vals = [r['borrowed_gpus'] for r in recs]
            roi_vals = [r['roi_value'] for r in recs]
            avg_b = (sum(borrowed_vals) /
                     len(borrowed_vals)) if borrowed_vals else 0
            gpu_hours = sum(r['roi_value'] / self.price_table.get_price(
                r['gpu_type']) for r in recs if r['roi_value'] > 0)

            stats[ctx][queue] = {
                'borrowed_avg': avg_b,
                'borrowed_min': min(borrowed_vals) if borrowed_vals else 0,
                'borrowed_max': max(borrowed_vals) if borrowed_vals else 0,
                'gpu_hours': gpu_hours,
                'roi_total': sum(roi_vals),
                'gpu_type': recs[0]['gpu_type'] if recs else 'H100',
            }
        return dict(stats)


def _parse_range(range_str: str) -> float:
    """Parse range string like '1h', '24h', '7d', '30d' to seconds."""
    s = range_str.strip().lower()
    if s.endswith('d'):
        return float(s[:-1]) * 86400
    elif s.endswith('h'):
        return float(s[:-1]) * 3600
    elif s.endswith('m'):
        return float(s[:-1]) * 60
    else:
        raise ValueError(f'Invalid range format: {range_str}. '
                         f'Use e.g. 1h, 24h, 7d, 30d')


def _ns_to_datetime(ns: int) -> datetime:
    """Convert nanosecond timestamp to datetime."""
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)


def _format_dt(dt: datetime) -> str:
    return dt.strftime('%Y-%m-%d %H:%M')


# ── Plotting ──────────────────────────────────────────────────────────

def plot_gpu_allocation(analyzer: HeartbeatAnalyzer,
                        heartbeats: List[dict],
                        output_path: str):
    """Plot GPU allocation over time, one subplot per context."""
    records = analyzer.gpu_allocation_timeseries(heartbeats)
    if not records:
        print('  No GPU allocation data to plot.')
        return

    # Group by context
    ctx_data: Dict[str, Dict[str, List]] = defaultdict(
        lambda: defaultdict(lambda: {'ts': [], 'alloc': [], 'total': []}))
    for r in records:
        entry = ctx_data[r['context']][r['gpu_type']]
        entry['ts'].append(r['timestamp'])
        entry['alloc'].append(r['allocated'])
        entry['total'].append(r['total'])

    contexts = sorted(ctx_data.keys())
    n = len(contexts)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n),
                             squeeze=False, sharex=False)

    for i, ctx in enumerate(contexts):
        ax = axes[i, 0]
        gpu_types = ctx_data[ctx]
        for gpu_type, vals in sorted(gpu_types.items()):
            allocs = vals['alloc']
            totals = vals['total']
            avg_a = sum(allocs) / len(allocs) if allocs else 0
            min_a = min(allocs) if allocs else 0
            max_a = max(allocs) if allocs else 0
            max_t = max(totals) if totals else 0

            label = (f'{gpu_type} (avg={avg_a:.1f}, min={min_a}, '
                     f'max={max_a} of {max_t} total)')
            ax.plot(vals['ts'], allocs, label=label, linewidth=1.5)
            ax.axhline(y=max_t, linestyle='--', alpha=0.4,
                       linewidth=1)

        ax.set_title(f'GPU Allocation: {ctx}')
        ax.set_ylabel('GPU Count')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter('%m-%d %H:%M'))
        fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  {os.path.basename(output_path)}')


def plot_kueue_borrowing(analyzer: HeartbeatAnalyzer,
                         heartbeats: List[dict],
                         output_path: str) -> bool:
    """Plot Kueue borrowed GPUs over time. Returns False if skipped."""
    records = analyzer.kueue_borrowing_timeseries(heartbeats)
    has_borrowing = any(r['borrowed_gpus'] > 0 for r in records)

    if not has_borrowing:
        return False

    # Group by context -> queue
    ctx_queue: Dict[str, Dict[str, Dict[str, list]]] = defaultdict(
        lambda: defaultdict(lambda: {'ts': [], 'borrowed': []}))
    for r in records:
        if r['borrowed_gpus'] > 0 or r['context'] in ctx_queue:
            entry = ctx_queue[r['context']][r['queue']]
            entry['ts'].append(r['timestamp'])
            entry['borrowed'].append(r['borrowed_gpus'])

    # Only contexts with non-zero borrowing
    contexts = [c for c in ctx_queue
                if any(any(v > 0 for v in q['borrowed'])
                       for q in ctx_queue[c].values())]

    if not contexts:
        return False

    stats = analyzer.kueue_borrowing_stats(heartbeats)

    n = len(contexts)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n),
                             squeeze=False, sharex=False)

    for i, ctx in enumerate(contexts):
        ax = axes[i, 0]
        queues = ctx_queue[ctx]
        for queue_name, vals in sorted(queues.items()):
            q_stats = stats.get(ctx, {}).get(queue_name, {})
            avg_b = q_stats.get('borrowed_avg', 0)
            gpu_hrs = q_stats.get('gpu_hours', 0)
            roi = q_stats.get('roi_total', 0)
            label = (f'{queue_name} (avg={avg_b:.1f}, '
                     f'GPU-hrs={gpu_hrs:.1f}, ROI=${roi:.2f})')
            ax.fill_between(vals['ts'], vals['borrowed'],
                            alpha=0.5, label=label)
            ax.plot(vals['ts'], vals['borrowed'], linewidth=1)

        ax.set_title(f'Kueue Borrowed GPUs: {ctx}')
        ax.set_ylabel('Borrowed GPUs')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter('%m-%d %H:%M'))
        fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  {os.path.basename(output_path)}')
    return True


def plot_gpu_utilization_heatmap(analyzer: HeartbeatAnalyzer,
                                 heartbeats: List[dict],
                                 output_path: str):
    """Plot GPU utilization heatmap (contexts x hourly buckets)."""
    ctx_ts = analyzer.gpu_allocation_context_timeseries(heartbeats)
    if not ctx_ts:
        print('  No GPU utilization data for heatmap.')
        return

    # Bucket by hour
    ctx_hourly: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list))
    for r in ctx_ts:
        hour_key = r['timestamp'].strftime('%Y-%m-%d %H:00')
        total = r['total_gpus']
        alloc = r['allocated_gpus']
        util = (alloc / total * 100) if total > 0 else 0
        ctx_hourly[r['context']][hour_key].append(util)

    contexts = sorted(ctx_hourly.keys())
    all_hours = sorted(
        set(h for ctx in ctx_hourly.values() for h in ctx.keys()))

    if not all_hours or not contexts:
        print('  Insufficient data for heatmap.')
        return

    # Build matrix
    matrix = []
    for ctx in contexts:
        row = []
        for hour in all_hours:
            vals = ctx_hourly[ctx].get(hour, [0])
            row.append(sum(vals) / len(vals))
        matrix.append(row)

    matrix_np = np.array(matrix)

    # Reduce x-axis labels if too many
    n_hours = len(all_hours)
    step = max(1, n_hours // 24)
    x_labels = [all_hours[i] if i % step == 0 else ''
                for i in range(n_hours)]

    fig, ax = plt.subplots(figsize=(max(12, n_hours * 0.3),
                                    max(3, len(contexts) * 0.8)))
    im = ax.imshow(matrix_np, aspect='auto', cmap='YlOrRd',
                   vmin=0, vmax=100)

    ax.set_yticks(range(len(contexts)))
    ax.set_yticklabels(contexts, fontsize=8)
    ax.set_xticks(range(n_hours))
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=6)

    cbar = fig.colorbar(im, ax=ax, label='Utilization %')
    ax.set_title('GPU Utilization Heatmap (hourly avg)')
    ax.set_xlabel('Time')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  {os.path.basename(output_path)}')


# ── CSV Output ────────────────────────────────────────────────────────

def write_gpu_allocation_csv(analyzer: HeartbeatAnalyzer,
                             servers: Dict[str, dict],
                             output_path: str):
    """Write GPU allocation time series to CSV."""
    path = output_path.replace('.csv', '_gpu_allocation.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'timestamp', 'server_hash', 'hostname', 'context',
            'gpu_type', 'total', 'allocated'
        ])
        for sh, info in servers.items():
            records = analyzer.gpu_allocation_timeseries(
                info['heartbeats'])
            for r in records:
                writer.writerow([
                    r['timestamp'].isoformat(),
                    sh,
                    info['hostname'],
                    r['context'],
                    r['gpu_type'],
                    r['total'],
                    r['allocated'],
                ])
    print(f'  {os.path.basename(path)}')


def write_kueue_borrowing_csv(analyzer: HeartbeatAnalyzer,
                              servers: Dict[str, dict],
                              output_path: str):
    """Write Kueue borrowing time series to CSV."""
    path = output_path.replace('.csv', '_kueue_borrowing.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'timestamp', 'server_hash', 'hostname', 'context',
            'queue', 'borrowed_gpus', 'gpu_type', 'roi_value'
        ])
        for sh, info in servers.items():
            records = analyzer.kueue_borrowing_timeseries(
                info['heartbeats'])
            for r in records:
                writer.writerow([
                    r['timestamp'].isoformat(),
                    sh,
                    info['hostname'],
                    r['context'],
                    r['queue'],
                    r['borrowed_gpus'],
                    r['gpu_type'],
                    f'{r["roi_value"]:.4f}',
                ])
    print(f'  {os.path.basename(path)}')


# ── CLI ───────────────────────────────────────────────────────────────

def parse_gpu_prices(s: str) -> Dict[str, float]:
    """Parse 'H100=3.50,H200=4.50' into dict."""
    prices = {}
    for pair in s.split(','):
        pair = pair.strip()
        if '=' not in pair:
            continue
        k, v = pair.split('=', 1)
        prices[k.strip()] = float(v.strip())
    return prices


def print_report(analyzer: HeartbeatAnalyzer,
                 servers: Dict[str, dict],
                 range_str: str):
    """Print CLI report."""
    total_hbs = sum(len(s['heartbeats']) for s in servers.values())

    # Compute date range from data
    all_ts = []
    for info in servers.values():
        for hb in info['heartbeats']:
            st = hb.get('send_time', 0)
            if st > 0:
                all_ts.append(st)

    if all_ts:
        dt_start = _ns_to_datetime(min(all_ts))
        dt_end = _ns_to_datetime(max(all_ts))
        range_display = (f'{_format_dt(dt_start)} to '
                         f'{_format_dt(dt_end)} ({range_str})')
    else:
        range_display = range_str

    print(f'\n=== SkyPilot Heartbeat ROI Report ===')
    print(f'Range: {range_display}')
    print(f'Total heartbeats: {total_hbs}')

    for sh, info in sorted(servers.items()):
        hbs = info['heartbeats']
        hostname = info['hostname']
        release = info['release_name']
        version = info['sky_version']
        label = hostname
        if release:
            label = f'{hostname}, release={release}'

        print(f'\n--- Server: {sh} ({label}) ---')
        print(f'    SkyPilot version: {version}')
        print(f'    Heartbeats: {len(hbs)}')

        # GPU Allocation
        gpu_stats = analyzer.gpu_allocation_stats(hbs)
        print(f'\n    GPU Allocation (across all types):')
        if gpu_stats:
            print(f'      {"Context":<25s} {"Total GPUs":<13s} '
                  f'{"Allocated (avg/min/max)":<28s} {"Util% (avg)"}')
            for ctx, s in sorted(gpu_stats.items()):
                alloc_str = (f'{s["allocated_avg"]:.1f} / '
                             f'{s["allocated_min"]} / '
                             f'{s["allocated_max"]}')
                print(f'      {ctx:<25s} {s["total_gpus"]:<13d} '
                      f'{alloc_str:<28s} {s["util_pct_avg"]:.1f}%')
        else:
            print(f'      (no GPU data)')

        # Kueue Borrowing
        kueue_stats = analyzer.kueue_borrowing_stats(hbs)
        print(f'\n    Kueue Borrowing ROI:')
        has_borrowing = any(
            any(q['borrowed_max'] > 0 for q in queues.values())
            for queues in kueue_stats.values())

        total_gpu_hrs = 0.0
        total_roi = 0.0

        if has_borrowing:
            print(f'      {"Context":<25s} {"Queue":<18s} '
                  f'{"Borrowed (avg/min/max)":<26s} '
                  f'{"GPU-Hrs":<10s} {"Value($)"}')
            for ctx, queues in sorted(kueue_stats.items()):
                for queue, s in sorted(queues.items()):
                    if s['borrowed_max'] == 0:
                        continue
                    borrow_str = (f'{s["borrowed_avg"]:.1f} / '
                                  f'{s["borrowed_min"]} / '
                                  f'{s["borrowed_max"]}')
                    total_gpu_hrs += s['gpu_hours']
                    total_roi += s['roi_total']
                    print(
                        f'      {ctx:<25s} {queue:<18s} '
                        f'{borrow_str:<26s} '
                        f'{s["gpu_hours"]:<10.1f} '
                        f'${s["roi_total"]:.2f}')
        else:
            print(f'      (no borrowing in period)')

        print(f'\n    Total Borrowed GPU-Hours: {total_gpu_hrs:.1f}')
        print(f'    Estimated ROI: ${total_roi:.2f}')


def main():
    parser = argparse.ArgumentParser(
        description='Analyze SkyPilot server heartbeat ROI metrics')
    parser.add_argument('--grafana-url', required=True,
                        help='Grafana endpoint URL')
    parser.add_argument('--grafana-user', required=True,
                        help='Grafana basic auth user')
    parser.add_argument('--grafana-password', required=True,
                        help='Grafana basic auth password')
    parser.add_argument('--range', default='7d', dest='time_range',
                        help='Time range (1h, 24h, 7d, 30d)')
    parser.add_argument('--server-hash', default=None,
                        help='Filter to specific server hash')
    parser.add_argument('--gpu-prices', default=None,
                        help='GPU prices as K=V,K=V '
                             '(e.g. H100=3.50,H200=4.50)')
    parser.add_argument('--csv-output', default=None,
                        help='CSV output base path')
    parser.add_argument('--plot-dir', default=None,
                        help='Directory to save matplotlib plots')
    args = parser.parse_args()

    # GPU prices
    overrides = None
    if args.gpu_prices:
        overrides = parse_gpu_prices(args.gpu_prices)
    price_table = GPUPriceTable(overrides)

    # Fetch data
    print(f'Fetching heartbeats from {args.grafana_url} '
          f'(range={args.time_range})...')
    client = LokiClient(args.grafana_url, args.grafana_user,
                        args.grafana_password)
    heartbeats = client.query_heartbeats(args.time_range,
                                         args.server_hash)

    if not heartbeats:
        print('No heartbeat data found.')
        sys.exit(0)

    print(f'Fetched {len(heartbeats)} heartbeats.')

    analyzer = HeartbeatAnalyzer(heartbeats, price_table)
    servers = analyzer.servers()

    # CLI report
    print_report(analyzer, servers, args.time_range)

    # Plots
    if args.plot_dir:
        if not HAS_MATPLOTLIB:
            print('\nWarning: matplotlib not installed. '
                  'Skipping plots. Install with: '
                  'pip install matplotlib')
        else:
            os.makedirs(args.plot_dir, exist_ok=True)
            print(f'\nPlots saved to: {args.plot_dir}/')

            all_hbs = [hb for info in servers.values()
                       for hb in info['heartbeats']]

            plot_gpu_allocation(
                analyzer, all_hbs,
                os.path.join(args.plot_dir, 'gpu_allocation.png'))

            ok = plot_kueue_borrowing(
                analyzer, all_hbs,
                os.path.join(args.plot_dir, 'kueue_borrowing.png'))
            if not ok:
                print('  kueue_borrowing.png  '
                      '(skipped - no borrowing data)')

            plot_gpu_utilization_heatmap(
                analyzer, all_hbs,
                os.path.join(args.plot_dir, 'gpu_utilization.png'))

    # CSV
    if args.csv_output:
        print(f'\nCSVs saved to:')
        write_gpu_allocation_csv(analyzer, servers, args.csv_output)
        write_kueue_borrowing_csv(analyzer, servers, args.csv_output)

    print()


if __name__ == '__main__':
    main()
