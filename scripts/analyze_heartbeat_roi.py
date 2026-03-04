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
        'H100': 1.30,
        'H200': 2.00,
        'A100': 1.00,
        'A100-80GB': 1.10,
        'L40S': 0.70,
        'L40': 0.70,
        'L4': 0.70,
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
        """Compute per-context GPU allocation stats with per-type breakdown.

        Returns:
            {context: {total_gpus, allocated_avg, allocated_min,
                       allocated_max, util_pct_avg,
                       by_type: {gpu_type: {total, alloc_avg,
                                            alloc_min, alloc_max,
                                            util_pct_avg}}}}
        """
        # Collect per-context per-heartbeat totals
        ctx_data: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        # Per-context per-type: list of (total, allocated)
        ctx_type_data: Dict[str, Dict[str, List[Tuple[int, int]]]] = \
            defaultdict(lambda: defaultdict(list))

        for hb in heartbeats:
            billing = hb.get('plugins', {}).get('billing', {})
            for ctx in billing.get('contexts', []):
                ctx_name = ctx.get('context_name', 'unknown')
                gpus = ctx.get('gpus', {})
                total = gpus.get('total', 0)
                allocated = gpus.get('allocated', 0)
                ctx_data[ctx_name].append((total, allocated))
                for gpu_type, counts in gpus.get('by_type', {}).items():
                    ctx_type_data[ctx_name][gpu_type].append(
                        (counts.get('total', 0),
                         counts.get('allocated', 0)))

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

            by_type = {}
            for gpu_type, type_samples in ctx_type_data[ctx_name].items():
                t_totals = [s[0] for s in type_samples]
                t_allocs = [s[1] for s in type_samples]
                t_max = max(t_totals) if t_totals else 0
                t_avg = (sum(t_allocs) / len(t_allocs)
                         if t_allocs else 0)
                t_min = min(t_allocs) if t_allocs else 0
                t_max_a = max(t_allocs) if t_allocs else 0
                t_util = (t_avg / t_max * 100) if t_max > 0 else 0
                by_type[gpu_type] = {
                    'total': t_max,
                    'alloc_avg': t_avg,
                    'alloc_min': t_min,
                    'alloc_max': t_max_a,
                    'util_pct_avg': t_util,
                }

            stats[ctx_name] = {
                'total_gpus': max_total,
                'allocated_avg': avg_alloc,
                'allocated_min': min_alloc,
                'allocated_max': max_alloc,
                'util_pct_avg': util_pct,
                'by_type': by_type,
            }
        return stats

    def _context_gpu_types(
            self, heartbeats: List[dict]) -> Dict[str, str]:
        """Determine dominant GPU type per context from billing data."""
        ctx_gpu_counts = self._context_gpu_mix(heartbeats)
        result = {}
        for ctx_name, gpu_types in ctx_gpu_counts.items():
            if gpu_types:
                best = max(gpu_types.keys(),
                           key=lambda g: self.price_table.get_price(g))
                result[ctx_name] = best
        return result

    def _context_gpu_mix(
            self,
            heartbeats: List[dict]) -> Dict[str, Dict[str, int]]:
        """Get cumulative GPU counts per type per context."""
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
        return dict(ctx_gpu_counts)

    def fleet_overview(self,
                       servers: Dict[str, dict]) -> Dict[str, Any]:
        """Compute aggregate fleet metrics from the latest heartbeat per server.

        Returns:
            {num_servers, num_contexts, total_gpus, gpus_by_type,
             total_nodes, nodes_ready, total_clusters, total_pods}
        """
        total_gpus = 0
        gpus_by_type: Dict[str, int] = defaultdict(int)
        total_nodes = 0
        nodes_ready = 0
        total_clusters = 0
        total_pods = 0
        contexts_seen: set = set()

        for _sh, info in servers.items():
            hbs = info['heartbeats']
            if not hbs:
                continue
            # Use latest heartbeat per server
            latest = max(hbs, key=lambda h: h.get('send_time', 0))
            billing = latest.get('plugins', {}).get('billing', {})
            for ctx in billing.get('contexts', []):
                ctx_name = ctx.get('context_name', 'unknown')
                contexts_seen.add(ctx_name)
                gpus = ctx.get('gpus', {})
                total_gpus += gpus.get('total', 0)
                for gpu_type, counts in gpus.get('by_type', {}).items():
                    gpus_by_type[gpu_type] += counts.get('total', 0)
                nodes = ctx.get('nodes', {})
                total_nodes += nodes.get('total', 0)
                nodes_ready += nodes.get('ready', 0)
                total_clusters += ctx.get('skypilot_clusters', 0)
                total_pods += ctx.get('skypilot_pods', 0)

        return {
            'num_servers': len(servers),
            'num_contexts': len(contexts_seen),
            'total_gpus': total_gpus,
            'gpus_by_type': dict(gpus_by_type),
            'total_nodes': total_nodes,
            'nodes_ready': nodes_ready,
            'total_clusters': total_clusters,
            'total_pods': total_pods,
        }

    def kueue_queue_health_timeseries(
            self, heartbeats: List[dict]
    ) -> List[Dict[str, Any]]:
        """Build per-heartbeat queue admitted/pending time series.

        Returns list of records:
            {timestamp, context, queue, admitted_workloads,
             num_pending_workloads}
        """
        records = []
        for hb in heartbeats:
            ts = _ns_to_datetime(hb.get('send_time', 0))
            kueue = hb.get('plugins', {}).get('kueue', {})
            for ctx in kueue.get('contexts', []):
                ctx_name = ctx.get('context', 'unknown')
                if not ctx.get('kueue_enabled', True):
                    continue
                for queue in ctx.get('cluster_queues', []):
                    records.append({
                        'timestamp': ts,
                        'context': ctx_name,
                        'queue': queue.get('name', 'unknown'),
                        'admitted_workloads':
                            queue.get('admitted_workloads', 0),
                        'num_pending_workloads':
                            queue.get('num_pending_workloads',
                                      queue.get('pending_workloads', 0)),
                    })
        return records

    def kueue_queue_health_stats(
        self, heartbeats: List[dict]
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Compute per-context, per-queue queue health stats.

        Returns:
            {context: {queue: {admitted_avg, admitted_max,
                               pending_avg, pending_max,
                               preempted_total}}}
        """
        records = self.kueue_queue_health_timeseries(heartbeats)

        groups: Dict[Tuple[str, str],
                      List[Dict[str, Any]]] = defaultdict(list)
        for r in records:
            groups[(r['context'], r['queue'])].append(r)

        # Gather preempted_events per context from raw heartbeats
        preempted: Dict[str, int] = defaultdict(int)
        for hb in heartbeats:
            kueue = hb.get('plugins', {}).get('kueue', {})
            for ctx in kueue.get('contexts', []):
                ctx_name = ctx.get('context', 'unknown')
                preempted[ctx_name] += ctx.get('preempted_events', 0)

        stats: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        for (ctx, queue), recs in groups.items():
            admitted = [r['admitted_workloads'] for r in recs]
            pending = [r['num_pending_workloads'] for r in recs]
            stats[ctx][queue] = {
                'admitted_avg':
                    sum(admitted) / len(admitted) if admitted else 0,
                'admitted_max': max(admitted) if admitted else 0,
                'pending_avg':
                    sum(pending) / len(pending) if pending else 0,
                'pending_max': max(pending) if pending else 0,
                'preempted_total': preempted.get(ctx, 0),
            }
        return dict(stats)

    def gpu_hours_stats(
        self, heartbeats: List[dict]
    ) -> Dict[str, Dict[str, Any]]:
        """Compute per-context GPU-hours (total, allocated, idle).

        Uses interval_seconds from each heartbeat to compute hours.

        Returns:
            {context: {total_gpu_hours, allocated_gpu_hours,
                       idle_gpu_hours, idle_pct,
                       by_type: {gpu_type: {total_hrs, alloc_hrs,
                                            idle_hrs}}}}
        """
        # Accumulate per context, per gpu_type
        ctx_type_data: Dict[str, Dict[str, Dict[str, float]]] = \
            defaultdict(lambda: defaultdict(
                lambda: {'total_hrs': 0.0, 'alloc_hrs': 0.0}))

        for hb in heartbeats:
            interval_hrs = hb.get('interval_seconds', 600) / 3600.0
            billing = hb.get('plugins', {}).get('billing', {})
            for ctx in billing.get('contexts', []):
                ctx_name = ctx.get('context_name', 'unknown')
                for gpu_type, counts in ctx.get(
                        'gpus', {}).get('by_type', {}).items():
                    total = counts.get('total', 0)
                    allocated = counts.get('allocated', 0)
                    ctx_type_data[ctx_name][gpu_type][
                        'total_hrs'] += total * interval_hrs
                    ctx_type_data[ctx_name][gpu_type][
                        'alloc_hrs'] += allocated * interval_hrs

        stats: Dict[str, Dict[str, Any]] = {}
        for ctx_name, gpu_types in ctx_type_data.items():
            total_hrs = sum(v['total_hrs'] for v in gpu_types.values())
            alloc_hrs = sum(v['alloc_hrs'] for v in gpu_types.values())
            idle_hrs = total_hrs - alloc_hrs
            idle_pct = (idle_hrs / total_hrs * 100) if total_hrs > 0 else 0

            by_type = {}
            for gpu_type, vals in gpu_types.items():
                t = vals['total_hrs']
                a = vals['alloc_hrs']
                by_type[gpu_type] = {
                    'total_hrs': t,
                    'alloc_hrs': a,
                    'idle_hrs': t - a,
                }

            stats[ctx_name] = {
                'total_gpu_hours': total_hrs,
                'allocated_gpu_hours': alloc_hrs,
                'idle_gpu_hours': idle_hrs,
                'idle_pct': idle_pct,
                'by_type': by_type,
            }
        return stats

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

        Borrowed GPUs are distributed proportionally across GPU types
        present in each context (from billing data) for the by_type
        breakdown.

        Returns:
            {context: {queue: {borrowed_avg, borrowed_min, borrowed_max,
                               gpu_hours, roi_total, gpu_type,
                               by_type: {gpu_type: {gpu_hours, roi}}}}}
        """
        records = self.kueue_borrowing_timeseries(heartbeats)
        gpu_mix = self._context_gpu_mix(heartbeats)

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

            # Per-GPU-type breakdown: distribute proportionally
            ctx_types = gpu_mix.get(ctx, {})
            total_ctx_gpus = sum(ctx_types.values())
            by_type: Dict[str, Dict[str, float]] = {}
            if total_ctx_gpus > 0 and gpu_hours > 0:
                for gtype, gcount in sorted(ctx_types.items()):
                    frac = gcount / total_ctx_gpus
                    type_gpu_hrs = gpu_hours * frac
                    type_price = self.price_table.get_price(gtype)
                    by_type[gtype] = {
                        'gpu_hours': type_gpu_hrs,
                        'roi': type_gpu_hrs * type_price,
                    }

            stats[ctx][queue] = {
                'borrowed_avg': avg_b,
                'borrowed_min': min(borrowed_vals) if borrowed_vals else 0,
                'borrowed_max': max(borrowed_vals) if borrowed_vals else 0,
                'gpu_hours': gpu_hours,
                'roi_total': sum(roi_vals),
                'gpu_type': recs[0]['gpu_type'] if recs else 'H100',
                'by_type': by_type,
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


def plot_fleet_overview(analyzer: HeartbeatAnalyzer,
                        servers: Dict[str, dict],
                        output_path: str):
    """Plot fleet GPU composition as horizontal bar chart by GPU type."""
    overview = analyzer.fleet_overview(servers)
    gpus_by_type = overview.get('gpus_by_type', {})
    if not gpus_by_type:
        print('  No GPU data for fleet overview plot.')
        return

    gpu_types = sorted(gpus_by_type.keys(),
                       key=lambda g: gpus_by_type[g], reverse=True)
    counts = [gpus_by_type[g] for g in gpu_types]

    fig, ax = plt.subplots(figsize=(10, max(3, len(gpu_types) * 0.6)))
    bars = ax.barh(gpu_types, counts, color='steelblue', edgecolor='white')

    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(count), va='center', fontsize=10)

    ax.set_xlabel('Total GPUs')
    ax.set_title(f'Fleet GPU Composition '
                 f'({overview["num_servers"]} servers, '
                 f'{overview["num_contexts"]} contexts)')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  {os.path.basename(output_path)}')


def plot_kueue_queue_health(analyzer: HeartbeatAnalyzer,
                             heartbeats: List[dict],
                             output_path: str) -> bool:
    """Plot queue admitted vs pending workloads over time.

    One subplot per context that has kueue enabled.
    Returns False if no kueue data.
    """
    records = analyzer.kueue_queue_health_timeseries(heartbeats)
    if not records:
        return False

    # Group by context -> queue
    ctx_queue: Dict[str, Dict[str, Dict[str, list]]] = defaultdict(
        lambda: defaultdict(
            lambda: {'ts': [], 'admitted': [], 'pending': []}))
    for r in records:
        entry = ctx_queue[r['context']][r['queue']]
        entry['ts'].append(r['timestamp'])
        entry['admitted'].append(r['admitted_workloads'])
        entry['pending'].append(r['num_pending_workloads'])

    contexts = sorted(ctx_queue.keys())
    if not contexts:
        return False

    n = len(contexts)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n),
                             squeeze=False, sharex=False)

    for i, ctx in enumerate(contexts):
        ax = axes[i, 0]
        queues = ctx_queue[ctx]
        for queue_name, vals in sorted(queues.items()):
            ax.fill_between(vals['ts'], vals['admitted'],
                            alpha=0.4, color='green',
                            label=f'{queue_name} admitted')
            ax.fill_between(vals['ts'], vals['pending'],
                            alpha=0.4, color='orange',
                            label=f'{queue_name} pending')

        ax.set_title(f'Queue Health: {ctx}')
        ax.set_ylabel('Workloads')
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


def write_gpu_hours_csv(analyzer: HeartbeatAnalyzer,
                         servers: Dict[str, dict],
                         output_path: str):
    """Write GPU-hours breakdown per context per GPU type to CSV."""
    path = output_path.replace('.csv', '_gpu_hours.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'server_hash', 'hostname', 'context', 'gpu_type',
            'total_gpu_hours', 'allocated_gpu_hours', 'idle_gpu_hours'
        ])
        for sh, info in servers.items():
            stats = analyzer.gpu_hours_stats(info['heartbeats'])
            for ctx_name, ctx_stats in sorted(stats.items()):
                for gpu_type, type_stats in sorted(
                        ctx_stats['by_type'].items()):
                    writer.writerow([
                        sh,
                        info['hostname'],
                        ctx_name,
                        gpu_type,
                        f'{type_stats["total_hrs"]:.2f}',
                        f'{type_stats["alloc_hrs"]:.2f}',
                        f'{type_stats["idle_hrs"]:.2f}',
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


def _sanitize_dirname(name: str) -> str:
    """Sanitize a string for use as a directory name."""
    return name.replace('/', '_').replace('\\', '_').replace(
        ':', '_').replace(' ', '_')


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

    # Fleet Overview
    overview = analyzer.fleet_overview(servers)
    print(f'\n=== Fleet Overview ===')
    print(f'Servers: {overview["num_servers"]}')
    print(f'Contexts: {overview["num_contexts"]} '
          f'(across {overview["num_servers"]} servers)')
    gpu_parts = ', '.join(
        f'{g}: {c}' for g, c in sorted(
            overview['gpus_by_type'].items(),
            key=lambda x: x[1], reverse=True))
    print(f'Total GPUs: {overview["total_gpus"]}'
          + (f' ({gpu_parts})' if gpu_parts else ''))
    print(f'Total Nodes: {overview["total_nodes"]} '
          f'({overview["nodes_ready"]} ready)')
    print(f'Total SkyPilot Clusters: {overview["total_clusters"]}')
    print(f'Total SkyPilot Pods: {overview["total_pods"]}')

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
        print(f'\n    GPU Allocation:')
        if gpu_stats:
            print(f'      {"Context":<25s} {"Total GPUs":<13s} '
                  f'{"Allocated (avg/min/max)":<28s} {"Util% (avg)"}')
            for ctx, s in sorted(gpu_stats.items()):
                alloc_str = (f'{s["allocated_avg"]:.1f} / '
                             f'{s["allocated_min"]} / '
                             f'{s["allocated_max"]}')
                print(f'      {ctx:<25s} {s["total_gpus"]:<13d} '
                      f'{alloc_str:<28s} {s["util_pct_avg"]:.1f}%')
                for gpu_type, ts in sorted(
                        s.get('by_type', {}).items()):
                    t_alloc_str = (f'{ts["alloc_avg"]:.1f} / '
                                   f'{ts["alloc_min"]} / '
                                   f'{ts["alloc_max"]}')
                    print(f'        {gpu_type:<23s} '
                          f'{ts["total"]:<13d} '
                          f'{t_alloc_str:<28s} '
                          f'{ts["util_pct_avg"]:.1f}%')
        else:
            print(f'      (no GPU data)')

        # GPU-Hours Summary
        gpu_hrs = analyzer.gpu_hours_stats(hbs)
        print(f'\n    GPU-Hours Summary:')
        if gpu_hrs:
            print(f'      {"Context":<25s} {"Total GPU-Hrs":<16s} '
                  f'{"Allocated GPU-Hrs":<20s} '
                  f'{"Idle GPU-Hrs":<15s} {"Idle%"}')
            for ctx, s in sorted(gpu_hrs.items()):
                print(f'      {ctx:<25s} '
                      f'{s["total_gpu_hours"]:<16.1f} '
                      f'{s["allocated_gpu_hours"]:<20.1f} '
                      f'{s["idle_gpu_hours"]:<15.1f} '
                      f'{s["idle_pct"]:.1f}%')
                for gpu_type, ts in sorted(s['by_type'].items()):
                    print(f'        {gpu_type:<23s} '
                          f'{ts["total_hrs"]:<16.1f} '
                          f'{ts["alloc_hrs"]:<20.1f} '
                          f'{ts["idle_hrs"]:<15.1f}')
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
                    for gtype, gstats in sorted(
                            s.get('by_type', {}).items()):
                        print(
                            f'        {gtype:<23s} {"":18s} '
                            f'{"":26s} '
                            f'{gstats["gpu_hours"]:<10.1f} '
                            f'${gstats["roi"]:.2f}')
        else:
            print(f'      (no borrowing in period)')

        # Server summary totals
        total_alloc_hrs = sum(
            s['allocated_gpu_hours'] for s in gpu_hrs.values())
        total_idle_hrs = sum(
            s['idle_gpu_hours'] for s in gpu_hrs.values())
        print(f'\n    Total Allocated GPU-Hours: {total_alloc_hrs:.1f}')
        print(f'    Total Idle GPU-Hours: {total_idle_hrs:.1f}')
        print(f'    Total Borrowed GPU-Hours: {total_gpu_hrs:.1f}')
        range_hrs = _parse_range(range_str) / 3600.0
        yearly_roi = (total_roi / range_hrs * 8760
                      if range_hrs > 0 else 0)
        print(f'    Estimated ROI for {range_str}: ${total_roi:.2f}')
        print(f'    Estimated ROI per year: ${yearly_roi:,.2f}')

        # Kueue Queue Health
        qh_stats = analyzer.kueue_queue_health_stats(hbs)
        has_queues = any(
            bool(queues) for queues in qh_stats.values())
        print(f'\n    Kueue Queue Health:')
        if has_queues:
            print(f'      {"Context":<25s} {"Queue":<18s} '
                  f'{"Admitted (avg/max)":<22s} '
                  f'{"Pending (avg/max)":<20s} {"Preemptions"}')
            for ctx, queues in sorted(qh_stats.items()):
                for queue, s in sorted(queues.items()):
                    admitted_str = (f'{s["admitted_avg"]:.1f} / '
                                    f'{s["admitted_max"]}')
                    pending_str = (f'{s["pending_avg"]:.1f} / '
                                   f'{s["pending_max"]}')
                    print(f'      {ctx:<25s} {queue:<18s} '
                          f'{admitted_str:<22s} '
                          f'{pending_str:<20s} '
                          f'{s["preempted_total"]}')
        else:
            print(f'      (no queue data)')


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

            # Fleet-level plot at top level
            plot_fleet_overview(
                analyzer, servers,
                os.path.join(args.plot_dir, 'fleet_overview.png'))

            # Per-server subdirectories
            for sh, info in sorted(servers.items()):
                hostname = _sanitize_dirname(info['hostname'])
                subdir = os.path.join(args.plot_dir,
                                      f'{sh}_{hostname}')
                os.makedirs(subdir, exist_ok=True)
                hbs = info['heartbeats']

                print(f'  {sh}_{hostname}/')

                plot_gpu_allocation(
                    analyzer, hbs,
                    os.path.join(subdir, 'gpu_allocation.png'))

                ok = plot_kueue_borrowing(
                    analyzer, hbs,
                    os.path.join(subdir, 'kueue_borrowing.png'))
                if not ok:
                    print('    kueue_borrowing.png  '
                          '(skipped - no borrowing data)')

                plot_gpu_utilization_heatmap(
                    analyzer, hbs,
                    os.path.join(subdir, 'gpu_utilization.png'))

                ok = plot_kueue_queue_health(
                    analyzer, hbs,
                    os.path.join(subdir, 'kueue_queue_health.png'))
                if not ok:
                    print('    kueue_queue_health.png  '
                          '(skipped - no queue data)')

    # CSV
    if args.csv_output:
        print(f'\nCSVs saved to:')
        write_gpu_allocation_csv(analyzer, servers, args.csv_output)
        write_kueue_borrowing_csv(analyzer, servers, args.csv_output)
        write_gpu_hours_csv(analyzer, servers, args.csv_output)

    print()


if __name__ == '__main__':
    main()
