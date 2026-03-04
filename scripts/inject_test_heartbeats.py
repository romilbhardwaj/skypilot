#!/usr/bin/env python3
"""Inject realistic mock server heartbeat data into Loki for testing.

Generates heartbeats for 3 mock servers with varied GPU allocation patterns,
Kueue borrowing dynamics, and edge cases to exercise every code path in
scripts/analyze_heartbeat_roi.py.

Usage:
    python scripts/inject_test_heartbeats.py \
        --loki-url http://54.202.123.249:9090 \
        --duration 24h \
        --interval 600 \
        --batch-size 100 \
        --dry-run

See scripts/heartbeat_schema_example.json for the heartbeat schema reference.
"""

import argparse
import json
import math
import random
import sys
import time
from typing import Dict, List, Optional

import requests

# ---------------------------------------------------------------------------
# Allocation pattern helpers
# ---------------------------------------------------------------------------


def sinusoidal_alloc(t: float, period: float, total: int, min_pct: float,
                     max_pct: float) -> int:
    """Business-hours sine wave: peaks mid-period, troughs at edges.

    Args:
        t: Current time position within a day (0..86400 seconds from midnight).
        period: Cycle length in seconds (86400 for daily).
        total: Max GPU count.
        min_pct: Minimum allocation fraction (0-1).
        max_pct: Maximum allocation fraction (0-1).
    """
    # Phase: peak at t/period = 0.5 (midday), trough at 0 and 1 (midnight)
    phase = (t % period) / period
    wave = 0.5 * (1 - math.cos(2 * math.pi * phase))  # 0..1
    pct = min_pct + wave * (max_pct - min_pct)
    # Add small noise (±5%)
    pct += random.uniform(-0.05, 0.05)
    pct = max(0.0, min(1.0, pct))
    return max(0, min(total, round(total * pct)))


def sporadic_alloc(total: int, probability: float) -> int:
    """Random on/off allocation — GPU used with given probability."""
    if random.random() < probability:
        return random.randint(1, total)
    return 0


def ramp_alloc(t: float, t_start: float, t_end: float, start_val: int,
               end_val: int) -> int:
    """Linear ramp from start_val to end_val over [t_start, t_end]."""
    if t <= t_start:
        return start_val
    if t >= t_end:
        return end_val
    frac = (t - t_start) / (t_end - t_start)
    return round(start_val + frac * (end_val - start_val))


def steady_alloc(total: int, min_pct: float, max_pct: float) -> int:
    """Steady allocation with small random noise."""
    pct = random.uniform(min_pct, max_pct)
    return max(0, min(total, round(total * pct)))


# ---------------------------------------------------------------------------
# Heartbeat construction
# ---------------------------------------------------------------------------


def make_billing_context(context_name: str,
                         gpu_types: Dict[str, Dict[str, int]],
                         num_nodes: int = 2,
                         total_cpus: float = 64.0,
                         alloc_cpus: float = 16.0,
                         total_mem_gb: float = 256.0,
                         alloc_mem_gb: float = 64.0) -> dict:
    """Build a billing context entry.

    Args:
        context_name: K8s context name.
        gpu_types: {gpu_type: {"total": N, "allocated": M}}
        num_nodes: Number of K8s nodes.
        total_cpus: Total CPU cores.
        alloc_cpus: Allocated CPU cores.
        total_mem_gb: Total memory in GB.
        alloc_mem_gb: Allocated memory in GB.
    """
    total = sum(v['total'] for v in gpu_types.values())
    allocated = sum(v['allocated'] for v in gpu_types.values())
    by_type = {}
    for gpu_type, counts in gpu_types.items():
        by_type[gpu_type] = {
            'total': counts['total'],
            'allocated': counts['allocated'],
        }

    return {
        'context_name': context_name,
        'infra_type': 'kubernetes',
        'nodes': {
            'total': num_nodes,
            'ready': num_nodes,
            'cordoned': 0,
        },
        'cpus': {
            'total': total_cpus,
            'allocated': alloc_cpus,
        },
        'memory_gb': {
            'total': total_mem_gb,
            'allocated': alloc_mem_gb,
        },
        'gpus': {
            'total': total,
            'allocated': allocated,
            'by_type': by_type,
        },
        'skypilot_clusters': max(1, allocated),
        'skypilot_pods': max(1, allocated),
        'cluster_resources': [],
    }


def make_kueue_context(context_name: str,
                        queues: List[dict],
                        kueue_enabled: bool = True) -> dict:
    """Build a kueue context entry.

    Args:
        context_name: K8s context name (matches billing context_name).
        queues: List of queue dicts with keys:
            name, admitted_workloads, num_pending_workloads,
            gpu_borrowed, gpu_total (for nvidia.com/gpu resource)
    """
    if not kueue_enabled:
        return {
            'context': context_name,
            'kueue_enabled': False,
        }

    cluster_queues = []
    for q in queues:
        num_pending = q.get('num_pending_workloads', 0)
        cluster_queues.append({
            'name': q['name'],
            'admitted_workloads': q.get('admitted_workloads', 0),
            'num_pending_workloads': num_pending,
            'flavors_reservation': [{
                'name':
                    'default-flavor',
                'resources': [
                    {
                        'name': 'cpu',
                        'total': str(q.get('cpu_total', 0)),
                        'borrowed': str(q.get('cpu_borrowed', 0)),
                    },
                    {
                        'name': 'memory',
                        'total': str(q.get('mem_total', 0)),
                        'borrowed': str(q.get('mem_borrowed', 0)),
                    },
                    {
                        'name': 'nvidia.com/gpu',
                        'total': str(q.get('gpu_total', 0)),
                        'borrowed': str(q.get('gpu_borrowed', 0)),
                    },
                ],
            }],
            'flavors_usage': [{
                'name':
                    'default-flavor',
                'resources': [
                    {
                        'name': 'cpu',
                        'total': str(q.get('cpu_total', 0)),
                        'borrowed': str(q.get('cpu_borrowed', 0)),
                    },
                    {
                        'name': 'memory',
                        'total': str(q.get('mem_total', 0)),
                        'borrowed': str(q.get('mem_borrowed', 0)),
                    },
                    {
                        'name': 'nvidia.com/gpu',
                        'total': str(q.get('gpu_total', 0)),
                        'borrowed': str(q.get('gpu_borrowed', 0)),
                    },
                ],
            }],
            'pending_workloads': [],
        })

    return {
        'context': context_name,
        'kueue_enabled': kueue_enabled,
        'cluster_queues': cluster_queues,
        'preempted_events': 0,
    }


def make_heartbeat(send_time_ns: int, server_hash: str, hostname: str,
                   release_name: Optional[str], sky_version: str,
                   billing_contexts: List[dict],
                   kueue_contexts: List[dict]) -> dict:
    """Construct a full heartbeat message dict."""
    return {
        'schema_version': 1,
        'start_time': send_time_ns,
        'send_time': send_time_ns,
        'interval_seconds': 600,
        'hostname': hostname,
        'release_name': release_name,
        'server_hash': server_hash,
        'sky_version': sky_version,
        'ingress_host': None,
        'plugins': {
            'billing': {
                'contexts': billing_contexts,
            },
            'kueue': {
                'contexts': kueue_contexts,
            },
        },
    }


# ---------------------------------------------------------------------------
# Server generators
# ---------------------------------------------------------------------------


def _time_of_day(ts_ns: int) -> float:
    """Seconds since midnight UTC for a nanosecond timestamp."""
    return (ts_ns / 1e9) % 86400


def _elapsed_frac(ts_ns: int, start_ns: int, end_ns: int) -> float:
    """Fraction of elapsed time from start to end (0..1)."""
    if end_ns <= start_ns:
        return 0.0
    return max(0.0, min(1.0, (ts_ns - start_ns) / (end_ns - start_ns)))


def generate_server_a(start_ns: int, end_ns: int,
                      interval: int) -> List[dict]:
    """Server A: 'prod-gke.company.com', release_name='skypilot-prod'.

    Large production cluster. Multi-GPU-type contexts, Kueue borrowing.
    - gke-prod-us-central: H100:32 + A100:16, sinusoidal, 2 kueue queues
    - gke-prod-eu-west: H100:16, steady 50-75%, 1 kueue queue
    """
    heartbeats = []
    ts = start_ns
    while ts <= end_ns:
        tod = _time_of_day(ts)

        # --- Context 1: gke-prod-us-central (H100:32 + A100:16) ---
        h100_alloc = sinusoidal_alloc(tod, 86400, 32, 0.20, 0.95)
        a100_alloc = sinusoidal_alloc(tod, 86400, 16, 0.20, 0.95)

        billing_ctx1 = make_billing_context('gke-prod-us-central', {
            'H100': {
                'total': 32,
                'allocated': h100_alloc
            },
            'A100': {
                'total': 16,
                'allocated': a100_alloc
            },
        })

        # Kueue: ml-training borrows 0-8 GPUs peaking during business hours
        ml_borrowed = sinusoidal_alloc(tod, 86400, 8, 0.0, 0.9)
        # inference: small occasional borrowing
        inf_borrowed = random.choice([0, 0, 0, 1, 2])

        kueue_ctx1 = make_kueue_context('gke-prod-us-central', [
            {
                'name': 'ml-training',
                'admitted_workloads': random.randint(2, 10),
                'num_pending_workloads': random.randint(0, 3),
                'gpu_borrowed': ml_borrowed,
                'gpu_total': h100_alloc + a100_alloc,
                'cpu_total': 64,
                'cpu_borrowed': ml_borrowed * 4,
            },
            {
                'name': 'inference',
                'admitted_workloads': random.randint(1, 5),
                'num_pending_workloads': random.randint(0, 2),
                'gpu_borrowed': inf_borrowed,
                'gpu_total': inf_borrowed + random.randint(0, 3),
                'cpu_total': 32,
                'cpu_borrowed': inf_borrowed * 2,
            },
        ])

        # --- Context 2: gke-prod-eu-west (H100:16) ---
        eu_alloc = steady_alloc(16, 0.50, 0.75)

        billing_ctx2 = make_billing_context('gke-prod-eu-west', {
            'H100': {
                'total': 16,
                'allocated': eu_alloc,
            },
        })

        eu_borrowed = random.choice([0, 0, 0, 0, 1, 2, 3, 4])
        kueue_ctx2 = make_kueue_context('gke-prod-eu-west', [{
            'name': 'default',
            'admitted_workloads': random.randint(0, 4),
            'num_pending_workloads': random.randint(0, 1),
            'gpu_borrowed': eu_borrowed,
            'gpu_total': eu_alloc,
            'cpu_total': 32,
            'cpu_borrowed': eu_borrowed * 2,
        }])

        hb = make_heartbeat(
            send_time_ns=ts,
            server_hash='aaa11111',
            hostname='prod-gke.company.com',
            release_name='skypilot-prod',
            sky_version='0.11.2',
            billing_contexts=[billing_ctx1, billing_ctx2],
            kueue_contexts=[kueue_ctx1, kueue_ctx2],
        )
        heartbeats.append(hb)
        ts += interval * int(1e9)

    return heartbeats


def generate_server_b(start_ns: int, end_ns: int,
                      interval: int) -> List[dict]:
    """Server B: 'dev-macbook.local', release_name=None.

    Dev laptop. Small cluster, zero-GPU context, no Kueue.
    - kind-dev: T4:2, sporadic
    - minikube: 0 GPUs (tests div-by-zero guard)
    """
    heartbeats = []
    ts = start_ns
    while ts <= end_ns:
        # --- Context 1: kind-dev (T4:2, sporadic) ---
        t4_alloc = sporadic_alloc(2, 0.3)

        billing_ctx1 = make_billing_context('kind-dev', {
            'T4': {
                'total': 2,
                'allocated': t4_alloc,
            },
        })

        # --- Context 2: minikube (0 GPUs — zero-GPU context) ---
        billing_ctx2 = make_billing_context('minikube', {})

        # No kueue on dev laptop — kueue_enabled=False (no cluster_queues)
        kueue_ctx1 = make_kueue_context('kind-dev', [],
                                         kueue_enabled=False)
        kueue_ctx2 = make_kueue_context('minikube', [],
                                         kueue_enabled=False)
        hb = make_heartbeat(
            send_time_ns=ts,
            server_hash='bbb22222',
            hostname='dev-macbook.local',
            release_name=None,
            sky_version='0.11.1',
            billing_contexts=[billing_ctx1, billing_ctx2],
            kueue_contexts=[kueue_ctx1, kueue_ctx2],
        )
        heartbeats.append(hb)
        ts += interval * int(1e9)

    return heartbeats


def generate_server_c(start_ns: int, end_ns: int,
                      interval: int) -> List[dict]:
    """Server C: 'ml-server.internal', release_name='skypilot-ml'.

    Medium cluster with interesting Kueue dynamics.
    - coreweave-ml: H200:8 (scales to 16 halfway), ramp-up, 3 queues
    - nebius-dev: L40S:4, low util, 1 queue with zero borrowing
    """
    heartbeats = []
    ts = start_ns
    mid_ns = start_ns + (end_ns - start_ns) // 2
    while ts <= end_ns:
        frac = _elapsed_frac(ts, start_ns, end_ns)
        tod = _time_of_day(ts)

        # --- Context 1: coreweave-ml (H200:8->16, ramp up) ---
        h200_total = 8 if ts < mid_ns else 16
        # Ramp allocation from ~10% to ~80% over the time window
        h200_alloc = ramp_alloc(frac, 0.0, 1.0, 1,
                                round(h200_total * 0.8))
        # Add noise
        h200_alloc = max(0,
                         min(h200_total, h200_alloc + random.randint(-1, 1)))

        billing_ctx1 = make_billing_context('coreweave-ml', {
            'H200': {
                'total': h200_total,
                'allocated': h200_alloc,
            },
        })

        # 3 Kueue queues: batch-jobs has heavy borrowing spikes
        # Burst windows: borrowing spikes every ~4 hours for ~1 hour
        is_burst = (tod % 14400) < 3600  # 1h burst every 4h
        batch_borrowed = random.randint(4, 12) if is_burst else random.choice(
            [0, 0, 0, 1, 2])
        interactive_borrowed = random.choice([0, 0, 1])
        priority_borrowed = random.choice([0, 0, 0, 0, 1])

        kueue_ctx1 = make_kueue_context('coreweave-ml', [
            {
                'name': 'batch-jobs',
                'admitted_workloads': random.randint(1, 8),
                'pending_workloads':
                    random.randint(2, 6) if is_burst else random.randint(0, 1),
                'gpu_borrowed': batch_borrowed,
                'gpu_total': h200_alloc,
                'cpu_total': 48,
                'cpu_borrowed': batch_borrowed * 4,
            },
            {
                'name': 'interactive',
                'admitted_workloads': random.randint(0, 3),
                'num_pending_workloads': 0,
                'gpu_borrowed': interactive_borrowed,
                'gpu_total': interactive_borrowed,
                'cpu_total': 16,
                'cpu_borrowed': interactive_borrowed * 2,
            },
            {
                'name': 'priority',
                'admitted_workloads': random.randint(0, 2),
                'num_pending_workloads': 0,
                'gpu_borrowed': priority_borrowed,
                'gpu_total': priority_borrowed,
                'cpu_total': 8,
                'cpu_borrowed': priority_borrowed * 2,
            },
        ])

        # --- Context 2: nebius-dev (L40S:4, low util, zero borrowing) ---
        l40s_alloc = random.choice([0, 0, 0, 1])

        billing_ctx2 = make_billing_context('nebius-dev', {
            'L40S': {
                'total': 4,
                'allocated': l40s_alloc,
            },
        })

        kueue_ctx2 = make_kueue_context('nebius-dev', [{
            'name': 'default',
            'admitted_workloads': random.randint(0, 1),
            'num_pending_workloads': 0,
            'gpu_borrowed': 0,
            'gpu_total': l40s_alloc,
            'cpu_total': 8,
            'cpu_borrowed': 0,
        }])

        hb = make_heartbeat(
            send_time_ns=ts,
            server_hash='ccc33333',
            hostname='ml-server.internal',
            release_name='skypilot-ml',
            sky_version='0.11.2',
            billing_contexts=[billing_ctx1, billing_ctx2],
            kueue_contexts=[kueue_ctx1, kueue_ctx2],
        )
        heartbeats.append(hb)
        ts += interval * int(1e9)

    return heartbeats


# ---------------------------------------------------------------------------
# Loki push
# ---------------------------------------------------------------------------


def push_to_loki(loki_url: str, heartbeats: List[dict],
                 batch_size: int) -> int:
    """Push heartbeats to Loki in batches. Returns count pushed.

    Heartbeats are sorted by timestamp before pushing, since Loki
    requires entries within a stream to have monotonically increasing
    timestamps.
    """
    url = f'{loki_url.rstrip("/")}/loki/api/v1/push'
    headers = {'Content-Type': 'application/json'}
    pushed = 0

    # Sort all heartbeats by send_time — Loki requires monotonic order
    sorted_hbs = sorted(heartbeats, key=lambda hb: hb['send_time'])

    for i in range(0, len(sorted_hbs), batch_size):
        batch = sorted_hbs[i:i + batch_size]
        values = []
        for hb in batch:
            ts_str = str(hb['send_time'])
            line = json.dumps(hb)
            values.append([ts_str, line])

        payload = {
            'streams': [{
                'stream': {
                    'type': 'server_heartbeat',
                    'environment': 'dev',
                    'schema_version': '1',
                },
                'values': values,
            }]
        }

        resp = requests.post(url,
                             data=json.dumps(payload),
                             headers=headers,
                             timeout=30)

        if resp.status_code == 204:
            pushed += len(batch)
        elif resp.status_code == 400 and 'too far behind' in resp.text:
            # Loki rejects entries older than its ingestion window.
            # Count how many in the batch are within the window.
            print(f'  WARN: Batch {i // batch_size} has entries older '
                  f'than Loki ingestion window, skipping old entries.')
        else:
            print(f'  ERROR: Loki returned {resp.status_code}: '
                  f'{resp.text[:200]}')

        if (i // batch_size) % 10 == 0 and i > 0:
            print(f'  Pushed {pushed}/{len(sorted_hbs)} heartbeats...')

    return pushed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_duration(s: str) -> float:
    """Parse duration like '24h', '7d', '30d' to seconds."""
    s = s.strip().lower()
    if s.endswith('d'):
        return float(s[:-1]) * 86400
    elif s.endswith('h'):
        return float(s[:-1]) * 3600
    elif s.endswith('m'):
        return float(s[:-1]) * 60
    else:
        raise ValueError(f'Invalid duration: {s}. Use e.g. 24h, 7d, 30d')


def main():
    parser = argparse.ArgumentParser(
        description='Inject test heartbeat data into Loki')
    parser.add_argument('--loki-url',
                        required=True,
                        help='Loki push URL (e.g. http://host:9090)')
    parser.add_argument('--duration',
                        default='24h',
                        help='Duration of data to generate (24h, 7d, 30d)')
    parser.add_argument('--interval',
                        type=int,
                        default=600,
                        help='Seconds between heartbeats (default: 600)')
    parser.add_argument('--batch-size',
                        type=int,
                        default=100,
                        help='Heartbeats per Loki push request')
    parser.add_argument('--dry-run',
                        action='store_true',
                        help='Print stats without pushing to Loki')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='Random seed for reproducibility')
    parser.add_argument(
        '--compress-time',
        default=None,
        help='Compress all timestamps into the last N minutes to '
        'fit within Loki ingestion window. E.g. "50m" compresses '
        '24h of pattern variation into the last 50 minutes. '
        'Heartbeat send_time fields are remapped but '
        'interval_seconds stays at --interval for correct ROI.')
    args = parser.parse_args()

    random.seed(args.seed)

    duration_sec = parse_duration(args.duration)
    end_ns = int(time.time() * 1e9)
    start_ns = end_ns - int(duration_sec * 1e9)

    print(f'Generating test heartbeats:')
    print(f'  Duration: {args.duration} ({duration_sec:.0f}s)')
    print(f'  Interval: {args.interval}s')
    print(f'  Time range: {time.strftime("%Y-%m-%d %H:%M", time.gmtime(start_ns / 1e9))} '
          f'to {time.strftime("%Y-%m-%d %H:%M", time.gmtime(end_ns / 1e9))} UTC')

    # Generate heartbeats for all 3 servers
    print('\nGenerating Server A (prod-gke.company.com, H100+A100, Kueue)...')
    hb_a = generate_server_a(start_ns, end_ns, args.interval)
    print(f'  {len(hb_a)} heartbeats')

    print('Generating Server B (dev-macbook.local, T4+zero-GPU, no Kueue)...')
    hb_b = generate_server_b(start_ns, end_ns, args.interval)
    print(f'  {len(hb_b)} heartbeats')

    print('Generating Server C (ml-server.internal, H200 scale-up, 3 queues)...')
    hb_c = generate_server_c(start_ns, end_ns, args.interval)
    print(f'  {len(hb_c)} heartbeats')

    all_hbs = hb_a + hb_b + hb_c
    total = len(all_hbs)

    print(f'\nTotal: {total} heartbeats '
          f'({len(hb_a)} + {len(hb_b)} + {len(hb_c)})')

    # Compress timestamps if requested
    if args.compress_time:
        compress_sec = parse_duration(args.compress_time)
        compress_end_ns = int(time.time() * 1e9)
        compress_start_ns = compress_end_ns - int(compress_sec * 1e9)

        # Map [start_ns, end_ns] -> [compress_start_ns, compress_end_ns]
        span = end_ns - start_ns
        if span > 0:
            for hb in all_hbs:
                old_ts = hb['send_time']
                frac = (old_ts - start_ns) / span
                new_ts = int(compress_start_ns + frac *
                             (compress_end_ns - compress_start_ns))
                hb['send_time'] = new_ts
                hb['start_time'] = new_ts

        print(f'\n  Compressed timestamps: {args.duration} -> '
              f'{args.compress_time} (Loki timestamps remapped, '
              f'interval_seconds unchanged at {args.interval}s)')

    # Print summary of what each server tests
    print('\nTest coverage:')
    print('  Server A (aaa11111): multi-GPU-type, Kueue borrowing, '
          'release_name set')
    print('  Server B (bbb22222): sporadic alloc, zero-GPU context, '
          'no Kueue, release_name=None')
    print('  Server C (ccc33333): scale-up event, burst borrowing, '
          '3 queues, zero-borrowing queue')

    if args.dry_run:
        print('\n[DRY RUN] Would push to:', args.loki_url)
        # Print one sample heartbeat
        print('\nSample heartbeat (Server A):')
        print(json.dumps(hb_a[len(hb_a) // 2], indent=2)[:2000])
        return

    print(f'\nPushing to {args.loki_url}...')
    pushed = push_to_loki(args.loki_url, all_hbs, args.batch_size)
    print(f'\nDone: {pushed}/{total} heartbeats pushed successfully.')

    if pushed < total:
        print(f'WARNING: {total - pushed} heartbeats failed to push.',
              file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
