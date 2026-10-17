#!/usr/bin/env python3
"""Summarize actual training throughput and process DRM GPU activity."""
import argparse,json,statistics
from datetime import datetime
from pathlib import Path

def summarize(path):
    path=Path(path)
    def read(name,default):
        try:return json.loads((path/name).read_text())
        except (OSError,ValueError):return default
    manifest=read('manifest.json',{});status=read('status.json',{});metrics=read('metrics.json',[])
    rows=[]
    if (path/'telemetry.jsonl').exists():
        for line in (path/'telemetry.jsonl').read_text().splitlines():
            try:rows.append(json.loads(line))
            except ValueError:pass
    phases={}
    for row in rows:
        phase=(row.get('training_progress') or {}).get('phase','unknown')
        value=row.get('gpu_engine_utilization_percent')
        if value is not None:phases.setdefault(phase,[]).append(value)
    gpu={phase:dict(samples=len(v),mean_percent=statistics.mean(v),median_percent=statistics.median(v),max_percent=max(v)) for phase,v in phases.items()}
    duration=sum(r['seconds'] for r in metrics)
    active=[];elapsed=busy=0.;windows=[];window_time=window_busy=0.
    for prev,row in zip(rows,rows[1:]):
        phase=(row.get('training_progress') or {}).get('phase','unknown')
        value=row.get('gpu_engine_utilization_percent')
        dt=(datetime.fromisoformat(row['utc'])-datetime.fromisoformat(prev['utc'])).total_seconds()
        if value is None or phase in ('initializing','unknown','completed') or dt<=0:continue
        elapsed+=dt;busy+=dt*value;active.append(value)
        window_time+=dt;window_busy+=dt*value
        if window_time>=10:
            windows.append(window_busy/window_time);window_time=window_busy=0.
    steady=metrics[1:]
    steady_duration=sum(r['seconds'] for r in steady)

    return dict(output=str(path.resolve()),state=status.get('state'),command=manifest.get('command'),
                iterations_completed=len(metrics),samples_per_second=sum(r['samples'] for r in metrics)/duration if duration else None,
                mean_iteration_seconds=duration/len(metrics) if metrics else None,gpu_by_phase=gpu,
                steady_samples_per_second=sum(r['samples'] for r in steady)/steady_duration if steady_duration else None,
                gpu_active_mean_percent=busy/elapsed if elapsed else None,
                gpu_10s_window_median_percent=statistics.median(windows) if windows else None,
                gpu_10s_window_max_percent=max(windows) if windows else None,
                raw_windows_over_100=sum(v>100 for v in active),
                measurement_note='Process DRM engine-time deltas; deferred driver accounting can exceed 100% in short windows. Phase attribution is approximate. Use long-window means and throughput, not raw peaks.',
                iteration_timings=[{k:r[k] for k in ('iteration','seconds','samples','collection_seconds','learning_seconds')} for r in metrics])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('runs',nargs='+');p.add_argument('--output');a=p.parse_args()
    result=[summarize(path) for path in a.runs];text=json.dumps(result,indent=2)
    if a.output:Path(a.output).write_text(text+'\n')
    print(text)
