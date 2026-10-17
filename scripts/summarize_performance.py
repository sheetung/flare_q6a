#!/usr/bin/env python3
"""Summarize completed training runs; excludes first iteration from steady timing."""
import argparse,json,statistics
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('runs',nargs='+');ap.add_argument('--output',default='records/hardware_tuning_results.json')
a=ap.parse_args()
def read(path,default):
    try:return json.loads(path.read_text())
    except FileNotFoundError:return default
def lines(path):
    if not path.exists():return []
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
results=[]
for run in a.runs:
    p=Path(run).resolve();status=read(p/'status.json',{});config=read(p/'config.json',{})
    metrics=lines(p/'metrics.jsonl');steady=metrics[1:];tele=lines(p/'telemetry.jsonl')
    active=[r for r in tele if (r.get('training_progress') or {}).get('phase') not in (None,'initializing','completed')]
    util=[r['gpu_engine_utilization_percent'] for r in active if r.get('gpu_engine_utilization_percent') is not None]
    item={'output':str(p),'state':status.get('state'),'source_commit':status.get('git_commit'),
        'arguments':config.get('arguments'),'execution':config.get('execution'),
        'runtime_flags':{k:status.get(k) for k in ('ppo_batch_realize','ppo_jit','ppo_jit_adam','rollout_jit')},
        'iterations_completed':len(metrics),'iteration_timings':[{k:r[k] for k in ('iteration','samples','seconds','fps','collection_seconds','learning_seconds','updates')} for r in metrics],
        'steady_samples_per_second':sum(r['samples'] for r in steady)/sum(r['seconds'] for r in steady) if steady else None,
        'active_gpu_engine_mean_percent':statistics.mean(util) if util else None,
        'raw_gpu_windows_over_100':sum(v>100 for v in util),
        'peak_rss_bytes':max((r.get('process_rss_bytes',0) for r in tele),default=0),
        'peak_sensor_temperature_c':max((r.get('maximum_sensor_temperature_c',0) for r in tele),default=0)}
    checkpoint=p/'checkpoint.npz'
    if checkpoint.exists():
        with np.load(checkpoint,allow_pickle=False) as d:
            meta=json.loads(str(d['metadata']))
            item['checkpoint']={'iteration':meta['iteration'],'updates':meta['updates'],
                'sigma':d['model.std'].tolist(),'all_model_tensors_finite':all(np.isfinite(d[k]).all() for k in d.files if k.startswith('model.'))}
    results.append(item)
report={'measurement_note':'DRM process engine-time deltas, approximate phase attribution and short-window accounting; not global GPU utilization. Steady throughput excludes first iteration. Different runtime flags are distinct configurations; short benchmarks do not prove long-term stability or convergence.',
        'npu':'unused; current PPO backward/Adam backend is OpenCL GPU','runs':results}
Path(a.output).write_text(json.dumps(report,indent=2))
print(json.dumps([{'output':r['output'],'state':r['state'],'steady_samples_per_second':r['steady_samples_per_second']} for r in results],indent=2))
