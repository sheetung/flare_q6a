#!/usr/bin/env python3
"""Compare completed same-configuration runs, excluding each first iteration."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--qcom-run',required=True,type=Path)
parser.add_argument('--rusticl-run',required=True,type=Path)
parser.add_argument('--output',type=Path,default=Path('records/opencl_driver_comparison.json'))
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
subprocess.run([sys.executable,str(root/'scripts/summarize_performance.py'),str(args.qcom_run),str(args.rusticl_run),'--output',str(args.output)],check=True,capture_output=True,text=True)
report=json.loads(args.output.read_text())
configs=[json.loads((p/'config.json').read_text()) for p in [args.qcom_run,args.rusticl_run]]
manifests=[json.loads((p/'manifest.json').read_text()) for p in [args.qcom_run,args.rusticl_run]]
checks={key:configs[0][key]==configs[1][key] for key in ['algorithm','environment','task_signature','num_obs','upstream_train_config','random_initial_episode_ages']}
checks['runtime_source_hashes_identical']=manifests[0]['source_sha256']==manifests[1]['source_sha256']
for key in ['envs','steps','iterations','physics','exp_name','timeout_bootstrap','epochs','minibatches','microbatch_size']:
    checks['argument_'+key]=configs[0]['arguments'][key]==configs[1]['arguments'][key]
for key in ['microbatch_size','inference_chunk_size','row_lookup','cpu_staging_threads','jit','jit_adam','rollout_jit','batch_realize','host_dispatch']:
    checks['execution_'+key]=configs[0]['execution'][key]==configs[1]['execution'][key]
assert all(checks.values()),checks
summary={}
for label,path,item in zip(['qcom','rusticl'],[args.qcom_run,args.rusticl_run],report['runs']):
    assert item['state']=='completed'
    rows=[json.loads(s) for s in (path/'metrics.jsonl').read_text().splitlines()]
    assert len(rows)>=3
    steady=rows[1:]
    telemetry=[json.loads(s) for s in (path/'telemetry.jsonl').read_text().splitlines()]
    with np.load(path/'checkpoint.npz',allow_pickle=False) as checkpoint:
        numeric=[checkpoint[k] for k in checkpoint.files if checkpoint[k].dtype.kind in 'fc']
        assert all(np.isfinite(x).all() for x in numeric)
        assert all(x.dtype==np.float32 for x in numeric)
    assert item['checkpoint']['all_model_tensors_finite'] and min(item['checkpoint']['sigma'])>0
    summary[label]={
        'hardware':configs[['qcom','rusticl'].index(label)]['execution']['hardware'],
        'iterations':len(rows),'adam_updates':item['checkpoint']['updates'],
        'samples_per_second':item['steady_samples_per_second'],
        'mean_collection_seconds':sum(x['collection_seconds'] for x in steady)/len(steady),
        'mean_learning_seconds':sum(x['learning_seconds'] for x in steady)/len(steady),
        'min_system_available_gib':min(x['system_memory_available_bytes'] for x in telemetry if x.get('system_memory_available_bytes') is not None)/1024**3,
        'max_sensor_temperature_c':item['peak_sensor_temperature_c'],
        'checkpoint_numeric_arrays_finite_float32':True,
    }
summary['qcom_over_rusticl_throughput']=summary['qcom']['samples_per_second']/summary['rusticl']['samples_per_second']
summary['qcom_learning_speedup']=summary['rusticl']['mean_learning_seconds']/summary['qcom']['mean_learning_seconds']
report.update(status='PASS',controlled_configuration_checks=checks,comparison=summary,
    scope='single sequential pair, three iterations each; excludes compilation first iteration; not repeated trials, convergence or long-term stability',
    system_driver_replaced=False)
args.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(summary,indent=2))
