#!/usr/bin/env python3
"""Supervise one bounded training run and persist resource telemetry."""
import argparse,json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
import psutil

ROOT=Path(__file__).resolve().parents[1]
def utc():return datetime.now(timezone.utc).isoformat()
def write(path,data):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2));os.replace(temp,path)
def read(path):
    try:return Path(path).read_text().strip()
    except (OSError,ValueError):return None
def drm(pid):
    clients={}
    for f in Path(f'/proc/{pid}/fdinfo').glob('*'):
        text=read(f)
        if not text:continue
        fields=dict(line.split(':',1) for line in text.splitlines() if ':' in line)
        if 'drm-client-id' not in fields:continue
        client=fields['drm-client-id'].strip();engines={}
        for name,val in fields.items():
            if name.startswith('drm-engine-'):
                parts=val.split()
                if len(parts)==2 and parts[1]=='ns':engines[name]=int(parts[0])
        if engines:clients[client]=engines
    return clients

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--envs',type=int,default=8)
    ap.add_argument('--steps',type=int,default=100);ap.add_argument('--iterations',type=int,default=200)
    ap.add_argument('--interval',type=float,default=5);ap.add_argument('--resume');a=ap.parse_args()
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'status.json').exists():raise RuntimeError('Existing supervised run directory; use a new output directory')
    cmd=[str(ROOT/'scripts/run.sh'),'flare_q6a/train.py','--physics','genesis','--envs',str(a.envs),
         '--steps',str(a.steps),'--iterations',str(a.iterations),'--output',str(out)]
    if a.resume:cmd+=['--resume',str(Path(a.resume).resolve())]
    try:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    except subprocess.CalledProcessError:commit=None
    manifest={'started_utc':utc(),'git_commit':commit,'command':cmd,'samples_requested':a.envs*a.steps*a.iterations,
        'monitor_interval_seconds':a.interval,'npu':'not used by this training pipeline',
        'gpu_utilization':'DRM engine-time delta when available, otherwise null; frequency is not utilization'}
    write(out/'manifest.json',manifest)
    with (out/'training.log').open('a',buffering=1) as log, (out/'telemetry.jsonl').open('a',buffering=1) as tele:
        child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        proc=psutil.Process(child.pid);proc.cpu_percent()
        status={'state':'running','training_pid':child.pid,'supervisor_pid':os.getpid(),**manifest}
        write(out/'status.json',status);previous={};prev_time=time.monotonic();last_progress=None
        try:
            while True:
                now=time.monotonic();counters=drm(child.pid);util=None
                if counters and previous and now>prev_time:
                    delta=0;valid=False
                    for client,engines in counters.items():
                        for engine,value in engines.items():
                            old=previous.get(client,{}).get(engine)
                            if old is not None and value>=old:delta+=value-old;valid=True
                    if valid:util=100*delta/((now-prev_time)*1e9)
                previous,prev_time=counters,now
                try:
                    cpu=proc.cpu_percent();mem=proc.memory_info();rss=mem.rss;threads=proc.num_threads()
                except psutil.Error:cpu=rss=threads=None
                vm=psutil.virtual_memory();swap=psutil.swap_memory()
                temps={}
                for zone in Path('/sys/class/thermal').glob('thermal_zone*'):
                    value=read(zone/'temp');name=read(zone/'type') or zone.name
                    try:temps[name]=float(value)/1000
                    except (TypeError,ValueError):pass
                freq=read('/sys/class/devfreq/3d00000.gpu/cur_freq')
                try:last_progress=json.loads((out/'progress.json').read_text())
                except (OSError,ValueError):pass
                row={'utc':utc(),'pid':child.pid,'process_cpu_percent_one_core_basis':cpu,
                     'process_cpu_percent_total_capacity':cpu/psutil.cpu_count() if cpu is not None else None,
                     'process_rss_bytes':rss,'threads':threads,'system_memory_available_bytes':vm.available,
                     'system_swap_used_bytes':swap.used,'gpu_frequency_hz':int(freq) if freq else None,
                     'gpu_engine_utilization_percent':util,'gpu_drm_counters_available':bool(counters),
                     'temperatures_c':temps,'maximum_sensor_temperature_c':max(temps.values()) if temps else None,
                     'npu_used_by_training':False,'npu_utilization_percent':None,'training_progress':last_progress}
                tele.write(json.dumps(row)+'\n');write(out/'latest_telemetry.json',row)
                code=child.poll()
                if code is not None:
                    status.update(state='completed' if code==0 else 'failed',exit_code=code,ended_utc=utc())
                    write(out/'status.json',status);return code
                time.sleep(a.interval)
        except BaseException as exc:
            status.update(state='supervisor_error',error=repr(exc),updated_utc=utc());write(out/'status.json',status)
            if child.poll() is None:child.terminate();child.wait(timeout=30)
            raise
if __name__=='__main__':raise SystemExit(main())
