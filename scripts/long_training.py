#!/usr/bin/env python3
"""Supervise one bounded training run and persist resource telemetry."""
import argparse,json,os,subprocess,time,signal,sys,threading
from datetime import datetime,timezone
from pathlib import Path
import psutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'flare_q6a'))
from task_config import RIGID_TASK,FLEX_TASK,DEFAULT_ENVS,DEFAULT_ITERATIONS,train_config

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

def mirror_log(path,stop):
    """Follow the saved log without delaying resource sampling or child output."""
    with path.open('r') as source:
        while True:
            chunk=source.read(8192)
            if chunk:
                try:
                    sys.stdout.write(chunk);sys.stdout.flush()
                except (BrokenPipeError,OSError):return
            elif stop.is_set():return
            else:stop.wait(0.1)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',help='optional run directory; default: records/run_<UTC timestamp>');ap.add_argument('-B','--num_envs','--envs',dest='envs',type=int,default=DEFAULT_ENVS)
    ap.add_argument('--steps',type=int,default=train_config()['runner']['num_steps_per_env']);ap.add_argument('--max_iterations','--iterations',dest='iterations',type=int,default=DEFAULT_ITERATIONS)
    ap.add_argument('--microbatch-size',type=int,default=100,help='GPU compute chunk; preserves logical PPO minibatches')
    ap.add_argument('--interval',type=float,default=5);ap.add_argument('--resume');ap.add_argument('--stall-timeout',type=float,default=600)
    ap.add_argument('-e','--exp_name',choices=[FLEX_TASK,RIGID_TASK],default=RIGID_TASK)
    ap.add_argument('-v','--vis',action='store_true',help='show original Genesis viewer and target')
    a=ap.parse_args()
    out=Path(a.output).resolve() if a.output else ROOT/'records'/datetime.now(timezone.utc).strftime('run_'+a.exp_name+'_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=True)
    print(f'Training output: {out}',flush=True)
    if (out/'status.json').exists():raise RuntimeError('Existing supervised run directory; use a new output directory')
    cmd=[str(ROOT/'scripts/run.sh'),'flare_q6a/train.py','--physics','genesis','--envs',str(a.envs),
         '--steps',str(a.steps),'--iterations',str(a.iterations),'--output',str(out)]
    cmd+=['--exp_name',a.exp_name,'--microbatch-size',str(a.microbatch_size)]
    if a.vis:cmd+=['--vis']
    if a.resume:cmd+=['--resume',str(Path(a.resume).resolve())]
    try:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    except subprocess.CalledProcessError:commit=None
    manifest={'started_utc':utc(),'git_commit':commit,'command':cmd,'samples_requested':a.envs*a.steps*a.iterations,
        'stall_timeout_seconds':a.stall_timeout,'monitor_interval_seconds':a.interval,'ppo_batch_realize':os.environ.get('RL_PPO_BATCH_REALIZE','0'),'ppo_jit':os.environ.get('RL_PPO_JIT','0'),'npu':'not used by this training pipeline',
        'gpu_utilization':'DRM engine-time delta when available, otherwise null; frequency is not utilization'}
    write(out/'manifest.json',manifest)
    with (out/'training.log').open('a',buffering=1) as log, (out/'telemetry.jsonl').open('a',buffering=1) as tele:
        child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        mirror_stop=threading.Event()
        mirror=threading.Thread(target=mirror_log,args=(out/'training.log',mirror_stop),daemon=True)
        mirror.start()
        proc=psutil.Process(child.pid);proc.cpu_percent()
        status={'state':'running','training_pid':child.pid,'supervisor_pid':os.getpid(),**manifest}
        write(out/'status.json',status);previous={};prev_time=time.monotonic();last_progress=None;last_stamp=None;last_advance=time.monotonic()
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
                stamp=last_progress.get('updated_unix') if last_progress else None
                if stamp!=last_stamp:last_stamp,last_advance=stamp,now
                row={'utc':utc(),'pid':child.pid,'process_cpu_percent_one_core_basis':cpu,
                     'process_cpu_percent_total_capacity':cpu/psutil.cpu_count() if cpu is not None else None,
                     'process_rss_bytes':rss,'threads':threads,'system_memory_available_bytes':vm.available,
                     'system_swap_used_bytes':swap.used,'gpu_frequency_hz':int(freq) if freq else None,
                     'gpu_engine_utilization_percent':util,'gpu_drm_counters_available':bool(counters),
                     'temperatures_c':temps,'maximum_sensor_temperature_c':max(temps.values()) if temps else None,
                     'npu_used_by_training':False,'npu_utilization_percent':None,'training_progress':last_progress}
                tele.write(json.dumps(row)+'\n');write(out/'latest_telemetry.json',row)
                if a.stall_timeout>0 and now-last_advance>a.stall_timeout:
                    try:os.kill(child.pid,signal.SIGUSR1)
                    except ProcessLookupError:pass
                    time.sleep(1)
                    child.terminate()
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                    status.update(state='failed',reason='training progress timeout',exit_code=124,ended_utc=utc())
                    write(out/'status.json',status);return 124
                code=child.poll()
                if code is not None:
                    status.update(state='completed' if code==0 else 'failed',exit_code=code,ended_utc=utc())
                    write(out/'status.json',status);return code
                time.sleep(a.interval)
        except BaseException as exc:
            status.update(state='supervisor_error',error=repr(exc),updated_utc=utc());write(out/'status.json',status)
            if child.poll() is None:child.terminate();child.wait(timeout=30)
            raise
        finally:
            mirror_stop.set();mirror.join(timeout=2)
if __name__=='__main__':raise SystemExit(main())
