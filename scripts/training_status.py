#!/usr/bin/env python3
"""Read-only compact status of a supervised training run."""
import argparse,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def display(out):
    try:s=json.loads((out/'status.json').read_text());t=json.loads((out/'latest_telemetry.json').read_text())
    except (OSError,ValueError) as e:print(f'Waiting for monitor files: {e}');return
    p=t.get('training_progress') or {};temps=t.get('temperatures_c',{})
    def temp(prefix):
        a=[v for k,v in temps.items() if k.startswith(prefix)]
        return f'{max(a):.1f} C' if a else 'N/A'
    util=t.get('gpu_engine_utilization_percent');freq=t.get('gpu_frequency_hz');rss=t.get('process_rss_bytes');cpu=t.get('process_cpu_percent_one_core_basis')
    print('UTC:',t['utc'],' state:',s['state'],' PID:',s['training_pid'])
    print('Phase:',p.get('phase','initializing'),' iteration:',p.get('iteration','-'),' step:',p.get('step','-'),'/',p.get('steps','-'))
    print('GPU process engine:',f'{util:.1f}%' if util is not None else 'unavailable',
          ' frequency:',f'{freq/1e6:.0f} MHz' if freq is not None else 'N/A',' GPU sensors:',temp('gpuss'))
    print('CPU process:',f'{cpu:.1f}% (100% = one core)' if cpu is not None else 'N/A',
          ' RSS:',f'{rss/1024**3:.2f} GiB' if rss is not None else 'N/A',' CPU sensors:',temp('cpu'))
    print('NPU: not used by this training; utilization not measured')
    print('Logs:',out/'training.log')
    print(flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--watch',action='store_true');a=p.parse_args()
    out=Path(a.output).resolve()
    while True:
        display(out)
        if not a.watch:break
        time.sleep(5)
if __name__=='__main__':main()
