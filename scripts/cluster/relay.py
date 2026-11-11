#!/usr/bin/env python3
"""Transfer-only local relay. No GPU, extraction, AWS listing or Slurm submission.

Use SSH agent/key authentication, or --rsh 'sshpass -e ssh' with SSHPASS supplied
by a private launcher. Secrets are never accepted on this script's command line.
"""
import argparse,hashlib,json,os,shlex,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--remote',required=True)
p.add_argument('--run-root',required=True)
p.add_argument('--dataset',required=True)
p.add_argument('--archive',required=True)
p.add_argument('--recipient',required=True,help='Tailscale device name followed by a colon')
p.add_argument('--rsh',default='ssh')
p.add_argument('--interval',type=int,default=300)
a=p.parse_args();root=Path(a.archive);root.mkdir(parents=True,exist_ok=True)
ledger=root/'delivery.json';sent=json.loads(ledger.read_text()) if ledger.exists() else {}
def save():
 t=ledger.with_suffix('.tmp');t.write_text(json.dumps(sent,indent=2));t.replace(ledger)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def pull(remote,destination):
 destination.mkdir(parents=True,exist_ok=True)
 subprocess.run(['rsync','-rt','--partial','--exclude=*.tmp*','--timeout=120','-e',a.rsh,
                 a.remote+':'+remote+'/',str(destination)+'/'],check=True,stdout=subprocess.DEVNULL)
while True:
 try:
  for name in ('markers','reports','logs'):
   pull(a.run_root+'/'+name,root/name)
  result=subprocess.run(shlex.split(a.rsh)+[a.remote,'cat '+shlex.quote(a.run_root+'/status.json')],capture_output=True,text=True,check=True)
  status=json.loads(result.stdout);(root/'status.json').write_text(json.dumps(status,indent=2))
  for marker in sorted((root/'markers').glob('*.json')):
   row=json.loads(marker.read_text())
   for remote,expected in row.get('output_sha256',{}).items():
    if not remote.endswith('.mp4'):continue
    relative=Path(remote).relative_to(a.run_root);video=root/relative
    if not video.exists() or sha(video)!=expected:continue
    if sent.get(str(relative))==expected:continue
    # Taildrop basenames must identify the episode, not all be both_cameras.mp4.
    delivery=root/'delivery'/('_'.join(relative.parts[1:]));delivery.parent.mkdir(exist_ok=True)
    import shutil
    shutil.copy2(video,delivery)
    result=subprocess.run(['tailscale','file','cp',str(delivery),a.recipient],timeout=180)
    if result.returncode==0:
     sent[str(relative)]=expected;save();print('DELIVERED',relative,flush=True)
  if status.get('state')=='complete':
   pull(a.dataset,root/'dataset_v2')
   checks=root/'dataset_v2/SHA256SUMS'
   for line in checks.read_text().splitlines():
    digest,relative=line.split(maxsplit=1);f=root/'dataset_v2'/relative.lstrip('* ')
    assert f.resolve().is_relative_to((root/'dataset_v2').resolve())
    assert sha(f)==digest, f'Archive checksum mismatch: {relative}'
   (root/'ARCHIVE_VERIFIED.json').write_text(json.dumps(dict(time=time.time(),dataset=a.dataset,files=len(checks.read_text().splitlines())),indent=2))
   print('ARCHIVE VERIFIED; Orion files preserved',flush=True)
   if all(sent.get(str(Path(remote).relative_to(a.run_root)))==digest
          for marker in (root/'markers').glob('*.json')
          for remote,digest in json.loads(marker.read_text()).get('output_sha256',{}).items()
          if remote.endswith('.mp4')):break
 except Exception as error:
  print(type(error).__name__,str(error),flush=True)
 time.sleep(a.interval)
