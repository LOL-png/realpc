#!/usr/bin/env python
"""Upload finalized v2, download the full payload, and verify its SHA256SUMS."""
import os,json,hashlib,subprocess,datetime
from pathlib import Path
OUT=Path('/home/howardji/data_ext/realpc_out/v2');RUN=Path('/home/howardji/data_ext/astra_meshes/dataset_v2');DEST='s3://plwm-eun1-howardji/data/agentwm_v2/'
AWS=['/home/howardji/venvs/awscli/bin/aws','--profile','nebius'];ENV=dict(os.environ,AWS_CONFIG_FILE='/home/howardji/.config/nebius-s3/config',AWS_SHARED_CREDENTIALS_FILE='/home/howardji/.config/nebius-s3/credentials')
assert len(list(OUT.glob('*/episode_*.npz')))==52 and (OUT/'SHA256SUMS').exists()
subprocess.run(['sha256sum','--check','--quiet','SHA256SUMS'],cwd=OUT,check=True)
subprocess.run(AWS+['s3','sync',str(OUT)+'/',DEST,'--no-progress','--only-show-errors'],env=ENV,check=True)
verify=RUN/'s3_verified_copy';verify.mkdir(exist_ok=True)
subprocess.run(AWS+['s3','sync',DEST,str(verify)+'/', '--no-progress','--only-show-errors'],env=ENV,check=True)
assert (OUT/'SHA256SUMS').read_bytes()==(verify/'SHA256SUMS').read_bytes()
subprocess.run(['sha256sum','--check','--quiet','SHA256SUMS'],cwd=verify,check=True)
local={str(p.relative_to(OUT)) for p in OUT.rglob('*') if p.is_file()};remote={str(p.relative_to(verify)) for p in verify.rglob('*') if p.is_file()};assert local==remote,(local-remote,remote-local)
receipt={'destination':DEST,'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'verification':'full download and SHA256SUMS check; identical file set','files':len(local),'episodes':52,'sha256sums_sha256':hashlib.sha256((OUT/'SHA256SUMS').read_bytes()).hexdigest()}
json.dump(receipt,open(RUN/'s3_transfer_verified.json','w'),indent=2);print(json.dumps(receipt))
