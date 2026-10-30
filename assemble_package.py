"""Reassemble SHA-verified model volumes using only the Python standard library."""
from pathlib import Path
from zipfile import ZipFile
import hashlib,json,shutil,sys
ROOT=Path(__file__).resolve().parent

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def merge_archive(item,source,temporary):
 temporary=Path(temporary);temporary.mkdir(parents=True,exist_ok=True);target=temporary/item['name'];pending=target.with_name(target.name+'.partial')
 if target.exists() and sha(target)==item['sha256']:return target
 h=hashlib.sha256();total=0
 with pending.open('wb') as output:
  for part in item['parts']:
   path=(Path(source)/part['path']).resolve()
   if not path.is_relative_to(Path(source).resolve()) or not path.is_file():raise RuntimeError('Missing volume: '+part['path'])
   ph=hashlib.sha256();size=0
   with path.open('rb') as f:
    for data in iter(lambda:f.read(1<<20),b''):output.write(data);h.update(data);ph.update(data);size+=len(data);total+=len(data)
   if size!=part['bytes'] or ph.hexdigest()!=part['sha256']:raise RuntimeError('Volume checksum mismatch: '+part['path'])
 if total!=item['bytes'] or h.hexdigest()!=item['sha256']:raise RuntimeError('Combined archive checksum mismatch: '+item['name'])
 pending.replace(target);return target

def extract_archive(archive,destination):
 destination=Path(destination).resolve()
 with ZipFile(archive) as z:
  for info in z.infolist():
   path=(destination/info.filename).resolve()
   if not path.is_relative_to(destination):raise RuntimeError('Unsafe archive entry: '+info.filename)
  bad=z.testzip()
  if bad:raise RuntimeError('ZIP CRC failure: '+bad)
  z.extractall(destination)

def verify_package(package):
 package=Path(package).resolve();manifest=json.loads((package/'PACKAGE_MANIFEST.json').read_text('utf-8'))
 for name,digest in manifest['files'].items():
  p=(package/name).resolve()
  if not p.is_relative_to(package) or not p.is_file() or sha(p)!=digest:raise RuntimeError('Extracted file checksum failed: '+name)
 return manifest

def main():
 manifestpath=ROOT/'transfer_manifest.json';raw=manifestpath.read_bytes();manifest=json.loads(raw);fingerprint=hashlib.sha256(raw).hexdigest();output=ROOT/'assembled';output.mkdir(exist_ok=True);statepath=output/'.rank16_assembly_state.json'
 if statepath.exists():
  state=json.loads(statepath.read_text());assert state['manifest_sha256']==fingerprint,'Different package already present'
 else:
  if any(output.iterdir()):raise RuntimeError('Use an empty assembled folder to preserve existing files.')
  state=dict(manifest_sha256=fingerprint,archives_complete=[]);statepath.write_text(json.dumps(state,indent=2))
 temp=output/'.volumes';temp.mkdir(exist_ok=True)
 for number,item in enumerate(manifest['archives'],1):
  if item['name'] in state['archives_complete']:continue
  print('Assembling',number,'/',len(manifest['archives']),item['name'],flush=True)
  archive=merge_archive(item,ROOT,temp);extract_archive(archive,output);state['archives_complete'].append(item['name']);statepath.write_text(json.dumps(state,indent=2));archive.unlink()
 package=output/manifest['package_directory'];verify_package(package)
 if sha(package/'submission_rank16.xlsx')!=manifest['submission_sha256']:raise RuntimeError('Submission workbook mismatch')
 state['complete']=True;statepath.write_text(json.dumps(state,indent=2))
 print('COMPLETE:',package,flush=True);print('MANUALLY SUBMIT ONLY:',package/'submission_rank16.xlsx',flush=True)
 print('No competition result was uploaded. Recomputing is optional.',flush=True)

if __name__=='__main__':
 try:main()
 except Exception as e:print('FAILED:',str(e),file=sys.stderr);raise
