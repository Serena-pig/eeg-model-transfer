"""Restore and extract the verified model ZIP downloaded from your private repository."""
from pathlib import Path,PurePosixPath
from zipfile import ZipFile
import hashlib,json,os
ROOT=Path(__file__).resolve().parent

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    m=json.loads((ROOT/'parts_manifest.json').read_text(encoding='utf-8'));target=ROOT/'eeg_best_model.zip'
    if target.is_file() and sha256(target)==m['full_sha256']:
        print('已找到校验正确的完整模型ZIP。')
    else:
        missing=[p['path'] for p in m['parts'] if not (ROOT/p['path']).is_file()]
        if missing:raise RuntimeError('文件不完整，请重新下载并完整解压仓库ZIP。缺少分块：'+str(missing[:8]))
        temp=ROOT/'eeg_best_model.zip.partial';whole=hashlib.sha256();total=0
        with temp.open('wb') as output:
            for i,part in enumerate(m['parts'],1):
                path=ROOT/part['path'];data=path.read_bytes()
                if len(data)!=part['bytes'] or hashlib.sha256(data).hexdigest()!=part['sha256']:raise RuntimeError('分块校验失败：'+part['path'])
                output.write(data);whole.update(data);total+=len(data)
                if i%40==0 or i==len(m['parts']):print('合并并核验：',i,'/',len(m['parts']))
            output.flush();os.fsync(output.fileno())
        if total!=m['full_bytes'] or whole.hexdigest()!=m['full_sha256']:raise RuntimeError('模型总校验失败，请勿使用。')
        temp.replace(target);print('完整模型ZIP SHA256校验通过。')
    with ZipFile(target) as z:
        for name in z.namelist():
            p=PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:raise RuntimeError('ZIP路径不安全。')
        z.extractall(ROOT)
    folder=ROOT/'EEG_BEST_AUC_20261025'
    print('\n模型已还原并解压至：',folder)
    print('下一步：打开该目录，双击01_install_dependencies.cmd。')
    print('之后将官方测试MAT或ZIP放进test_data，再运行02_generate_submission.cmd。')
    print('本脚本只合并、校验和解压文件；不会提交比赛结果。')
if __name__=='__main__':main()
