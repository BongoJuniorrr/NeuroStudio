"""Create a same-platform offline application with managed Python and hashed wheels."""
from pathlib import Path
import hashlib,json,os,platform,re,shutil,subprocess,sys,zipfile
from .util import atomic_json,file_hash


def local_requirements(text):
    from urllib.parse import unquote,urlparse
    def replace(match):
        name,url=match.group(1),match.group(2);version=unquote(Path(urlparse(url).path).name).split('-')[1]
        return name+'=='+version
    return re.sub(r'([A-Za-z0-9_-]+) @ (https://[^\s;]+)',replace,text)


def build_offline(destination,profile='tabular',control=lambda:None,emit=lambda *args:None):
    if profile not in ['tabular','analysis','cpu']:raise ValueError('OFFLINE_PROFILE: Choose tabular, analysis or cpu.')
    if platform.machine().lower() not in ['x86_64','amd64']:raise ValueError('OFFLINE_PLATFORM: x86-64 required.')
    dest=Path(destination).resolve();dest.mkdir(parents=True,exist_ok=True);package=Path(__file__).parent
    uv=Path(os.environ.get('NEUROSTUDIO_UV') or shutil.which('uv') or '')
    if not uv.is_file():raise ValueError('OFFLINE_UV: uv executable is unavailable.')
    wheels=dest/'wheels';wheels.mkdir(exist_ok=True);tools=dest/'tools';tools.mkdir(exist_ok=True);shutil.copy2(uv,tools/('uv.exe' if os.name=='nt' else 'uv'))
    locks=[package/'core-runtime.lock',package/'engine-base.lock']
    if profile=='analysis':locks.append(package/'engine-analysis.lock')
    if profile=='cpu':locks.append(package/'torch-cpu.lock')
    requirements='\n'.join(local_requirements(p.read_text(encoding='utf-8')) for p in locks);(dest/'requirements.lock').write_text(requirements, encoding='utf-8')
    def command(args):
        control();emit('log',dict(message='Preparing offline dependency files. This may take several minutes.'))
        subprocess.run(args,check=True);control()
    download=[str(uv),'run','--no-project','--python','3.12','--with','pip==25.1.1','python','-m','pip','download','--only-binary=:all:','--require-hashes','--dest',str(wheels)]
    for lock in locks:download+=['-r',str(lock)]
    command(download)
    command([str(uv),'python','install','3.12','--install-dir',str(tools/'python')])
    app=dest/'app'/'neurostudio';shutil.copytree(package,app,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (dest/'offline_bootstrap.py').write_text('''from pathlib import Path
import hashlib,json,os,subprocess,sys
root=Path(__file__).resolve().parent
manifest=json.loads((root/'OFFLINE_MANIFEST.json').read_text(encoding='utf-8'))
if manifest['platform']!=sys.platform:raise SystemExit('Use an offline bundle built for this operating system.')
for name,expected in manifest['files'].items():
    path=root/name
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise SystemExit('OFFLINE_CHECKSUM: '+name)
uv=root/'tools'/('uv.exe' if os.name=='nt' else 'uv')
python=root/'.venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
env=dict(os.environ);env.update(UV_OFFLINE='1',UV_PYTHON_DOWNLOADS='never',UV_NO_INDEX='1',UV_FIND_LINKS=str(root/'wheels'))
if not python.exists():subprocess.run([str(uv),'venv',str(root/'.venv'),'--python',sys.executable],env=env,check=True)
marker=root/'.venv'/'offline-ready'
if not marker.exists():
    subprocess.run([str(uv),'pip','install','--python',str(python),'--no-index','--find-links',str(root/'wheels'),'-r',str(root/'requirements.lock'),'--require-hashes'],env=env,check=True)
    marker.write_text('ready')
env['PYTHONPATH']=str(root/'app');env['NEUROSTUDIO_ENGINE_PYTHON']=str(python);env['NEUROSTUDIO_UV']=str(uv)
raise SystemExit(subprocess.call([str(python),'-m','neurostudio.launcher',*sys.argv[1:]],env=env))
''', encoding='utf-8')
    (dest/'Start Offline.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
chmod +x "$root/tools/uv"
python="$(find "$root/tools/python" -type f -name python3.12 -print -quit)"
if [[ -z "$python" ]]; then echo "Bundled Python is missing"; exit 1; fi
chmod +x "$python"
exec "$python" "$root/offline_bootstrap.py" "$@"
''', encoding='utf-8')
    (dest/'Start Offline.ps1').write_text('''$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$python = Get-ChildItem (Join-Path $root 'tools/python') -Filter python.exe -Recurse | Select-Object -First 1
if (!$python) { throw 'Bundled Python is missing' }
& $python.FullName (Join-Path $root 'offline_bootstrap.py') @args
exit $LASTEXITCODE
''', encoding='utf-8')
    (dest/'Start Offline.cmd').write_text('@echo off\r\npowershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start Offline.ps1" %*\r\n', encoding='utf-8')
    (dest/'README.txt').write_text('Neuro Studio offline bundle\nBuilt for '+sys.platform+' x86-64, profile '+profile+'.\nLinux: bash "Start Offline.sh". Windows: open "Start Offline.cmd".\nPython, application, uv and selected dependency wheels are included. No Node or online package index is needed.\nModel weights and datasets are excluded: download/import them before going offline. GPU profiles and pretrained-model caches are not included in this bundle.\nThis bundle uses its own selected training engine. Pass --data-dir to choose a workspace; existing project data is retained.\n', encoding='utf-8')
    files={p.relative_to(dest).as_posix():file_hash(p) for p in dest.rglob('*') if p.is_file() and p.name!='OFFLINE_MANIFEST.json' and p.suffix!='.pyc' and '__pycache__' not in p.parts}
    atomic_json(dest/'OFFLINE_MANIFEST.json',dict(platform=sys.platform,architecture=platform.machine(),profile=profile,files=files))
    target=dest.with_suffix('.zip');pending=target.with_suffix('.zip.pending')
    with zipfile.ZipFile(pending,'w',zipfile.ZIP_DEFLATED) as z:
        for p in dest.rglob('*'):
            control()
            if p.is_file() and p.suffix!='.pyc' and '__pycache__' not in p.parts:z.write(p,p.relative_to(dest).as_posix())
    pending.replace(target);return dict(path=str(target),sha256=file_hash(target),bytes=target.stat().st_size,platform=sys.platform,profile=profile)
