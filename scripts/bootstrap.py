"""Runs under uv's managed Python. No system Python or Node.js is required."""
import os
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parents[1]
if not (root/'src/neurostudio/static/index.html').exists():
    raise SystemExit('Web assets chưa có. Dùng release ZIP hoặc chạy npm ci && npm run build trong apps/web cho developer.')
uv=os.environ.get('NEUROSTUDIO_UV','uv');venv=root/'.venv'
python=venv/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
if not python.exists():subprocess.run([uv,'venv',str(venv),'--python',sys.executable],check=True)
marker=venv/'neurostudio-installed'
lock=root/'packaging/core.lock'
import hashlib
key=hashlib.sha256(lock.read_bytes()+ (root/'pyproject.toml').read_bytes()).hexdigest()
if not marker.exists() or marker.read_text(encoding='utf-8')!=key:
    subprocess.run([uv,'pip','install','--python',str(python),'-r',str(lock),'--require-hashes'],check=True)
    subprocess.run([uv,'pip','install','--python',str(python),'--no-deps','-e',str(root)],check=True)
    marker.write_text(key, encoding='utf-8')
raise SystemExit(subprocess.call([str(python),'-m','neurostudio.launcher',*sys.argv[1:]]))
