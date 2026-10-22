from __future__ import annotations
import argparse
import os
from pathlib import Path
import socket
import threading
import time
import webbrowser
from .util import data_home,read_json


def main():
    parser=argparse.ArgumentParser(description='Neuro Studio — local visual ML studio')
    parser.add_argument('--port',type=int,default=8765);parser.add_argument('--no-browser',action='store_true');parser.add_argument('--data-dir',type=Path)
    args=parser.parse_args();root=(args.data_dir or data_home()).resolve();root.mkdir(parents=True,exist_ok=True)
    lock=open(root/'workspace.lock','a+b')
    try:
        if os.name=='nt':
            import msvcrt
            lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        print('Neuro Studio đang chạy với thư mục này. Mở cửa sổ đang có hoặc chọn --data-dir khác.');return
    from .api import create_app
    import uvicorn
    engine=read_json(root/'engine.json',{}).get('python')
    if engine and not Path(engine).exists():engine=None
    app=create_app(root,os.environ.get('NEUROSTUDIO_ENGINE_PYTHON') or engine)
    sock=socket.socket();sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    try:sock.bind(('127.0.0.1',args.port))
    except OSError:sock.bind(('127.0.0.1',0))
    sock.listen(128);port=sock.getsockname()[1];url=f'http://127.0.0.1:{port}'
    print(f'Neuro Studio: {url}\nDữ liệu: {root}\nĐóng tab vẫn tiếp tục train. Ctrl+C dừng ứng dụng và worker có kiểm soát.',flush=True)
    if not args.no_browser:
        def open_when_ready():
            import urllib.request
            for _ in range(50):
                try:
                    urllib.request.urlopen(url,timeout=1);webbrowser.open(url);return
                except Exception:time.sleep(.2)
        threading.Thread(target=open_when_ready,daemon=True).start()
    try:uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=port,log_level='info')).run(sockets=[sock])
    except KeyboardInterrupt:print('Neuro Studio đã dừng. Dự án và checkpoint đã lưu được giữ lại.',flush=True)
    finally:lock.close();sock.close()

if __name__=='__main__':main()
