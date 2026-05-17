"""
启动Web服务（无控制台窗口）- 双击运行
启动后在浏览器访问 http://localhost:8080

命令行用法:
    python start_web.pyw              # 启动服务
    python start_web.pyw --stop       # 停止服务
    python start_web.pyw --restart    # 重启服务
    python start_web.pyw --status     # 查看状态
"""
import os
import sys
import socket
import webbrowser
import threading
import signal
import subprocess
import time

# PID 文件路径
PID_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".web_server.pid")
PORT = 8080

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

def open_browser():
    time.sleep(1.5)
    webbrowser.open(f"http://localhost:{PORT}")

def write_pid():
    """写入当前进程 PID"""
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

def read_pid():
    """读取保存的 PID"""
    if os.path.exists(PID_FILE):
        with open(PID_FILE, "r") as f:
            return int(f.read().strip())
    return None

def is_process_running(pid):
    """检查进程是否在运行"""
    try:
        # Windows: 用 tasklist 检查
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True
        )
        return str(pid) in result.stdout
    except:
        return False

def is_port_in_use(port):
    """检查端口是否被占用"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            return s.connect_ex(("127.0.0.1", port)) == 0
    except:
        return False

def get_port_process_info(port):
    """获取占用端口的进程信息"""
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True
        )
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                parts = line.split()
                pid = parts[-1]
                # 获取进程名
                proc_result = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                    capture_output=True, text=True
                )
                return pid, proc_result.stdout.strip()
    except:
        pass
    return None, None

def stop_server():
    """停止服务"""
    pid = read_pid()
    if pid and is_process_running(pid):
        print(f"[INFO] 正在停止服务 (PID: {pid})...")
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(1)
            if is_process_running(pid):
                os.kill(pid, signal.SIGKILL)
            print("[OK] 服务已停止")
            return True
        except Exception as e:
            print(f"[ERROR] 停止失败: {e}")
    
    # 尝试通过端口查找进程
    if is_port_in_use(PORT):
        port_pid, _ = get_port_process_info(PORT)
        if port_pid:
            print(f"[INFO] 端口 {PORT} 被进程 {port_pid} 占用，尝试停止...")
            try:
                os.kill(int(port_pid), signal.SIGTERM)
                time.sleep(1)
                print("[OK] 已停止占用端口的进程")
                return True
            except:
                pass
    
    print("[INFO] 没有发现运行中的服务")
    return False

def show_status():
    """显示服务状态"""
    pid = read_pid()
    
    print("=" * 50)
    print("Web 服务状态")
    print("=" * 50)
    
    if pid:
        if is_process_running(pid):
            print(f"[OK]   服务正在运行 (PID: {pid})")
        else:
            print(f"[WARN] PID 文件存在但进程未运行 (旧 PID: {pid})")
    else:
        print("[--]   没有 PID 记录")
    
    if is_port_in_use(PORT):
        port_pid, proc_info = get_port_process_info(PORT)
        print(f"[OK]   端口 {PORT} 正在监听 (进程: {port_pid})")
        print(f"       访问地址: http://localhost:{PORT}")
        print(f"       局域网:   http://{get_local_ip()}:{PORT}")
    else:
        print(f"[--]   端口 {PORT} 未监听")
    
    print("=" * 50)

def start_server():
    """启动服务"""
    # 检查是否已有服务在运行
    if is_port_in_use(PORT):
        print(f"[WARN] 端口 {PORT} 已被占用!")
        port_pid, proc_info = get_port_process_info(PORT)
        if port_pid:
            print(f"       进程 PID: {port_pid}")
            print(f"       进程信息: {proc_info}")
        print(f"\n[INFO] 请先停止现有服务，或使用 --restart 重启")
        show_status()
        return
    
    # 写入 PID
    write_pid()
    
    print(f"[INFO] 启动 Web 服务...")
    print(f"[INFO] 本地访问: http://localhost:{PORT}")
    print(f"[INFO] 局域网:   http://{get_local_ip()}:{PORT}")
    
    # 延迟打开浏览器
    threading.Thread(target=open_browser, daemon=True).start()
    
    # 切换到脚本所在目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 启动服务
    from web_server import app, load_tasks_from_disk
    load_tasks_from_disk()
    
    try:
        app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
    finally:
        # 清理 PID 文件
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)

def restart_server():
    """重启服务"""
    print("[INFO] 正在重启服务...")
    stop_server()
    time.sleep(2)
    start_server()

if __name__ == "__main__":
    # 解析命令行参数
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd == "--stop" or cmd == "-s":
            stop_server()
        elif cmd == "--restart" or cmd == "-r":
            restart_server()
        elif cmd == "--status" or cmd == "-t":
            show_status()
        else:
            print(f"[ERROR] 未知命令: {cmd}")
            print("用法: python start_web.pyw [--stop|--restart|--status]")
    else:
        start_server()
