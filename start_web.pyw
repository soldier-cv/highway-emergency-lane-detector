"""
启动Web服务（无控制台窗口）- 双击运行
启动后在浏览器访问 http://localhost:8080
"""
import os
import sys
import socket
import webbrowser
import threading

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
    import time
    time.sleep(1.5)
    webbrowser.open("http://localhost:8080")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 延迟打开浏览器
    threading.Thread(target=open_browser, daemon=True).start()

    # 启动服务
    from web_server import app, load_tasks_from_disk
    load_tasks_from_disk()
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)
