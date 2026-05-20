"""
启动 Web 服务控制台（带 GUI）

默认直接弹出服务控制窗口，方便查看运行状态、启动/停止服务和打开页面。
命令行用法:
    python start_web.pyw              # 打开 GUI 控制台
    python start_web.pyw --stop       # 停止服务
    python start_web.pyw --restart    # 重启服务
    python start_web.pyw --status     # 查看状态
"""

import os
import sys
import socket
import webbrowser
import subprocess
import threading
import time
import tkinter as tk
from tkinter import messagebox, scrolledtext

from werkzeug.serving import make_server


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PID_FILE = os.path.join(BASE_DIR, ".web_server.pid")
PORT = 8080
HOST = "0.0.0.0"


def _hidden_subprocess_kwargs():
    """返回隐藏控制台窗口的 subprocess 参数。"""
    kwargs = {
        "capture_output": True,
        "text": True,
    }

    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        kwargs["startupinfo"] = startupinfo

    return kwargs


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def write_pid(pid):
    with open(PID_FILE, "w", encoding="utf-8") as f:
        f.write(str(pid))


def read_pid():
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, "r", encoding="utf-8") as f:
                return int(f.read().strip())
        except Exception:
            return None
    return None


def remove_pid():
    if os.path.exists(PID_FILE):
        try:
            os.remove(PID_FILE)
        except Exception:
            pass


def is_process_running(pid):
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            **_hidden_subprocess_kwargs(),
        )
        return str(pid) in result.stdout
    except Exception:
        return False


def is_port_in_use(port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            return s.connect_ex(("127.0.0.1", port)) == 0
    except Exception:
        return False


def get_port_process_info(port):
    try:
        result = subprocess.run(["netstat", "-ano"], **_hidden_subprocess_kwargs())
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                parts = line.split()
                pid = int(parts[-1])
                proc_result = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                    **_hidden_subprocess_kwargs(),
                )
                return pid, proc_result.stdout.strip()
    except Exception:
        pass
    return None, None


def kill_process(pid):
    try:
        result = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            **_hidden_subprocess_kwargs(),
        )
        return result.returncode == 0, (result.stdout or result.stderr).strip()
    except Exception as e:
        return False, str(e)


def stop_server():
    pid = read_pid()
    if pid and is_process_running(pid):
        ok, msg = kill_process(pid)
        remove_pid()
        return ok, f"已停止服务进程 PID={pid}" if ok else f"停止失败: {msg}"

    if is_port_in_use(PORT):
        port_pid, _ = get_port_process_info(PORT)
        if port_pid:
            ok, msg = kill_process(port_pid)
            remove_pid()
            return ok, f"已停止占用端口的进程 PID={port_pid}" if ok else f"停止失败: {msg}"

    remove_pid()
    return True, "当前没有运行中的 Web 服务"


def show_status():
    pid = read_pid()
    lines = ["=" * 50, "Web 服务状态", "=" * 50]

    if pid:
        if is_process_running(pid):
            lines.append(f"[OK]   服务正在运行 (PID: {pid})")
        else:
            lines.append(f"[WARN] PID 文件存在但进程未运行 (旧 PID: {pid})")
    else:
        lines.append("[--]   没有 PID 记录")

    if is_port_in_use(PORT):
        port_pid, _ = get_port_process_info(PORT)
        lines.append(f"[OK]   端口 {PORT} 正在监听 (进程: {port_pid})")
        lines.append(f"       访问地址: http://localhost:{PORT}")
        lines.append(f"       局域网:   http://{get_local_ip()}:{PORT}")
    else:
        lines.append(f"[--]   端口 {PORT} 未监听")

    lines.append("=" * 50)
    text = "\n".join(lines)
    print(text)
    return text


class WebServerControlApp:
    """Web 服务桌面控制台。"""

    def __init__(self, root):
        self.root = root
        self.root.title("Web 服务控制台")
        self.root.geometry("860x620")
        self.root.minsize(760, 520)
        self.root.configure(bg="#f3f4f6")

        self.server = None
        self.server_thread = None

        self.status_var = tk.StringVar(value="正在检查服务状态...")
        self.local_url_var = tk.StringVar(value=f"http://localhost:{PORT}")
        self.lan_url_var = tk.StringVar(value=f"http://{get_local_ip()}:{PORT}")
        self.pid_var = tk.StringVar(value="-")

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._log("控制台已启动，可直接管理 Web 服务。")
        self._refresh_status()

    def _build_ui(self):
        header = tk.Frame(self.root, bg="#1f2937", height=64)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(
            header,
            text="应急车道违章检测 Web 服务控制台",
            font=("Microsoft YaHei", 18, "bold"),
            bg="#1f2937",
            fg="white",
        ).pack(pady=14)

        body = tk.Frame(self.root, bg="#f3f4f6")
        body.pack(fill=tk.BOTH, expand=True, padx=16, pady=12)

        info_card = tk.LabelFrame(
            body,
            text=" 服务状态 ",
            font=("Microsoft YaHei", 11, "bold"),
            bg="white",
            fg="#1f2937",
            padx=12,
            pady=10,
        )
        info_card.pack(fill=tk.X, pady=(0, 10))

        self.status_label = tk.Label(
            info_card,
            textvariable=self.status_var,
            font=("Microsoft YaHei", 11),
            bg="white",
            fg="#374151",
            anchor="w",
        )
        self.status_label.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        tk.Label(info_card, text="本机地址：", font=("Microsoft YaHei", 10), bg="white", fg="#6b7280").grid(row=1, column=0, sticky="w", pady=2)
        tk.Label(info_card, textvariable=self.local_url_var, font=("Consolas", 10), bg="white", fg="#111827").grid(row=1, column=1, sticky="w", pady=2)

        tk.Label(info_card, text="局域网地址：", font=("Microsoft YaHei", 10), bg="white", fg="#6b7280").grid(row=2, column=0, sticky="w", pady=2)
        tk.Label(info_card, textvariable=self.lan_url_var, font=("Consolas", 10), bg="white", fg="#111827").grid(row=2, column=1, sticky="w", pady=2)

        tk.Label(info_card, text="当前 PID：", font=("Microsoft YaHei", 10), bg="white", fg="#6b7280").grid(row=3, column=0, sticky="w", pady=2)
        tk.Label(info_card, textvariable=self.pid_var, font=("Consolas", 10), bg="white", fg="#111827").grid(row=3, column=1, sticky="w", pady=2)

        btn_row = tk.Frame(body, bg="#f3f4f6")
        btn_row.pack(fill=tk.X, pady=(0, 10))

        self.btn_start = tk.Button(btn_row, text="启动服务", font=("Microsoft YaHei", 11, "bold"), bg="#16a34a", fg="white", relief=tk.FLAT, cursor="hand2", command=self.start_server)
        self.btn_start.pack(side=tk.LEFT, ipadx=16, ipady=6)

        self.btn_stop = tk.Button(btn_row, text="停止服务", font=("Microsoft YaHei", 11, "bold"), bg="#dc2626", fg="white", relief=tk.FLAT, cursor="hand2", command=self.stop_server_gui)
        self.btn_stop.pack(side=tk.LEFT, padx=8, ipadx=16, ipady=6)

        self.btn_restart = tk.Button(btn_row, text="重启服务", font=("Microsoft YaHei", 11, "bold"), bg="#2563eb", fg="white", relief=tk.FLAT, cursor="hand2", command=self.restart_server)
        self.btn_restart.pack(side=tk.LEFT, padx=8, ipadx=16, ipady=6)

        self.btn_open = tk.Button(btn_row, text="打开页面", font=("Microsoft YaHei", 11), bg="#4b5563", fg="white", relief=tk.FLAT, cursor="hand2", command=self.open_browser)
        self.btn_open.pack(side=tk.RIGHT, ipadx=16, ipady=6)

        log_card = tk.LabelFrame(
            body,
            text=" 运行日志 ",
            font=("Microsoft YaHei", 11, "bold"),
            bg="white",
            fg="#1f2937",
            padx=10,
            pady=10,
        )
        log_card.pack(fill=tk.BOTH, expand=True)

        self.log_text = scrolledtext.ScrolledText(
            log_card,
            font=("Consolas", 9),
            bg="#111827",
            fg="#d1d5db",
            insertbackground="#d1d5db",
            relief=tk.FLAT,
            wrap=tk.WORD,
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.config(state=tk.DISABLED)

    def _log(self, message):
        timestamp = time.strftime("%H:%M:%S")

        def append():
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)

        self.root.after(0, append)

    def _set_buttons(self, running):
        self.btn_start.config(state=tk.DISABLED if running else tk.NORMAL)
        self.btn_stop.config(state=tk.NORMAL if running else tk.DISABLED)
        self.btn_restart.config(state=tk.NORMAL if running else tk.DISABLED)

    def _serve_forever(self):
        try:
            self.server.serve_forever()
        except Exception as e:
            self._log(f"服务线程异常退出: {e}")
        finally:
            self.server = None
            remove_pid()
            self.root.after(0, self._on_server_exit)

    def _on_server_exit(self):
        self._log("服务已停止。")
        self._refresh_status_once()

    def _shutdown_embedded_server(self):
        """优先停止当前控制台内嵌的 Web 服务，避免误杀控制台自身进程。"""
        if self.server is None:
            return False

        self._log("正在停止 Web 服务...")
        try:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
            remove_pid()
            self._refresh_status_once()
            return True
        except Exception as e:
            messagebox.showerror("停止失败", str(e))
            self._log(f"停止失败: {e}")
            return False

    def _get_running_pid(self):
        pid = read_pid()
        if pid and is_process_running(pid):
            return pid
        if is_port_in_use(PORT):
            port_pid, _ = get_port_process_info(PORT)
            return port_pid
        return None

    def _refresh_status_once(self):
        running_pid = self._get_running_pid()
        running = running_pid is not None and is_port_in_use(PORT)

        if running:
            self.status_var.set("服务运行中，可随时打开页面或停止服务。")
            self.pid_var.set(str(running_pid))
        else:
            self.status_var.set("服务未运行，点击“启动服务”即可开始。")
            self.pid_var.set("-")

        self.local_url_var.set(f"http://localhost:{PORT}")
        self.lan_url_var.set(f"http://{get_local_ip()}:{PORT}")
        self._set_buttons(running)

    def _refresh_status(self):
        self._refresh_status_once()
        self.root.after(2000, self._refresh_status)

    def start_server(self):
        if is_port_in_use(PORT):
            port_pid, proc_info = get_port_process_info(PORT)
            self._log(f"端口 {PORT} 已被占用，当前进程: {port_pid or '未知'}")
            if proc_info:
                self._log(proc_info)
            messagebox.showwarning("提示", f"端口 {PORT} 已被占用，请先停止现有服务。")
            return

        self._log("正在启动 Web 服务...")
        os.chdir(BASE_DIR)
        from web_server import app, load_tasks_from_disk

        load_tasks_from_disk()
        self.server = make_server(HOST, PORT, app, threaded=True)
        write_pid(os.getpid())
        self._log(f"服务已在当前控制台进程内启动，PID={os.getpid()}")
        self._set_buttons(True)

        self.server_thread = threading.Thread(target=self._serve_forever, daemon=True)
        self.server_thread.start()

        self.root.after(1200, self.open_browser)

    def stop_server_gui(self):
        if self._shutdown_embedded_server():
            return

        ok, message = stop_server()
        self._log(message)
        self._refresh_status_once()
        if not ok:
            messagebox.showerror("停止失败", message)

    def restart_server(self):
        self._log("正在重启服务...")
        if not self._shutdown_embedded_server():
            stop_server()
        time.sleep(1)
        self.start_server()

    def open_browser(self):
        try:
            webbrowser.open(f"http://localhost:{PORT}")
            self._log("已尝试打开浏览器。")
        except Exception as e:
            self._log(f"打开浏览器失败: {e}")

    def _on_close(self):
        running_pid = self._get_running_pid()
        if running_pid:
            should_stop = messagebox.askyesno("退出控制台", "检测到 Web 服务仍在运行，关闭窗口时是否一并停止服务？")
            if should_stop:
                if self.server is not None:
                    if not self._shutdown_embedded_server():
                        return
                else:
                    ok, message = stop_server()
                    self._log(message)
                    if not ok:
                        messagebox.showerror("停止失败", message)
                        return
        self.root.destroy()


def launch_gui():
    root = tk.Tk()
    WebServerControlApp(root)
    root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd in ("--stop", "-s"):
            ok, message = stop_server()
            print(message)
            sys.exit(0 if ok else 1)
        elif cmd in ("--restart", "-r"):
            ok, message = stop_server()
            print(message)
            time.sleep(1)
            launch_gui()
        elif cmd in ("--status", "-t"):
            show_status()
        else:
            print(f"[ERROR] 未知命令: {cmd}")
            print("用法: python start_web.pyw [--stop|--restart|--status]")
            sys.exit(1)
    else:
        launch_gui()
