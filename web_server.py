"""
应急车道违章检测系统 - Web服务
================================
目录结构: data/{client_ip}/{video_name}/
"""

import os
import sys
import json
import uuid
import time
import shutil
import threading
import queue
import socket
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_file, Response, render_template
from werkzeug.utils import secure_filename

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
WEB_DIR = os.path.join(BASE_DIR, "web")
TEMPLATES_DIR = os.path.join(WEB_DIR, "templates")

# 添加项目路径
sys.path.insert(0, os.path.join(BASE_DIR, "emergency_lane"))

app = Flask(__name__, template_folder=TEMPLATES_DIR)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB

# 允许的视频格式
ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv'}

# 任务管理
tasks = {}  # task_id -> task_info
task_queues = {}  # task_id -> queue for SSE


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def get_client_ip():
    """获取客户端真实IP"""
    if request.headers.get('X-Forwarded-For'):
        return request.headers['X-Forwarded-For'].split(',')[0].strip()
    return request.remote_addr or '127.0.0.1'


def sanitize_dirname(name):
    """清理目录名，移除非法字符"""
    # 保留中文、字母、数字、下划线、短横线、点
    import re
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    return name.strip()[:100]  # 限制长度


def get_task_dir(client_ip, video_name, task_id):
    """获取任务目录: data/{ip}/{video_name}_{task_id}/"""
    ip_dir = sanitize_dirname(client_ip)
    name_dir = sanitize_dirname(Path(video_name).stem)
    return os.path.join(DATA_DIR, ip_dir, f"{name_dir}_{task_id}")


def load_tasks_from_disk():
    """从磁盘加载历史任务"""
    if not os.path.exists(DATA_DIR):
        return

    for ip_dir in os.listdir(DATA_DIR):
        ip_path = os.path.join(DATA_DIR, ip_dir)
        if not os.path.isdir(ip_path):
            continue

        for task_dir in os.listdir(ip_path):
            task_path = os.path.join(ip_path, task_dir)
            if not os.path.isdir(task_path):
                continue

            # 从目录名提取 task_id
            parts = task_dir.rsplit('_', 1)
            if len(parts) != 2:
                continue

            task_id = parts[1]
            if len(task_id) != 8:
                continue

            # 查找视频文件
            video_files = list(Path(task_path).glob('*.mp4')) + list(Path(task_path).glob('*.avi')) + \
                         list(Path(task_path).glob('*.mkv')) + list(Path(task_path).glob('*.mov'))

            # 查找结果文件
            html_files = list(Path(task_path).glob('*_violation_report.html'))
            snapshot_dirs = [d for d in task_path.iterdir() if d.is_dir() and d.name.endswith('_snapshots')]

            original_name = video_files[0].name if video_files else task_dir
            status = 'completed' if html_files else 'uploaded'
            created_at = datetime.fromtimestamp(os.path.getctime(task_path)).isoformat()

            result = None
            if html_files:
                try:
                    # 扫描结果文件（排除原始视频和隐藏文件）
                    video_exts = {'.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv'}
                    result_files = []
                    for root, dirs, files in os.walk(task_path):
                        for file in files:
                            if file.startswith('.'):
                                continue
                            if Path(file).suffix.lower() in video_exts:
                                # 跳过位于任务根目录的原始视频
                                if os.path.dirname(os.path.join(root, file)) == str(task_path):
                                    continue
                            file_path = os.path.join(root, file)
                            rel_path = os.path.relpath(file_path, task_path).replace('\\', '/')
                            result_files.append({
                                'name': file,
                                'path': rel_path,
                                'size': os.path.getsize(file_path)
                            })

                    clips_dir = task_path / "举报视频片段"
                    clips_count = len(list(clips_dir.glob('*.mp4'))) if clips_dir.exists() else 0

                    result = {
                        'total_violations': len(snapshot_dirs),
                        'recognized_plates': 0,
                        'total_time': 0,
                        'clips_count': clips_count,
                        'files': result_files
                    }
                except:
                    pass

            tasks[task_id] = {
                'id': task_id,
                'original_name': original_name,
                'video_path': str(video_files[0]) if video_files else '',
                'task_dir': task_path,
                'status': status,
                'progress': 100 if status == 'completed' else 0,
                'message': '已完成' if status == 'completed' else '待检测',
                'client_ip': ip_dir,
                'created_at': created_at,
                'completed_at': created_at if status == 'completed' else None,
                'result': result
            }


def run_detection_task(task_id, video_path, task_dir, params):
    """后台运行检测任务"""
    try:
        task = tasks[task_id]
        task['status'] = 'processing'
        task['progress'] = 0
        task['message'] = '正在初始化...'
        task['task_dir'] = task_dir

        os.makedirs(task_dir, exist_ok=True)

        def progress_callback(msg, pct):
            if pct >= 0:
                task['progress'] = pct
            task['message'] = msg
            if task_id in task_queues:
                task_queues[task_id].put({
                    'type': 'progress',
                    'progress': pct,
                    'message': msg
                })

        from traffic_violation_gui import run_detection

        # 复制视频到任务目录
        video_name = Path(video_path).name
        dest_video = os.path.join(task_dir, video_name)
        if video_path != dest_video:
            shutil.copy2(video_path, dest_video)

        # 运行检测
        result = run_detection(
            video_path=dest_video,
            lane_x=params.get('lane_x', 0.84),
            lane_width=params.get('lane_width', 0.16),
            lane_top=params.get('lane_top', 0.15),
            detection_scale=params.get('detection_scale', 0.5),
            conf_threshold=params.get('conf_threshold', 0.5),
            use_gpu=params.get('use_gpu', True),
            progress_callback=progress_callback
        )

        # 移动结果文件到任务目录
        video_stem = Path(video_name).stem

        # 移动报告文件
        html_path = result.get('html_path', '')
        if html_path and os.path.exists(html_path):
            dst = os.path.join(task_dir, os.path.basename(html_path))
            if html_path != dst:
                shutil.move(html_path, dst)

        # 移动截图目录
        snapshot_src = result.get('snapshot_dir', '')
        snapshot_dst = os.path.join(task_dir, f"{video_stem}_snapshots")
        if snapshot_src and os.path.exists(snapshot_src):
            if snapshot_src != snapshot_dst:
                if os.path.exists(snapshot_dst):
                    shutil.rmtree(snapshot_dst)
                shutil.move(snapshot_src, snapshot_dst)
        # 更新截图路径
        for v in result.get('violations', []):
            old_snap = v.get('snapshot', '')
            if old_snap:
                v['snapshot'] = os.path.join(snapshot_dst, os.path.basename(old_snap))

        # 移动视频片段目录
        clips_src = os.path.join(os.path.dirname(video_path), "举报视频片段")
        if not os.path.exists(clips_src):
            clips_src = os.path.join(os.path.dirname(dest_video), "举报视频片段")
        if os.path.exists(clips_src):
            clips_dst = os.path.join(task_dir, "举报视频片段")
            if clips_src != clips_dst:
                if os.path.exists(clips_dst):
                    shutil.rmtree(clips_dst)
                shutil.move(clips_src, clips_dst)

        # 清理源目录残留
        for cleanup_dir in [
            os.path.join(os.path.dirname(video_path), f"{video_stem}_snapshots"),
            os.path.join(os.path.dirname(video_path), f"{video_stem}_检测结果"),
        ]:
            if os.path.exists(cleanup_dir):
                shutil.rmtree(cleanup_dir, ignore_errors=True)

        # 重新生成HTML报告（使用Web服务URL，确保截图和视频链接正确）
        from traffic_violation_gui import generate_html_report
        base_url = f"/task-files/{task_id}"
        html_report_path = os.path.join(task_dir, f"{video_stem}_violation_report.html")
        generate_html_report(
            result['violations'], dest_video, task_dir, video_stem,
            params.get('lane_x', 0.84), params.get('lane_width', 0.16),
            params.get('lane_top', 0.15), result.get('clips', []),
            base_url=base_url
        )

        # 生成结果文件列表（排除原始视频）
        video_filenames = {video_name, os.path.basename(video_path)}
        result_files = []
        for root, dirs, files in os.walk(task_dir):
            for file in files:
                if file.startswith('.'):
                    continue
                if file in video_filenames:
                    continue
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, task_dir).replace('\\', '/')
                result_files.append({
                    'name': file,
                    'path': rel_path,
                    'size': os.path.getsize(file_path)
                })

        task['status'] = 'completed'
        task['progress'] = 100
        task['message'] = '检测完成！'
        task['completed_at'] = datetime.now().isoformat()
        task['result'] = {
            'total_violations': result['total_violations'],
            'recognized_plates': result['recognized_plates'],
            'total_time': round(result['total_time'], 1),
            'clips_count': len(result.get('clips', [])),
            'files': result_files
        }

        if task_id in task_queues:
            task_queues[task_id].put({
                'type': 'completed',
                'result': task['result']
            })

    except Exception as e:
        import traceback
        traceback.print_exc()
        task['status'] = 'error'
        task['message'] = f'检测失败: {str(e)}'
        if task_id in task_queues:
            task_queues[task_id].put({
                'type': 'error',
                'message': str(e)
            })


# ============ 路由 ============

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': '没有选择文件'}), 400

    file = request.files['video']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': '不支持的视频格式'}), 400

    task_id = str(uuid.uuid4())[:8]
    client_ip = get_client_ip()
    original_name = file.filename

    # 创建任务目录
    task_dir = get_task_dir(client_ip, original_name, task_id)
    os.makedirs(task_dir, exist_ok=True)

    # 保存视频
    safe_name = secure_filename(original_name)
    if not safe_name:
        safe_name = f"video_{task_id}.mp4"
    video_path = os.path.join(task_dir, safe_name)
    file.save(video_path)

    tasks[task_id] = {
        'id': task_id,
        'original_name': original_name,
        'video_path': video_path,
        'task_dir': task_dir,
        'status': 'uploaded',
        'progress': 0,
        'message': '视频已上传，等待开始检测',
        'client_ip': client_ip,
        'created_at': datetime.now().isoformat(),
        'completed_at': None,
        'result': None
    }

    return jsonify({'task_id': task_id, 'filename': original_name, 'message': '上传成功'})


@app.route('/api/detect/<task_id>', methods=['POST'])
def start_detection(task_id):
    task = tasks.get(task_id)
    if not task:
        return jsonify({'error': '任务不存在'}), 404
    if task['status'] == 'processing':
        return jsonify({'error': '任务正在处理中'}), 400

    params = request.json or {}
    task_queues[task_id] = queue.Queue()

    thread = threading.Thread(
        target=run_detection_task,
        args=(task_id, task['video_path'], task['task_dir'], params),
        daemon=True
    )
    thread.start()

    return jsonify({'message': '检测已开始', 'task_id': task_id})


@app.route('/api/status/<task_id>')
def get_status(task_id):
    task = tasks.get(task_id)
    if not task:
        return jsonify({'error': '任务不存在'}), 404

    return jsonify({
        'id': task['id'],
        'status': task['status'],
        'progress': task['progress'],
        'message': task['message'],
        'result': task.get('result')
    })


@app.route('/api/events/<task_id>')
def sse_events(task_id):
    def generate():
        if task_id not in task_queues:
            yield f"data: {json.dumps({'type': 'error', 'message': '任务不存在'})}\n\n"
            return

        q = task_queues[task_id]
        while True:
            try:
                event = q.get(timeout=30)
                yield f"data: {json.dumps(event)}\n\n"
                if event['type'] in ['completed', 'error']:
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"

    return Response(generate(), mimetype='text/event-stream')


@app.route('/task-files/<task_id>/<path:filename>')
def serve_task_file(task_id, filename):
    """提供任务文件的静态访问（用于HTML报告中的截图和视频片段链接）"""
    task = tasks.get(task_id)
    if not task:
        return jsonify({'error': '任务不存在'}), 404

    file_path = os.path.join(task['task_dir'], filename)
    if not os.path.exists(file_path) or os.path.isdir(file_path):
        return jsonify({'error': '文件不存在'}), 404

    return send_file(file_path)


@app.route('/api/download/<task_id>/<path:filename>')
def download_file(task_id, filename):
    task = tasks.get(task_id)
    if not task:
        return jsonify({'error': '任务不存在'}), 404

    file_path = os.path.join(task['task_dir'], filename)
    if not os.path.exists(file_path):
        return jsonify({'error': '文件不存在'}), 404

    return send_file(file_path, as_attachment=True)


@app.route('/api/download-all/<task_id>')
def download_all(task_id):
    import zipfile
    import io

    task = tasks.get(task_id)
    if not task:
        return jsonify({'error': '任务不存在'}), 404

    task_dir = task['task_dir']
    if not os.path.exists(task_dir):
        return jsonify({'error': '结果目录不存在'}), 404

    # 排除原始视频文件（体积大，不属于检测结果）
    original_video = os.path.basename(task.get('video_path', ''))
    video_exts = {'.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv'}

    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(task_dir):
            for file in files:
                if file.startswith('.'):
                    continue
                # 跳过任务根目录下的原始视频
                if file == original_video and os.path.abspath(root) == os.path.abspath(task_dir):
                    continue
                if Path(file).suffix.lower() in video_exts and os.path.abspath(root) == os.path.abspath(task_dir):
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, task_dir).replace('\\', '/')
                zf.write(file_path, arcname)

    memory_file.seek(0)
    # 使用ASCII安全的文件名，通过Content-Disposition的filename*参数传递UTF-8文件名
    safe_filename = f"detection_result_{task_id}.zip"
    utf8_filename = f"检测结果_{task_id}.zip"
    response = send_file(memory_file, mimetype='application/zip', as_attachment=True, download_name=safe_filename)
    # 添加RFC 5987编码的UTF-8文件名
    from urllib.parse import quote
    response.headers['Content-Disposition'] = (
        f"attachment; filename={safe_filename}; "
        f"filename*=UTF-8''{quote(utf8_filename)}"
    )
    return response


@app.route('/api/history')
def get_history():
    client_ip = get_client_ip()

    task_list = []
    for task_id, task in tasks.items():
        if task.get('client_ip') == client_ip:
            task_list.append({
                'id': task['id'],
                'original_name': task['original_name'],
                'status': task['status'],
                'progress': task['progress'],
                'created_at': task['created_at'],
                'completed_at': task.get('completed_at'),
                'result': task.get('result')
            })

    task_list.sort(key=lambda x: x['created_at'], reverse=True)
    return jsonify(task_list)


@app.route('/api/task/<task_id>', methods=['DELETE'])
def delete_task(task_id):
    task = tasks.get(task_id)
    if not task:
        return jsonify({'error': '任务不存在'}), 404

    # 删除任务目录
    task_dir = task.get('task_dir', '')
    if task_dir and os.path.exists(task_dir):
        shutil.rmtree(task_dir)

        # 清理空的IP目录
        ip_dir = os.path.dirname(task_dir)
        if os.path.exists(ip_dir) and not os.listdir(ip_dir):
            os.rmdir(ip_dir)

    del tasks[task_id]
    if task_id in task_queues:
        del task_queues[task_id]

    return jsonify({'message': '任务已删除'})


# ============ 启动 ============

if __name__ == '__main__':
    import argparse

    # 从磁盘加载历史任务
    load_tasks_from_disk()

    parser = argparse.ArgumentParser(description='应急车道违章检测Web服务')
    parser.add_argument('--host', default='0.0.0.0', help='监听地址')
    parser.add_argument('--port', type=int, default=8080, help='端口号')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    args = parser.parse_args()

    print("=" * 50)
    print("  应急车道违章检测系统 - Web服务")
    print("=" * 50)
    print(f"  本机访问: http://localhost:{args.port}")
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        print(f"  局域网: http://{local_ip}:{args.port}")
    except:
        pass
    print(f"  数据目录: {DATA_DIR}")
    print("=" * 50)

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
