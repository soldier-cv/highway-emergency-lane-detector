"""共享 HTML 报告与证据索引工具。"""

import base64
import html
import json
import os
import re
import time

from utils import format_tc


INVALID_FILENAME_CHARS = r'[<>:"/\\|?*]'


def _plate_text(result):
    return result.get("plate") or result.get("plate_number") or "未识别"


def is_valid_plate(plate):
    if not plate or plate == "未识别":
        return False
    if len(plate) < 7:
        return False
    if any(ch in plate for ch in "?*#@ "):
        return False
    return True


def sanitize_plate_key(plate):
    plate = (plate or "未识别").strip()
    safe = re.sub(INVALID_FILENAME_CHARS, "_", plate)
    safe = safe.replace(" ", "_")
    return safe or "未识别"


def _confidence(result):
    value = result.get("confidence", result.get("plate_confidence", 0))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _time_text(result):
    if result.get("time"):
        return str(result["time"])
    if result.get("video_time"):
        return str(result["video_time"])
    if result.get("time_seconds") is not None:
        return format_tc(result["time_seconds"])
    return "-"


def build_evidence_key(result, fallback_index=None):
    plate = _plate_text(result)
    if is_valid_plate(plate):
        return sanitize_plate_key(plate)
    if fallback_index is not None:
        return f"未识别_{fallback_index:03d}"
    return "未识别"


def dedupe_violations(results, time_window=5.0):
    best_by_plate = {}

    for r in results:
        plate = _plate_text(r)
        if not is_valid_plate(plate):
            continue

        current_conf = _confidence(r)
        prev = best_by_plate.get(plate)
        if prev is None or current_conf > _confidence(prev):
            best_by_plate[plate] = r

    deduped = []
    for index, result in enumerate(best_by_plate.values(), start=1):
        cloned = dict(result)
        cloned["plate_key"] = build_evidence_key(cloned, fallback_index=index)
        deduped.append(cloned)

    deduped.sort(key=lambda item: item.get("time_seconds", item.get("timestamp_seconds", 0)))
    return deduped


def build_manifest(results, output_dir, video_path):
    manifest = {
        "video": os.path.basename(video_path),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "evidence": [],
    }

    for result in results:
        manifest["evidence"].append(
            {
                "id": result.get("id"),
                "plate": _plate_text(result),
                "plate_key": result.get("plate_key") or build_evidence_key(result),
                "time": _time_text(result),
                "time_seconds": result.get("time_seconds", result.get("timestamp_seconds")),
                "video_time": result.get("video_time"),
                "dvr_time": result.get("dvr_time"),
                "confidence": _confidence(result),
                "plate_color_name": result.get("plate_color_name"),
                "snapshot": os.path.relpath(result["snapshot"], output_dir).replace(os.sep, "/") if result.get("snapshot") else None,
                "clip": os.path.relpath(result["clip_path"], output_dir).replace(os.sep, "/") if result.get("clip_path") else None,
            }
        )

    return manifest


def write_manifest(results, output_dir, video_path):
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(build_manifest(results, output_dir, video_path), f, ensure_ascii=False, indent=2)
    return manifest_path


def generate_html_report(results, video_path, output_dir, video_name, lane_x, lane_width, lane_top, clips=None, base_url=None, embed_snapshots=False):
    html_path = os.path.join(output_dir, f"{video_name}_violation_report.html")

    show_color = any(r.get("plate_color_name") for r in results)
    show_dvr = any(r.get("dvr_time") for r in results)
    show_video_time = any(r.get("video_time") for r in results)

    header_cells = ["<th>编号</th>", "<th>车牌号</th>", "<th>时间</th>", "<th>置信度</th>"]
    if show_color:
        header_cells.append("<th>车牌颜色</th>")
    if show_dvr:
        header_cells.append("<th>DVR时间</th>")
    elif show_video_time:
        header_cells.append("<th>视频时间</th>")
    header_cells.extend(["<th>截图</th>", "<th>视频</th>"])

    rows = ""
    color_html = {
        "蓝牌": "#0052CC",
        "黄牌": "#F5A623",
        "绿牌": "#00B140",
        "白牌": "#CCCCCC",
        "黑牌": "#333333",
        "未知": "#999999",
    }

    for index, r in enumerate(results, start=1):
        plate = html.escape(_plate_text(r))
        confidence = _confidence(r)
        time_text = html.escape(_time_text(r))

        snap_link = "-"
        snap = r.get("snapshot", "")
        if snap and os.path.exists(snap):
            rel = os.path.relpath(snap, output_dir).replace(os.sep, "/")
            snap_href = f"{base_url}/{rel}" if base_url else rel
            if embed_snapshots:
                with open(snap, "rb") as f:
                    img = base64.b64encode(f.read()).decode()
                snap_link = f'<a href="{snap_href}" target="_blank"><img src="data:image/jpeg;base64,{img}" style="max-width:420px;border:2px solid #c00;border-radius:8px;"></a>'
            else:
                snap_link = f'<a href="{snap_href}" target="_blank">查看截图</a>'

        clip_link = "-"
        clip_path = r.get("clip_path", "")
        if clip_path and os.path.exists(clip_path):
            rel = os.path.relpath(clip_path, output_dir).replace(os.sep, "/")
            clip_href = f"{base_url}/{rel}" if base_url else rel
            clip_link = f'<a href="{clip_href}" target="_blank">查看视频</a>'

        cells = [
            f"<td>{index}</td>",
            f"<td style='color:#27ae60;font-weight:bold;font-size:1.1em'>{plate}</td>",
            f"<td>{time_text}</td>",
            f"<td>{confidence:.0%}</td>",
        ]

        if show_color:
            color_name = r.get("plate_color_name", "未知")
            color_bg = color_html.get(color_name, "#999999")
            cells.append(f"<td><span style='display:inline-block;padding:2px 10px;border-radius:4px;background:{color_bg};color:white;font-weight:bold;'>{html.escape(str(color_name))}</span></td>")
        if show_dvr:
            cells.append(f"<td>{html.escape(str(r.get('dvr_time', '-')))}</td>")
        elif show_video_time:
            cells.append(f"<td>{html.escape(str(r.get('video_time', '-')))}</td>")

        cells.append(f"<td>{snap_link}</td>")
        cells.append(f"<td>{clip_link}</td>")
        rows += "<tr>" + "".join(cells) + "</tr>"

    html_text = f"""<!DOCTYPE html>
<html lang=\"zh-CN\">
<head>
<meta charset=\"UTF-8\">
<title>应急车道违章检测报告 - {html.escape(video_name)}</title>
<style>
body {{ font-family: 'Microsoft YaHei', sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
.card {{ background: white; border-radius: 12px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
h1 {{ color: #c00; border-bottom: 3px solid #c00; padding-bottom: 10px; }}
h2 {{ color: #2c3e50; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
th {{ background: #2c3e50; color: white; padding: 10px; text-align: center; }}
td {{ padding: 8px; text-align: center; border-bottom: 1px solid #eee; }}
tr:hover {{ background: #f0f0f0; }}
.stats {{ display: flex; gap: 20px; margin: 20px 0; flex-wrap: wrap; }}
.stat {{ background: #ecf0f1; border-radius: 8px; padding: 15px 25px; text-align: center; }}
.stat .num {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
.stat .label {{ color: #7f8c8d; margin-top: 5px; }}
a {{ color: #3498db; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<div class=\"card\">
<h1>🚨 应急车道违章检测报告</h1>
<p><strong>视频文件：</strong>{html.escape(os.path.basename(video_path))}</p>
<p><strong>检测时间：</strong>{time.strftime('%Y-%m-%d %H:%M:%S')}</p>
<p><strong>应急车道区域：</strong>X={lane_x:.2f}, 宽度={lane_width:.2f}, 顶部={lane_top:.2f}</p>
<div class=\"stats\">
    <div class=\"stat\"><div class=\"num\">{len(results)}</div><div class=\"label\">保留证据数</div></div>
    <div class=\"stat\"><div class=\"num\" style=\"color:#27ae60\">{len(results)}</div><div class=\"label\">有效车牌</div></div>
</div>
</div>

<div class=\"card\">
<h2>📋 证据详情</h2>
<table>
<tr>{''.join(header_cells)}</tr>
{rows}
</table>
</div>
</body>
</html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_text)

    return html_path
