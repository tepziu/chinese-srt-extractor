"""Video download from public URLs using yt-dlp or a browser fallback."""

from __future__ import annotations

import ipaddress
import os
import re
import socket
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

from config import MAX_DOWNLOAD_BYTES, UPLOAD_FOLDER, jobs
from services.whisper_engine import process_video


CHINESE_DOMAINS = {"douyin.com", "iesdouyin.com", "xiaohongshu.com", "kuaishou.com", "v.douyin.com"}


def validate_download_url(url: str) -> tuple[bool, str]:
    """Validate HTTP(S) URLs and reject private-network SSRF targets."""
    try:
        parsed = urlparse(str(url).strip())
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return False, "URL không hợp lệ"
        hostname = parsed.hostname.rstrip(".")
        try:
            addresses = {info[4][0] for info in socket.getaddrinfo(hostname, None)}
        except socket.gaierror:
            return False, "Không phân giải được hostname"
        for address in addresses:
            ip = ipaddress.ip_address(address)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
                return False, "URL trỏ tới mạng nội bộ hoặc địa chỉ bị hạn chế"
        return True, ""
    except (TypeError, ValueError):
        return False, "URL không hợp lệ"


def _check_download_size(size: int) -> None:
    if size > MAX_DOWNLOAD_BYTES:
        raise RuntimeError(
            f"Video vượt giới hạn tải xuống ({MAX_DOWNLOAD_BYTES / 1024 / 1024:.0f} MB)"
        )


def _job_dir(job_id: str) -> Path:
    path = UPLOAD_FOLDER / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_chinese_video(job_id: str, url: str) -> str:
    """Download video from Chinese platforms using undetected-chromedriver."""
    import requests as _requests

    jobs[job_id]["message"] = "Đang mở trình duyệt ẩn..."
    jobs[job_id]["progress"] = 5
    driver = None
    video_path = _job_dir(job_id) / "source.mp4"
    try:
        import undetected_chromedriver as uc

        def get_options():
            opts = uc.ChromeOptions()
            opts.add_argument("--headless=new")
            opts.add_argument("--no-sandbox")
            opts.add_argument("--disable-dev-shm-usage")
            opts.add_argument("--lang=zh-CN")
            opts.add_argument("--disable-gpu")
            return opts

        try:
            driver = uc.Chrome(options=get_options())
        except Exception as exc:
            match = re.search(r"Current browser version is (\d+)", str(exc))
            if not match:
                raise
            driver = uc.Chrome(options=get_options(), version_main=int(match.group(1)))

        jobs[job_id]["message"] = "Đang truy cập trang video..."
        jobs[job_id]["progress"] = 8
        driver.get(url)
        time.sleep(8)

        title = (driver.title or "Video").split(" - ")[0].strip()[:80]
        jobs[job_id]["original_name"] = title
        jobs[job_id]["message"] = f"Đang tải: {title[:40]}..."
        jobs[job_id]["progress"] = 12

        video_src = driver.execute_script(
            """
            const vids = document.querySelectorAll('video');
            for (const v of vids) {
                const src = v.src || v.currentSrc || '';
                if (src && !src.startsWith('blob:')) return src;
            }
            for (const v of vids) {
                const src = v.src || v.currentSrc || '';
                if (src) return src;
            }
            return '';
            """
        )
        if not video_src or video_src.startswith("blob:"):
            render_data = driver.execute_script(
                """
                const el = document.getElementById('RENDER_DATA');
                return el ? decodeURIComponent(el.textContent) : '';
                """
            )
            if render_data:
                urls = re.findall(
                    r'(https?://v[^"\s\\]+(?:zjcdn|douyinvod|bytecdn)[^"\s\\]*)',
                    render_data,
                )
                if urls:
                    video_src = urls[0].replace("\\u002F", "/")
        if not video_src or video_src.startswith("blob:"):
            raise RuntimeError("Không thể trích xuất URL video từ trang web")

        jobs[job_id]["message"] = "Đang tải video..."
        jobs[job_id]["progress"] = 15
        cookies = {cookie["name"]: cookie["value"] for cookie in driver.get_cookies()}
        headers = {
            "User-Agent": driver.execute_script("return navigator.userAgent"),
            "Referer": url,
        }
        with _requests.get(
            video_src,
            headers=headers,
            cookies=cookies,
            stream=True,
            timeout=(15, 120),
        ) as response:
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0) or 0)
            if total_size:
                _check_download_size(total_size)
            downloaded = 0
            with video_path.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 256):
                    if jobs.get(job_id, {}).get("cancel"):
                        raise RuntimeError("Đã hủy (Stop)")
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    _check_download_size(downloaded)
                    output.write(chunk)
                    if total_size:
                        pct = min(int(downloaded / total_size * 100), 99)
                        jobs[job_id]["progress"] = 15 + int(pct * 0.05)
                        jobs[job_id]["message"] = (
                            f"Đang tải: {downloaded / 1048576:.1f}/{total_size / 1048576:.1f}MB"
                        )

        file_size = video_path.stat().st_size
        jobs[job_id]["progress"] = 20
        jobs[job_id]["message"] = f"Đã tải xong ({file_size / 1048576:.1f}MB)"
        return str(video_path)
    except Exception:
        video_path.unlink(missing_ok=True)
        raise
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass


def download_with_ytdlp(job_id: str, url: str) -> str:
    """Download a single video with yt-dlp without PIPE deadlocks."""
    job_dir = _job_dir(job_id)
    output_template = str(job_dir / "source.%(ext)s")
    base_cmd = [
        "yt-dlp",
        "--user-agent",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36",
        "--referer",
        url,
    ]

    info_cmd = base_cmd + ["--no-download", "--print", "title", url]
    info_result = subprocess.run(info_cmd, capture_output=True, text=True, timeout=30)
    if info_result.returncode == 0:
        title = info_result.stdout.strip().splitlines()[0] if info_result.stdout.strip() else "Video"
        jobs[job_id]["message"] = f"Đang tải: {title[:50]}..."
        jobs[job_id]["original_name"] = title[:120]
    jobs[job_id]["progress"] = 10

    download_cmd = base_cmd + [
        "--newline",
        "-f",
        "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/bestvideo+bestaudio/best",
        "--merge-output-format",
        "mp4",
        "-o",
        output_template,
        "--no-playlist",
        "--max-filesize",
        "2G",
        url,
    ]
    log_path = job_dir / "yt-dlp.log"
    process = None
    try:
        with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
            process = subprocess.Popen(
                download_cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            jobs[job_id]["_download_process"] = process
            while process.poll() is None:
                if jobs.get(job_id, {}).get("cancel"):
                    process.terminate()
                    raise RuntimeError("Đã hủy (Stop)")
                time.sleep(1)
        if process.returncode != 0:
            error_msg = log_path.read_text(encoding="utf-8", errors="replace")
            lowered = error_msg.lower()
            if "cookies" in lowered or "login" in lowered:
                raise RuntimeError("Cần đăng nhập hoặc trang web yêu cầu cookies")
            if "not found" in lowered or "unavailable" in lowered:
                raise RuntimeError("Video không tồn tại hoặc đã bị xóa")
            if "geo" in lowered or "region" in lowered:
                raise RuntimeError("Video bị giới hạn vùng địa lý")
            raise RuntimeError(f"Lỗi tải video: {error_msg[-500:]}")

        files = sorted(job_dir.glob("source.*"), key=lambda item: item.stat().st_mtime, reverse=True)
        if not files:
            raise RuntimeError("Không tìm thấy file đã tải")
        video_path = files[0]
        _check_download_size(video_path.stat().st_size)
        file_size = video_path.stat().st_size
        jobs[job_id]["progress"] = 20
        jobs[job_id]["message"] = f"Đã tải xong ({file_size / 1048576:.1f}MB)"
        return str(video_path)
    finally:
        jobs.get(job_id, {}).pop("_download_process", None)


def download_from_url(job_id: str, url: str) -> str:
    """Download from URL, selecting browser mode for Chinese platforms."""
    valid, error = validate_download_url(url)
    if not valid:
        raise ValueError(error)
    jobs[job_id]["status"] = "downloading_video"
    jobs[job_id]["message"] = "Đang phân tích URL..."
    jobs[job_id]["progress"] = 3

    use_browser = any(domain in url.lower() for domain in CHINESE_DOMAINS)
    try:
        if use_browser:
            return download_chinese_video(job_id, url)
        return download_with_ytdlp(job_id, url)
    except Exception as exc:
        if not use_browser and "cookie" in str(exc).lower():
            jobs[job_id]["message"] = "yt-dlp thất bại, thử phương pháp trình duyệt..."
            return download_chinese_video(job_id, url)
        raise


def process_url_video(
    job_id: str,
    url: str,
    model_size: str,
    translate_langs: list,
    translate_method: str = "ai",
) -> None:
    """Download a video, then run the normal Whisper pipeline."""
    try:
        video_path = download_from_url(job_id, url)
        process_video(job_id, video_path, model_size, translate_langs, translate_method)
    except Exception as exc:
        if jobs.get(job_id):
            jobs[job_id]["status"] = "error"
            jobs[job_id]["message"] = f"Lỗi: {exc}"
            import traceback
            jobs[job_id]["traceback"] = traceback.format_exc()
