"""Video download from public URLs using yt-dlp or a browser fallback."""

from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
import subprocess
import sys
import shutil
import time
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from config import MAX_DOWNLOAD_BYTES, UPLOAD_FOLDER, jobs
from services.whisper_engine import process_video


CHINESE_DOMAINS = {"douyin.com", "iesdouyin.com", "xiaohongshu.com", "kuaishou.com", "v.douyin.com"}


def clean_download_url(text: str) -> str:
    """Trích xuất và chuẩn hóa URL hợp lệ từ chuỗi văn bản (ví dụ văn bản chia sẻ Douyin)."""
    raw = str(text or "").strip()
    if not raw:
        return ""
    for part in raw.split():
        if part.startswith(("http://", "https://")):
            raw = part
            break
    else:
        m = re.search(r"https?://\S+", raw)
        if m:
            raw = m.group(0)
        elif re.match(r"^([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(/.*)?$", raw):
            raw = "https://" + raw
    raw = raw.strip(" ,.?!;:\"'()[]<>")
    return raw.strip()



def _is_douyin_media_url(url: str, mime: str = "") -> bool:
    lowered = str(url or "").lower()
    if not lowered.startswith(("http://", "https://")):
        return False
    if "douyinstatic.com" in lowered or "media-audio-" in lowered:
        return False
    host_ok = any(host in lowered for host in ("zjcdn.com", "douyinvod.com", "bytecdn", "video/tos"))
    return host_ok and ("video" in str(mime).lower() or "mime_type=video" in lowered or ".mp4" in lowered)


def _parse_douyin_media_responses(performance_logs) -> list[str]:
    urls = []
    for item in performance_logs or []:
        try:
            message = json.loads(item["message"])["message"]
            if message.get("method") != "Network.responseReceived":
                continue
            params = message.get("params") or {}
            response = params.get("response") or {}
            status = int(response.get("status") or 0)
            url = response.get("url") or ""
            mime = response.get("mimeType") or ""
            if 200 <= status < 400 and _is_douyin_media_url(url, mime):
                urls.append(url)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return list(dict.fromkeys(urls))


def _select_best_douyin_media_url(candidates) -> str:
    def score(url):
        try:
            query = parse_qs(urlparse(url).query)
            bitrate = int((query.get("br") or query.get("bt") or ["0"])[0])
        except (TypeError, ValueError):
            bitrate = 0
        combined_bonus = 1 if "media-video-" not in url.lower() else 0
        return bitrate, combined_bonus, len(url)
    valid = [url for url in candidates if _is_douyin_media_url(url, "video/mp4")]
    return max(valid, key=score) if valid else ""

def validate_download_url(url: str) -> tuple[bool, str]:
    """Validate HTTP(S) URLs and reject private-network SSRF targets."""
    try:
        clean_url = clean_download_url(url)
        if not clean_url:
            return False, "URL không hợp lệ"
        parsed = urlparse(clean_url)
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
    if MAX_DOWNLOAD_BYTES > 0 and size > MAX_DOWNLOAD_BYTES:
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

    url = clean_download_url(url)
    if not url:
        raise ValueError("URL tải video không hợp lệ")

    if "v.douyin.com" in url:
        try:
            resp = _requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"},
                allow_redirects=True,
                timeout=10,
            )
            if resp.url and resp.url.startswith("http"):
                url = resp.url
        except Exception:
            pass

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
            opts.add_argument("--autoplay-policy=no-user-gesture-required")
            opts.add_argument("--window-size=1280,900")
            opts.page_load_strategy = "eager"
            opts.set_capability("goog:loggingPrefs", {"performance": "ALL"})
            return opts

        try:
            driver = uc.Chrome(options=get_options())
        except Exception as exc:
            match = re.search(r"Current browser version is (\d+)", str(exc))
            if not match:
                raise
            driver = uc.Chrome(options=get_options(), version_main=int(match.group(1)))

        driver.set_page_load_timeout(35)
        try:
            driver.execute_cdp_cmd("Network.enable", {})
        except Exception:
            pass
        jobs[job_id]["message"] = "Đang truy cập trang video..."
        jobs[job_id]["progress"] = 8
        try:
            driver.get(url)
        except Exception as nav_err:
            if "timeout" not in str(nav_err).lower():
                raise
        def capture_media(wait_seconds=22):
            candidates = []
            first_media_at = None
            deadline = time.monotonic() + wait_seconds
            while time.monotonic() < deadline:
                if jobs.get(job_id, {}).get("cancel"):
                    raise RuntimeError("Đã hủy (Stop)")
                try:
                    driver.execute_script("document.querySelector('video')?.play().catch(()=>{})")
                    direct_sources = driver.execute_script(
                        "return [...document.querySelectorAll('video')].map(v => v.currentSrc || v.src || '')"
                    ) or []
                    candidates.extend(src for src in direct_sources if _is_douyin_media_url(src, "video/mp4"))
                    resources = driver.execute_script(
                        "return performance.getEntriesByType('resource').map(e => e.name)"
                    ) or []
                    candidates.extend(src for src in resources if _is_douyin_media_url(src, "video/mp4"))
                except Exception:
                    pass
                try:
                    candidates.extend(_parse_douyin_media_responses(driver.get_log("performance")))
                except Exception:
                    pass
                candidates = list(dict.fromkeys(candidates))
                if candidates and first_media_at is None:
                    first_media_at = time.monotonic()
                if first_media_at is not None and time.monotonic() - first_media_at >= 1.5:
                    break
                time.sleep(0.5)
            return candidates

        media_candidates = capture_media()
        for retry in range(2):
            if media_candidates:
                break
            jobs[job_id]["message"] = f"Douyin chưa cấp stream, đang tải lại trang ({retry + 2}/3)..."
            try:
                driver.get(url + ("&" if "?" in url else "?") + f"studio_retry={int(time.time())}")
            except Exception as nav_err:
                if "timeout" not in str(nav_err).lower():
                    continue
            media_candidates = capture_media()

        title = (driver.title or "Video").split(" - ")[0].strip()[:80]
        if title and "抖音精选" not in title:
            jobs[job_id]["original_name"] = title
        jobs[job_id]["message"] = f"Đang tải: {(jobs[job_id].get('original_name') or title or 'video')[:40]}..."
        jobs[job_id]["progress"] = 12

        video_src = _select_best_douyin_media_url(media_candidates)
        if not video_src:
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
                video_src = _select_best_douyin_media_url([url.replace("\\u002F", "/") for url in urls])
        if not video_src:
            raise RuntimeError("Chrome đã mở trang nhưng không bắt được luồng video CDN")

        jobs[job_id]["message"] = "Đang tải video..."
        jobs[job_id]["progress"] = 15
        cookies = {cookie["name"]: cookie["value"] for cookie in driver.get_cookies()}
        headers = {
            "User-Agent": driver.execute_script("return navigator.userAgent"),
            "Referer": "https://www.douyin.com/",
            "Accept-Encoding": "identity",
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
    info_result = subprocess.run(info_cmd, capture_output=True, text=True, timeout=60)
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
    ]
    if MAX_DOWNLOAD_BYTES > 0:
        download_cmd.extend(["--max-filesize", f"{MAX_DOWNLOAD_BYTES}"])
    download_cmd.append(url)
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



def download_douyin_master(job_id: str, url: str) -> str:
    """Tải video Master gốc không nén chất lượng cao nhất trực tiếp từ CDN Douyin."""
    spider_dir = Path(__file__).resolve().parent / "douyin_monitor" / "spider"
    if spider_dir.exists() and str(spider_dir.resolve()) not in sys.path:
        sys.path.insert(0, str(spider_dir.resolve()))

    import requests as _requests
    from services.douyin_monitor.crawler import get_douyin_auth, download_master_video
    from dy_apis.douyin_api import DouyinAPI, parse_aweme_id

    # 1. Trích xuất URL sạch nếu người dùng dán kèm cả đoạn văn bản chia sẻ của Douyin
    clean_input = clean_download_url(url)

    # 2. Xử lý link rút gọn v.douyin.com -> redirect tới link video đầy đủ
    if "v.douyin.com" in clean_input:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }
        r = _requests.get(clean_input, headers=headers, allow_redirects=True, timeout=10)
        clean_input = r.url

    aweme_id, canonical_url = parse_aweme_id(clean_input)
    auth = get_douyin_auth(force_refresh_uifid=False)
    if not auth:
        raise RuntimeError("Không khởi tạo được Douyin Auth credentials từ .env")

    jobs[job_id]["message"] = f"🛰️ Đang lấy thông tin video Douyin Master ({aweme_id})..."
    jobs[job_id]["progress"] = 8

    res = DouyinAPI.get_work_info(auth, canonical_url)
    aweme_detail = res.get("aweme_detail") if isinstance(res, dict) else None
    if not aweme_detail:
        raise RuntimeError(f"Không lấy được thông tin chi tiết của video Douyin ({aweme_id})")

    title = (aweme_detail.get("desc") or f"douyin_{aweme_id}").strip()[:80]
    jobs[job_id]["original_name"] = title
    jobs[job_id]["message"] = f"⬇️ Đang tải video gốc Master Douyin: {title[:40]}..."
    jobs[job_id]["progress"] = 12

    dest_dir = _job_dir(job_id)
    downloaded_file = download_master_video(aweme_detail, dest_dir)
    if not downloaded_file or not os.path.exists(downloaded_file):
        raise RuntimeError("Không thể tải file video gốc Master từ Douyin CDN")

    target_video = dest_dir / "source.mp4"
    if str(Path(downloaded_file).resolve()) != str(target_video.resolve()):
        shutil.copy2(downloaded_file, target_video)

    _check_download_size(target_video.stat().st_size)
    file_size = target_video.stat().st_size
    jobs[job_id]["progress"] = 20
    jobs[job_id]["message"] = f"Đã tải xong video Master Douyin ({file_size / 1048576:.1f}MB)"
    print(f"✅ [Downloader] Đã tải thành công Master video Douyin: {target_video} ({file_size / 1048576:.1f}MB)")
    return str(target_video)


def download_from_url(job_id: str, url: str) -> str:
    """Download from URL, ưu tiên dùng Douyin Master API cho các link Douyin."""
    url = clean_download_url(url)
    valid, error = validate_download_url(url)
    if not valid:
        raise ValueError(error)
    jobs[job_id]["status"] = "downloading_video"
    jobs[job_id]["message"] = "Đang phân tích URL..."
    jobs[job_id]["progress"] = 3

    # 1. Nếu là link Douyin, tải trực tiếp file gốc Master không nén qua Douyin Spider API
    is_douyin = any(domain in url.lower() for domain in ("douyin.com", "iesdouyin.com", "v.douyin.com"))
    if is_douyin:
        try:
            return download_douyin_master(job_id, url)
        except Exception as exc:
            detail = str(exc)
            fallback_enabled = os.getenv("DOUYIN_BROWSER_FALLBACK", "0").strip().lower() in {"1", "true", "yes", "on"}
            if fallback_enabled:
                print(f"[Downloader] Douyin Master API thất bại ({detail}), dùng Chrome fallback theo cấu hình...")
                jobs[job_id]["message"] = "Douyin Master API lỗi; đang dùng Chrome fallback theo cấu hình..."
                return download_chinese_video(job_id, url)
            auth_hint = (
                "Douyin Master API bị từ chối xác thực (Argus/403). "
                "Hãy cập nhật phiên đăng nhập trong .env: DY_COOKIES; và nếu bộ ký hiện tại sử dụng, "
                "cập nhật đồng bộ DY_TICKET, DY_TS_SIGN, DY_CLIENT_CERT, DY_PRIVATE_KEY. "
                "Sau đó restart Web/Bot và thử lại. Chrome fallback đang tắt để bảo toàn video gốc chất lượng cao."
            )
            jobs[job_id]["message"] = auth_hint
            raise RuntimeError(auth_hint) from exc

    # 2. Đối với các nền tảng khác
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
    translation_mode: str = "movie",
) -> None:
    """Download a video, then run the normal Whisper pipeline."""
    try:
        video_path = download_from_url(job_id, url)
        process_video(job_id, video_path, model_size, translate_langs, translate_method, translation_mode=translation_mode)
    except Exception as exc:
        if jobs.get(job_id):
            jobs[job_id]["status"] = "error"
            jobs[job_id]["message"] = f"Lỗi: {exc}"
            import traceback
            jobs[job_id]["traceback"] = traceback.format_exc()
