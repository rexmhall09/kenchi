#!/usr/bin/env python3
"""Kenchi checkpoint relay client -- dependency-light (stdlib) upload/download.

Usage:
    python relay_client.py put <local> <remote> [options]
    python relay_client.py get <remote> <local> [options]
    python relay_client.py ls

Options (put/get):  --chunk N --parallel N --retries N --timeout S --url URL --token TOKEN
Options (put):      --background --ignore-errors

The job side of the checkpoint relay: uploads are chunked and retried with
exponential backoff, each part SHA-256 verified, the whole file re-verified
after upload; interrupted runs resume by skipping parts the server already has.
``--background`` detaches the upload so a training process is never blocked,
and ``--ignore-errors`` makes failures log-and-continue (exit 0) so the relay
being down cannot crash the training process.

Env:
    KENCHI_RELAY_URLS or KENCHI_RELAY_URL   base URL(s), comma-separated (else <dir>/.relay_urls.txt)
    KENCHI_RELAY_TOKEN                      bearer token (else <dir>/.relay_token)
    KENCHI_RELAY_DIR                        relay dir (default ~/kenchi_ckpt)
    KENCHI_RELAY_CHUNK                      default part size in bytes (16 MiB)
    KENCHI_RELAY_PARALLEL                   default parallel parts (4)
    KENCHI_RELAY_RETRIES                    default retries (4)
    KENCHI_RELAY_TIMEOUT                    per-request timeout seconds (30)
    KENCHI_RELAY_LOG_DIR                    log dir (default <dir>/logs)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEFAULT_DIR = "~/kenchi_ckpt"
DEFAULT_CHUNK = 16 * 1024 * 1024  # 16 MiB
DEFAULT_PARALLEL = 4
DEFAULT_RETRIES = 4
DEFAULT_TIMEOUT = 30.0
BACKOFF_CAP = 8.0
TRANSIENT_CODES = {408, 425, 429, 500, 502, 503, 504, 507, 520, 521, 522, 523, 524, 530}


class RelayError(Exception):
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


class TransportError(RelayError):
    """Connection-level failure; callers may fail over to another URL."""


def _int_env(name, default):
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _float_env(name, default):
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def resolve_dir():
    return Path(os.environ.get("KENCHI_RELAY_DIR") or DEFAULT_DIR).expanduser()


def resolve_token():
    t = (os.environ.get("KENCHI_RELAY_TOKEN") or "").strip()
    if t:
        return t
    try:
        return resolve_dir().joinpath(".relay_token").read_text().strip()
    except OSError:
        return None


_REFRESHED_URLS = None
_CLIENT_URLS_FILE = "relay_urls_client.txt"


def _persist_urls(urls):
    try:
        resolve_dir().joinpath(_CLIENT_URLS_FILE).write_text("\n".join(urls) + "\n")
    except OSError:
        pass


def _fetch_urls(timeout=8.0):
    """The supervisor's current public URL list, from any reachable server (R2)."""
    tok = resolve_token()
    for url in resolve_urls():
        try:
            got = _get_json(url, tok, "/urls", timeout=timeout)
            if isinstance(got, list) and got:
                return [u for u in got if isinstance(u, str) and u.strip()]
        except Exception:
            continue
    return None


def refresh_urls():
    """Refresh + persist the known URLs from any reachable server (R2); returns
    the new list or None.  The pinned bore URL is the anchor."""
    global _REFRESHED_URLS
    urls = _fetch_urls()
    if urls:
        _REFRESHED_URLS = urls
        _persist_urls(urls)
    return urls


def resolve_urls():
    """Relay base URLs, best first: URLs learned from a live server (GET /urls),
    then the env list (KENCHI_RELAY_URLS / KENCHI_RELAY_URL), then the files in
    the relay dir.  The env list is never dropped: it holds the pinned bore
    address, which is the same after every tunnel restart."""
    out = []

    def add(urls):
        for u in urls:
            u = (u or "").strip().rstrip("/")
            if u and u not in out:
                out.append(u)

    add(_REFRESHED_URLS or [])
    raw = os.environ.get("KENCHI_RELAY_URLS") or os.environ.get("KENCHI_RELAY_URL")
    if raw:
        add(raw.split(","))
    for fn in (_CLIENT_URLS_FILE, ".relay_urls.txt"):
        try:
            add(resolve_dir().joinpath(fn).read_text().splitlines())
        except OSError:
            pass
    return out


def live_urls(timeout=6.0):
    """The relay addresses that answer right now, best effort: ask a server for
    the current list (a restarted Cloudflare tunnel has a new address), then
    keep the ones whose /health replies.  Falls back to every known URL."""
    try:
        refresh_urls()
    except Exception:  # noqa: BLE001
        pass
    urls = resolve_urls()
    if len(urls) <= 1:
        return urls
    tok = resolve_token()

    def ok(u):
        try:
            _get_json(u, tok, "/health", timeout=timeout)
            return True
        except Exception:  # noqa: BLE001
            return False

    with ThreadPoolExecutor(max_workers=len(urls)) as pool:
        flags = list(pool.map(ok, urls))
    live = [u for u, f in zip(urls, flags) if f]
    return live or urls


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _backoff(attempt, base=0.5, cap=BACKOFF_CAP):
    return min(cap, base * (2 ** attempt)) + random.uniform(0, 0.25)


def _log(msg):
    log_dir = Path(os.environ.get("KENCHI_RELAY_LOG_DIR") or resolve_dir() / "logs")
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "relay_client.log", "a") as f:
            f.write("%s %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg))
    except OSError:
        pass


# -- low-level HTTP ------------------------------------------------------

def _request(base, token, method, path, headers=None, data=None, timeout=DEFAULT_TIMEOUT):
    url = base.rstrip("/") + path
    req = urllib.request.Request(url, data=data, method=method)
    if token:
        req.add_header("Authorization", "Bearer " + token.strip())
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        try:
            body = e.read()
        except Exception:
            body = b""
        detail = ""
        try:
            detail = json.loads(body).get("error", "") or ""
        except Exception:
            pass
        raise RelayError("%s %s -> HTTP %s %s" % (method, url, e.code, detail or body[:100]), code=e.code)
    except urllib.error.URLError as e:
        raise TransportError("%s %s -> %s" % (method, url, e))
    except OSError as e:
        raise TransportError("%s %s -> %s" % (method, url, e))
    return resp


def _get_json(base, token, path, timeout=DEFAULT_TIMEOUT):
    with _request(base, token, "GET", path, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw or b"{}")


# -- upload --------------------------------------------------------------

def _upload_part(base, token, qname, index, data, file_size, part_total, file_sha, timeout=DEFAULT_TIMEOUT):
    part_sha = hashlib.sha256(data).hexdigest()
    path = "/file/%s/part/%d" % (qname, index)
    headers = {
        "X-File-Size": str(file_size),
        "X-File-Sha256": file_sha,
        "X-Part-Total": str(part_total),
        "X-Part-Sha256": part_sha,
    }
    with _request(base, token, "PUT", path, headers=headers, data=data, timeout=timeout) as resp:
        resp.read()


def _upload_part_with_retry(base, token, qname, index, local, chunk, total_parts,
                            file_size, file_sha, retries, timeout):
    with open(local, "rb") as f:
        f.seek(index * chunk)
        data = f.read(chunk) or b""
    last = None
    for attempt in range(retries):
        try:
            _upload_part(base, token, qname, index, data, file_size, total_parts, file_sha, timeout=timeout)
            return
        except TransportError as e:
            last = e
        except RelayError as e:
            if e.code not in TRANSIENT_CODES:
                raise
            last = e
        time.sleep(_backoff(attempt))
    if isinstance(last, TransportError):
        # Part-level transport failure (tunnel timeout, connection refused...):
        # let `upload()` retry the same part on the next URL instead of aborting
        # the whole upload.  Parts already on the server are re-verified.
        raise last
    raise RelayError("part %d failed after %d attempts: %s" % (index, retries, last))


def _remote_file_info(base, token, qname, timeout=DEFAULT_TIMEOUT):
    """Return (sha256, size) of the remote file as computed by the relay, or
    None if it does not exist."""
    try:
        d = _get_json(base, token, "/sha/%s" % qname, timeout=max(timeout, 180.0))
        return d.get("sha256"), int(d.get("size") or 0)
    except RelayError as e:
        if e.code == 404:
            return None
        raise


def _upload_once(urls, token, local, remote, file_sha, size, total_parts,
                 chunk, parallel, retries, timeout):
    """One pass over the missing parts.  Parts are spread over every URL in
    ``urls`` (both tunnels carry data at once); a part that fails on one URL is
    tried on the others.  Returns stats once the relay confirms sha256 + size."""
    qname = urllib.parse.quote(remote, safe="")
    t0 = time.monotonic()
    info, base, last = None, None, None
    for u in urls:
        try:
            info = _get_json(u, token, "/file/%s/parts" % qname, timeout=timeout)
            base = u
            break
        except TransportError as e:
            last = e
        except RelayError as e:
            if e.code not in TRANSIENT_CODES:
                raise
            last = e
    if info is None:
        raise TransportError("no relay URL reachable: %s" % last)
    existing = set(info.get("parts") or [])
    if bool(info.get("file_exists")):
        current = _remote_file_info(base, token, qname, timeout=timeout)
        if current is not None and current[0] == file_sha and current[1] == size:
            return _stats(remote, size, time.monotonic() - t0, file_sha, base)
    todo = [i for i in range(total_parts) if i not in existing]

    def one(i):
        order = urls[i % len(urls):] + urls[:i % len(urls)]
        err = None
        for u in order:
            try:
                _upload_part_with_retry(u, token, qname, i, local, chunk, total_parts,
                                        size, file_sha, retries, timeout)
                return
            except TransportError as e:
                err = e
            except RelayError as e:
                if e.code not in TRANSIENT_CODES and e.code is not None:
                    raise
                err = e
        raise TransportError("part %d failed on every URL: %s" % (i, err))

    if todo:
        workers = min(max(1, parallel) * len(urls), len(todo))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(one, i) for i in todo]
            for f in as_completed(futs):
                f.result()
    current = None
    for u in [base] + [x for x in urls if x != base]:
        try:
            current = _remote_file_info(u, token, qname, timeout=timeout)
            base = u
            break
        except RelayError as e:
            last = e
    if current is None or current[0] != file_sha or current[1] != size:
        # stale parts of an older file with the same name can block assembly:
        # clear them so the next attempt starts clean
        try:
            with _request(base, token, "DELETE", "/file/%s?parts_only=1" % qname, timeout=timeout) as resp:
                resp.read()
        except RelayError:
            pass
        raise RelayError("remote file %r is missing or checksum mismatch after upload" % remote, code=503)
    return _stats(remote, size, time.monotonic() - t0, file_sha, ",".join(urls) if todo else base)


def _stats(name, size, duration, sha=None, url=None):
    mbps = (size / (1024.0 * 1024.0)) / duration if duration > 0 else 0.0
    d = {"name": name, "size": size, "duration_s": round(duration, 3), "MBps": round(mbps, 1)}
    if sha:
        d["sha256"] = sha
    if url:
        d["url"] = ",".join(u.split("//")[-1].split("/")[0] for u in url.split(","))
    return d


def upload_snapshot(local, remote, **kw):
    """Upload a copy of a file that is still being written (a log): the copy
    cannot change between hashing and sending."""
    import shutil
    snap = Path(tempfile.gettempdir()) / ("kenchi-snap-%d-%s" % (os.getpid(), Path(local).name))
    shutil.copyfile(str(local), str(snap))
    try:
        return upload(snap, remote, **kw)
    finally:
        try:
            snap.unlink()
        except OSError:
            pass


def upload(local, remote, base=None, token=None, chunk=None, parallel=None,
           retries=None, timeout=None):
    """Upload ``local`` to ``remote`` on the relay at ``base`` (or from env/file URLs)."""
    local = Path(local)
    if not local.is_file():
        raise RelayError("local file not found: %s" % local)
    size = local.stat().st_size
    chunk = int(chunk if chunk is not None else _int_env("KENCHI_RELAY_CHUNK", DEFAULT_CHUNK))
    parallel = int(parallel if parallel is not None else _int_env("KENCHI_RELAY_PARALLEL", DEFAULT_PARALLEL))
    retries = int(retries if retries is not None else _int_env("KENCHI_RELAY_RETRIES", DEFAULT_RETRIES))
    timeout = float(timeout if timeout is not None else _float_env("KENCHI_RELAY_TIMEOUT", DEFAULT_TIMEOUT))
    if chunk <= 0:
        raise RelayError("chunk must be > 0")
    token = token or resolve_token()
    urls = [base] if base else resolve_urls()
    if not token:
        raise RelayError("no relay token: set KENCHI_RELAY_TOKEN or ensure <dir>/.relay_token exists")
    if not urls:
        raise RelayError("no relay URLs: set KENCHI_RELAY_URL(S) or run the supervisor to write .relay_urls.txt")
    file_sha = sha256_file(local)
    total_parts = max(1, -(-size // chunk))
    last = None
    for attempt in range(max(1, retries)):
        if base is None:
            urls = live_urls()
        try:
            return _upload_once(urls, token, local, remote, file_sha, size, total_parts,
                                chunk, parallel, retries, timeout)
        except TransportError as e:
            last = e
            _log("transport error: %s" % e)
        except RelayError as e:
            if e.code not in TRANSIENT_CODES:
                raise
            last = e
        # learn the current addresses from any server that answers, then retry
        if base is None:
            refresh_urls()
        if attempt + 1 < max(1, retries):
            time.sleep(_backoff(attempt))
    raise RelayError("upload of %r failed after %d attempts: %s" % (remote, retries, last))


# -- download ------------------------------------------------------------

def _download_once(base, token, qname, remote, local, timeout):
    t0 = time.monotonic()
    with _request(base, token, "GET", "/file/%s" % qname, timeout=timeout) as resp:
        tmp = Path(str(local) + ".part")
        h = hashlib.sha256()
        n = 0
        try:
            with open(tmp, "wb") as f:
                while True:
                    b = resp.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
                    h.update(b)
                    n += len(b)
        except OSError as e:
            with_suppress_unlink(tmp)
            raise RelayError("write failed: %s" % e)
        expected_sha = resp.headers.get("X-File-Sha256")
        expected_size = int(resp.headers.get("Content-Length") or 0)
    if expected_sha and h.hexdigest() != expected_sha:
        with_suppress_unlink(tmp)
        raise RelayError("download checksum mismatch")
    if expected_size and n != expected_size:
        with_suppress_unlink(tmp)
        raise RelayError("download size mismatch: got %d, want %d" % (n, expected_size))
    os.replace(tmp, local)
    return _stats(remote, n, time.monotonic() - t0)


def with_suppress_unlink(p):
    try:
        p.unlink()
    except OSError:
        pass


def download(remote, local, base=None, token=None, retries=None, timeout=None):
    """Download ``remote`` to ``local``."""
    local = Path(local)
    qname = urllib.parse.quote(remote, safe="")
    token = token or resolve_token()
    urls = [base] if base else resolve_urls()
    retries = int(retries if retries is not None else _int_env("KENCHI_RELAY_RETRIES", DEFAULT_RETRIES))
    timeout = float(timeout if timeout is not None else _float_env("KENCHI_RELAY_TIMEOUT", DEFAULT_TIMEOUT))
    if not token:
        raise RelayError("no relay token: set KENCHI_RELAY_TOKEN or ensure <dir>/.relay_token exists")
    if not urls:
        raise RelayError("no relay URLs: set KENCHI_RELAY_URL(S) or run the supervisor to write .relay_urls.txt")
    last = None
    for attempt in range(max(1, retries)):
        for url in urls:
            try:
                return _download_once(url, token, qname, remote, local, timeout)
            except TransportError as e:
                last = e
                continue
            except RelayError as e:
                if e.code in TRANSIENT_CODES:
                    last = e
                    continue
                raise
        if base is None:
            refresh_urls()
            urls = resolve_urls()
        if attempt + 1 < max(1, retries):
            time.sleep(_backoff(attempt))
    raise RelayError("download of %r failed after %d attempts: %s" % (remote, retries, last))


# -- misc -----------------------------------------------------------------

def list_files(base=None, token=None, timeout=None):
    """Return the list of files served by the relay: [{name,size,mtime}...]."""
    token = token or resolve_token()
    urls = [base] if base else resolve_urls()
    timeout = float(timeout if timeout is not None else _float_env("KENCHI_RELAY_TIMEOUT", DEFAULT_TIMEOUT))
    for url in urls:
        try:
            return _get_json(url, token, "/files", timeout=timeout)
        except TransportError:
            continue
        except RelayError:
            raise
    # R2: refresh the URL list and retry with it
    if refresh_urls():
        for url in resolve_urls():
            try:
                return _get_json(url, token, "/files", timeout=timeout)
            except TransportError:
                continue
    raise RelayError("no reachable relay URL")


def upload_async(local, remote, callback=None, **kw):
    """Start an upload in a daemon thread; errors are logged, never raised.

    Lets a training process call relay uploads without blocking, and without
    being able to crash if the relay is down: the failure is logged and the
    process continues.
    """

    def _run():
        try:
            stats = upload_snapshot(local=local, remote=remote, **kw)
            if callback:
                try:
                    callback(stats)
                except Exception:
                    pass
            _log("uploaded %s" % remote)
        except Exception as e:  # relay is best-effort: log and continue
            _log("upload %s failed: %s" % (remote, e))

    t = threading.Thread(target=_run, name="kenchi-relay-upload", daemon=True)
    t.start()
    return t


# -- CLI ------------------------------------------------------------------

def _spawn_background(argv):
    """Detach the upload into its own process; log file under <dir>/logs or KENCHI_RELAY_LOG_DIR."""
    cmd = [sys.executable, os.path.abspath(__file__)] + [a for a in argv if a != "--background"]
    log_dir = Path(os.environ.get("KENCHI_RELAY_LOG_DIR") or (resolve_dir() / "logs"))
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        log_dir = Path(tempfile.gettempdir())
    name = "upload"
    if len(cmd) > 3:
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(cmd[3]).name) or "upload"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / ("put_%s_%s.log" % (name, stamp))
    with open(log_path, "ab") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    print("started background upload pid=%s log=%s" % (proc.pid, log_path))
    return 0


def _add_common_args(p):
    p.add_argument("--chunk", type=int, default=None)
    p.add_argument("--parallel", type=int, default=None)
    p.add_argument("--retries", type=int, default=None)
    p.add_argument("--timeout", type=float, default=None)
    p.add_argument("--url", default=None)
    p.add_argument("--token", default=None)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="relay_client.py", description="Kenchi checkpoint relay client")
    sub = parser.add_subparsers(dest="command", required=True)
    put_p = sub.add_parser("put", help="upload a file (chunked, resumable)")
    put_p.add_argument("local")
    put_p.add_argument("remote")
    _add_common_args(put_p)
    put_p.add_argument("--background", action="store_true")
    put_p.add_argument("--ignore-errors", action="store_true")
    get_p = sub.add_parser("get", help="download a file")
    get_p.add_argument("remote")
    get_p.add_argument("local")
    _add_common_args(get_p)
    get_p.add_argument("--ignore-errors", action="store_true")
    ls_p = sub.add_parser("ls", help="list files on the relay")
    ls_p.add_argument("--url", default=None)
    ls_p.add_argument("--token", default=None)
    args = parser.parse_args(argv)

    token = args.token or resolve_token()
    ignore = getattr(args, "ignore_errors", False)
    try:
        if args.command == "put":
            if args.background:
                return _spawn_background(argv if argv is not None else sys.argv[1:])
            stats = upload(args.local, args.remote, base=args.url, token=token,
                           chunk=args.chunk, parallel=args.parallel,
                           retries=args.retries, timeout=args.timeout)
            print(json.dumps(stats))
        elif args.command == "get":
            stats = download(args.remote, args.local, base=args.url, token=token,
                             retries=args.retries, timeout=args.timeout)
            print(json.dumps(stats))
        elif args.command == "ls":
            files = list_files(base=args.url, token=token)
            for f in files:
                print("%s\t%s\t%s" % (f["name"], f["size"], time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(f["mtime"]))))
            if not files:
                print("(empty)")
        return 0
    except Exception as e:
        msg = "relay %s failed: %s" % (args.command, e)
        if ignore:
            _log(msg)
            return 0
        print(msg, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
