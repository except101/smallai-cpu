"""Local training dashboard; child processes use the existing training CLI."""

from datetime import datetime, timedelta, timezone
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, urlsplit
import webbrowser

from .model import ModelConfig
from .tokenizer import LocalTokenizer
from .training import checkpoint_metadata
from .public_data import PreparationCancelled, prepare_public_data
from .question_bank import (MAX_UPLOAD_BYTES, TEMPLATE, prepare_question_bank,
                            preview_question_bank, question_package)


ACTIVE = {"starting", "running", "stopping"}
WEB = Path(__file__).with_name("web")


class BusyError(ValueError):
    pass


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


class TrainingManager:
    def __init__(self, workspace):
        self.workspace = Path(workspace).resolve()
        self.control = self.workspace / "outputs/gui/_state"
        self.control.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.jobs = {}
        self.active_id = None
        self.process = None
        self.worker = None
        self.closing = False
        self.data_cancel = threading.Event()
        self.data_worker = None
        data_state_path = self.control / "public_data_state.json"
        self.data_job = read_json(data_state_path) if data_state_path.is_file() else {"status": "idle"}
        if self.data_job.get("status") in ("running", "cancelling"):
            self.data_job["status"] = "interrupted"
        for path in sorted(self.control.glob("*.json")):
            try:
                job = read_json(path)
                if job["status"] in ACTIVE:
                    job["status"] = "interrupted"
                job.update(_offset=0, _series=[], _logs=[])
                self._refresh_metrics(job)
                log = self.control / f"{job['id']}.log"
                if log.is_file():
                    job["_logs"] = log.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
                self.jobs[job["id"]] = job
            except (ValueError, OSError, KeyError, TypeError):
                continue

    def _path(self, value, must_exist=True):
        if not isinstance(value, str) or not value:
            raise ValueError("请选择有效的本地文件或目录")
        path = (self.workspace / value).resolve()
        if not path.is_relative_to(self.workspace):
            raise ValueError("文件必须位于当前项目目录内")
        if must_exist and not path.exists():
            raise ValueError(f"找不到文件或目录：{value}")
        return path

    def _relative(self, path):
        return Path(path).relative_to(self.workspace).as_posix()

    def _persist(self, job):
        atomic_json(self.control / f"{job['id']}.json", {k: v for k, v in job.items() if not k.startswith("_")})

    def _refresh_metrics(self, job):
        path = self._path(job["output"], False) / "metrics.jsonl"
        if not path.is_file():
            return
        with path.open(encoding="utf-8") as stream:
            stream.seek(job["_offset"])
            while True:
                line = stream.readline()
                if not line or not line.endswith("\n"):
                    break
                position = stream.tell()
                try:
                    record = json.loads(line)
                except ValueError:
                    job["_offset"] = position
                    continue
                job["latest"] = record
                if not job["_series"] or record["step"] > job["_series"][-1]["step"]:
                    job["_series"].append({k: record[k] for k in ("step", "loss", "elapsed_seconds", "validation_loss") if k in record})
                    if len(job["_series"]) > 1500:
                        # Full data stays on disk; bound the browser's in-memory curve.
                        points = job["_series"]
                        job["_series"] = points[::2]
                        if job["_series"][-1] is not points[-1]:
                            job["_series"].append(points[-1])
                job["_offset"] = position

    def bootstrap(self):
        tokenizer_path = self.workspace / "data/tokenizer.json"
        vocab = LocalTokenizer(tokenizer_path).vocab_size if tokenizer_path.is_file() else None
        datasets = []
        for path in sorted((self.workspace / "data/prepared").glob("**/manifest.json")):
            try:
                meta = read_json(path)
                token_path = meta.get("tokenizer_path", "data/tokenizer.json")
                datasets.append({"path": self._relative(path.parent), "stage": meta["stage"],
                                 "block_size": meta["block_size"], "splits": meta["splits"],
                                 "display_name": meta.get("display_name"), "source_type": meta.get("source_type", "demo"),
                                 "vocab_size": meta["vocab_size"], "tokenizer_path": token_path,
                                 "tokenizer_sha256": meta["tokenizer_sha256"],
                                 "tokenizer_ready": self._path(token_path, False).is_file(),
                                 "sources": meta.get("sources", []), "package": meta.get("package")})
            except (OSError, ValueError, KeyError):
                continue
        models = []
        for key, title in (("cpu_22m", "标准模型"), ("smoke", "快速检查模型")):
            cfg = ModelConfig.load(key)
            if vocab:
                cfg.vocab_size = vocab
            models.append({"id": key, "title": title, "parameters": cfg.parameter_count(),
                           "layers": cfg.num_layers, "max_seq_len": cfg.max_seq_len,
                           "hidden_size": cfg.hidden_size,
                           "non_embedding_parameters": cfg.parameter_count() - cfg.vocab_size * cfg.hidden_size})
        checkpoints = []
        for path in sorted((self.workspace / "outputs").glob("**/last.pt"), reverse=True):
            try:
                cfg = read_json(path.parent / "config.json")
                info = checkpoint_metadata(path)
                summary_path = path.parent / "summary.json"
                summary = read_json(summary_path) if summary_path.is_file() else {}
                checkpoints.append({"path": self._relative(path),
                                    "label": self._relative(path.parent),
                                    "parameters": ModelConfig(**cfg).parameter_count(),
                                    "step": summary.get("step"), "tokenizer_sha256": info["tokenizer_sha256"]})
            except (OSError, ValueError, KeyError):
                continue
        preferred = "outputs/cpu_demo/pretrain/last.pt"
        default_checkpoint = next((c["path"] for c in checkpoints if c["path"] == preferred),
                                  checkpoints[0]["path"] if checkpoints else None)
        datasets.sort(key=lambda d: ({"question_bank": 0, "public": 1}.get(d["source_type"], 2), d["path"]))
        return {"workspace": str(self.workspace), "models": models, "datasets": datasets,
                "checkpoints": checkpoints[:80], "default_checkpoint": default_checkpoint,
                "default_threads": min(4, os.cpu_count() or 1), "max_threads": min(32, os.cpu_count() or 1),
                "tokenizer_ready": vocab is not None, "data_job": self.data_job}

    def data_busy(self):
        return self.data_job.get("status") in ("running", "cancelling")

    def prepare_public(self, body):
        options = {key: self._integer(body, key, default, 20, 20000)
                   for key, default in (("web_docs", 5000), ("chat_rows", 1500), ("tool_rows", 1500))}
        return self._prepare_data("public", "正在连接公开数据源", prepare_public_data, options)

    def preview_questions(self, body):
        with self.lock:
            if self.closing or self.active_id or self.data_busy():
                raise BusyError("请先结束当前训练或数据准备任务")
        return preview_question_bank(self.workspace, body.get("filename"), body.get("content"))

    def prepare_questions(self, body):
        identifier = body.get("import_id")
        question_package(self.workspace, identifier)
        token_path = body.get("tokenizer_path")
        if token_path is not None:
            token_path = self._path(token_path)
            LocalTokenizer(token_path)
        return self._prepare_data("question_bank", "正在准备上传的题库", prepare_question_bank,
                                  {"import_id": identifier, "tokenizer_path": token_path})

    def _prepare_data(self, kind, message, prepare, options):
        with self.lock:
            if self.closing or self.active_id or self.data_busy():
                raise BusyError("请先结束当前训练或数据准备任务")
            self.data_cancel.clear()
            self.data_job = {"status": "running", "kind": kind, "phase": "sources", "message": message}
            json_file_path = self.control / "public_data_state.json"
            atomic_json(json_file_path, self.data_job)

            def progress(**value):
                with self.lock:
                    self.data_job.update({"current": None, "total": None, "unit": None, **value})
                    atomic_json(json_file_path, self.data_job)

            def work():
                try:
                    result = prepare(self.workspace, **options, progress=progress, cancel=self.data_cancel)
                    with self.lock:
                        self.data_job.update(status="completed", result=result,
                                             message="题库已准备好" if kind == "question_bank" else "公开数据已准备好")
                except PreparationCancelled as error:
                    with self.lock:
                        self.data_job.update(status="cancelled", message=str(error))
                except Exception as error:
                    with self.lock:
                        self.data_job.update(status="failed", message=str(error))
                finally:
                    with self.lock:
                        atomic_json(json_file_path, self.data_job)

            self.data_worker = threading.Thread(target=work, daemon=True)
            self.data_worker.start()
            return self.data_job.copy()

    def cancel_public(self):
        with self.lock:
            if not self.data_busy():
                raise ValueError("没有正在准备的数据")
            self.data_cancel.set()
            self.data_job.update(status="cancelling", message="正在取消，已保存的数据会保留")
            return self.data_job.copy()

    @staticmethod
    def _integer(body, name, default, minimum, maximum):
        value = body.get(name, default)
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"{name} 必须是 {minimum} 到 {maximum} 之间的整数")
        return value

    def start(self, body):
        with self.lock:
            if self.closing or self.active_id or self.data_busy():
                raise BusyError("已有训练正在运行，请先停止并保存")
            stage, model = body.get("stage", "pretrain"), body.get("model", "cpu_22m")
            if stage not in ("pretrain", "sft") or model not in ("cpu_22m", "smoke"):
                raise ValueError("训练阶段或模型配置无效")
            default_data = f"data/prepared/{stage}"
            data = self._path(body.get("data", default_data))
            meta = read_json(data / "manifest.json")
            if meta["stage"] != stage:
                raise ValueError("所选数据集不适用于这个训练阶段")
            token_path = meta.get("tokenizer_path", "data/tokenizer.json")
            tokenizer = LocalTokenizer(self._path(token_path))
            if tokenizer.fingerprint != meta["tokenizer_sha256"]:
                raise ValueError("数据集的分词器已改变，请重新准备数据")
            block_size = int(meta["block_size"]) if stage == "sft" else self._integer(body, "block_size", 128, 16, 512)
            learning_rate = body.get("learning_rate", 0.0001 if stage == "sft" else 0.0003)
            if type(learning_rate) not in (int, float) or not math.isfinite(learning_rate) or not 1e-7 <= learning_rate <= 0.01:
                raise ValueError("学习率必须在 0.0000001 到 0.01 之间")
            source = None
            if stage == "sft":
                source_path = self._path(body.get("init_from"))
                if source_path.suffix != ".pt":
                    raise ValueError("初始权重必须是本项目的 .pt checkpoint")
                if checkpoint_metadata(source_path)["tokenizer_sha256"] != tokenizer.fingerprint:
                    raise ValueError("所选权重使用不同分词器，请先用这份数据完成从零预训练")
                source = self._relative(source_path)
            stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
            identifier = stamp + "_" + secrets.token_hex(3)
            params = {"stage": stage, "model": model, "data": self._relative(data),
                      "steps": self._integer(body, "steps", 100, 1, 1_000_000),
                      "block_size": block_size,
                      "batch_size": self._integer(body, "batch_size", 1, 1, 8),
                      "accumulation": self._integer(body, "accumulation", 1, 1, 32),
                      "threads": self._integer(body, "threads", 4, 1, min(32, os.cpu_count() or 1)),
                      "learning_rate": learning_rate, "init_from": source, "tokenizer_path": token_path}
            job = {"id": identifier, "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
                   "output": f"outputs/gui/{identifier}", "status": "starting", "params": params,
                   "latest": None, "error": None, "summary": None,
                   "_offset": 0, "_series": [], "_logs": []}
            self.jobs[identifier] = job
            self._launch(job, resume=False)
            return self._snapshot(job)

    def _launch(self, job, resume):
        (self.control / f"{job['id']}.stop").unlink(missing_ok=True)
        job["status"], job["error"] = "starting", None
        self.active_id = job["id"]
        self._persist(job)
        self.worker = threading.Thread(target=self._run, args=(job, resume), daemon=True)
        self.worker.start()

    def _command(self, job, resume):
        p = job["params"]
        args = [sys.executable, "-u", "-m", "smallai", "train", "--stage", p["stage"],
                "--data", p["data"], "--tokenizer", p.get("tokenizer_path", "data/tokenizer.json"), "--output", job["output"],
                "--config", p["model"], "--steps", str(p["steps"]), "--block-size", str(p["block_size"]),
                "--batch-size", str(p["batch_size"]), "--accumulation", str(p["accumulation"]),
                "--threads", str(p["threads"]), "--learning-rate", str(p["learning_rate"]),
                "--log-every", "1", "--eval-every", str(min(25, p["steps"])),
                "--save-every", str(min(25, p["steps"])), "--eval-batches", "2",
                "--stop-file", str(self.control / f"{job['id']}.stop")]
        if resume:
            args += ["--resume", str(self._path(job["output"]) / "last.pt")]
        elif p["init_from"]:
            args += ["--init-from", p["init_from"]]
        return args

    def _run(self, job, resume):
        try:
            environment = os.environ.copy()
            package_root = str(Path(__file__).resolve().parent.parent)
            environment["PYTHONPATH"] = os.pathsep.join(filter(None, [package_root, environment.get("PYTHONPATH")]))
            environment["PYTHONIOENCODING"] = "utf-8"
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            process = subprocess.Popen(self._command(job, resume), cwd=self.workspace,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       encoding="utf-8", errors="replace", bufsize=1,
                                       env=environment, creationflags=creationflags)
            with self.lock:
                self.process = process
            with (self.control / f"{job['id']}.log").open("a", encoding="utf-8") as log:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    with self.lock:
                        job["_logs"].append(line.rstrip())
                        job["_logs"] = job["_logs"][-100:]
                        self._refresh_metrics(job)
                        if job["status"] == "starting":
                            job["status"] = "running"
            code = process.wait()
            with self.lock:
                self._refresh_metrics(job)
                summary_path = self._path(job["output"], False) / "summary.json"
                summary = read_json(summary_path) if summary_path.is_file() else None
                job["summary"] = summary
                if summary and summary["interrupted"] and code in (0, 130):
                    job["status"] = "stopped"
                elif code == 0 and summary and summary["step"] >= job["params"]["steps"]:
                    job["status"] = "completed"
                else:
                    job["status"] = "failed"
                    job["error"] = next((line for line in reversed(job["_logs"]) if line.strip()), f"训练进程退出，状态码 {code}")
        except Exception as error:
            with self.lock:
                job["status"], job["error"] = "failed", str(error)
                job["_logs"].append(str(error))
        finally:
            with self.lock:
                self.active_id, self.process = None, None
                self._persist(job)

    def stop(self):
        with self.lock:
            if not self.active_id:
                raise ValueError("当前没有正在运行的训练")
            job = self.jobs[self.active_id]
            (self.control / f"{job['id']}.stop").touch()
            job["status"] = "stopping"
            self._persist(job)
            return self._snapshot(job)

    def resume(self, identifier):
        with self.lock:
            if self.closing or self.active_id or self.data_busy():
                raise BusyError("已有训练正在运行")
            job = self.jobs.get(identifier)
            if not job or job["status"] not in ("stopped", "interrupted", "failed"):
                raise ValueError("只能继续已停止且保存过权重的训练")
            if not (self._path(job["output"], False) / "last.pt").is_file():
                raise ValueError("本次训练尚未保存 checkpoint，请重新开始")
            self._launch(job, resume=True)
            return self._snapshot(job)

    def _snapshot(self, job, details=True):
        latest = job.get("latest") or {}
        step, total = latest.get("step", 0), job["params"]["steps"]
        summary = job.get("summary") or {}
        step = max(step, summary.get("step", 0))
        elapsed = latest.get("elapsed_seconds", summary.get("elapsed_seconds", 0))
        speed = latest.get("target_tokens", 0) / elapsed if elapsed > 0 else None
        series = job["_series"]
        eta = None
        if len(series) > 1 and job["status"] in ACTIVE:
            a, b = series[max(0, len(series) - 12)], series[-1]
            if b["step"] > a["step"]:
                eta = max(0, total - step) * (b["elapsed_seconds"] - a["elapsed_seconds"]) / (b["step"] - a["step"])
        result = {k: v for k, v in job.items() if not k.startswith("_")}
        result.update(step=step, total=total, progress=min(100, step / total * 100),
                      elapsed_seconds=elapsed, tokens_per_second=speed, eta_seconds=eta,
                      checkpoint_ready=(self._path(job["output"], False) / "last.pt").is_file())
        if details:
            result.update(series=series, logs=job["_logs"])
        return result

    def state(self, identifier=None):
        with self.lock:
            selected = identifier or self.active_id or next(reversed(self.jobs), None)
            job = self.jobs.get(selected)
            if job:
                self._refresh_metrics(job)
            return {"active_id": self.active_id,
                    "run": self._snapshot(job) if job else None,
                    "data_job": self.data_job.copy(),
                    "runs": [self._snapshot(value, details=False) for value in reversed(list(self.jobs.values()))][:30]}

    def close(self, timeout=30):
        with self.lock:
            self.closing = True
            if self.active_id:
                self.stop()
            worker = self.worker
            data_worker = self.data_worker
            if self.data_busy():
                self.data_cancel.set()
        if worker and worker is not threading.current_thread():
            worker.join(timeout)
        if data_worker and data_worker is not threading.current_thread():
            data_worker.join(timeout)


def create_server(workspace, port=8765):
    manager = TrainingManager(workspace)
    csrf_token = secrets.token_urlsafe(24)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status, value, content_type="application/json; charset=utf-8"):
            body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8") if content_type.startswith("application/json") else value
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def same_host(self):
            port = self.server.server_address[1]
            return self.headers.get("Host") in (f"127.0.0.1:{port}", f"localhost:{port}")

        def do_GET(self):
            if not self.same_host():
                return self.send(403, {"error": "仅允许本机访问"})
            url = urlsplit(self.path)
            try:
                if url.path == "/api/health":
                    return self.send(200, {"app": "smallai-training-ui"})
                if url.path == "/api/bootstrap":
                    return self.send(200, {**manager.bootstrap(), "csrf_token": csrf_token})
                if url.path == "/api/state":
                    identifier = parse_qs(url.query).get("run_id", [None])[0]
                    return self.send(200, manager.state(identifier))
                if url.path == "/api/questions/template.csv":
                    return self.send(200, TEMPLATE.encode("utf-8-sig"), "text/csv; charset=utf-8")
                assets = {"/": ("index.html", "text/html; charset=utf-8"),
                          "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                          "/styles.css": ("styles.css", "text/css; charset=utf-8"),
                          "/favicon.svg": ("favicon.svg", "image/svg+xml")}
                if url.path in assets:
                    name, mime = assets[url.path]
                    return self.send(200, (WEB / name).read_bytes(), mime)
                return self.send(404, {"error": "页面不存在"})
            except (ValueError, OSError, KeyError) as error:
                return self.send(400, {"error": str(error)})

        def do_POST(self):
            origin = self.headers.get("Origin")
            allowed_origin = f"http://{self.headers.get('Host')}"
            if not self.same_host() or (origin and origin != allowed_origin) or not hmac.compare_digest(self.headers.get("X-SmallAI-Token", ""), csrf_token):
                return self.send(403, {"error": "请从本机训练界面发起操作"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                limit = ((MAX_UPLOAD_BYTES + 2) // 3 * 4 + 4096) if self.path == "/api/questions/preview" else 16384
                if not 2 <= length <= limit:
                    raise ValueError("请求内容无效或过大；题库最多 10 MiB")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("请求内容必须是对象")
                if self.path == "/api/train":
                    return self.send(200, manager.start(body))
                if self.path == "/api/data/prepare":
                    return self.send(200, manager.prepare_public(body))
                if self.path == "/api/data/cancel":
                    return self.send(200, manager.cancel_public())
                if self.path == "/api/questions/preview":
                    return self.send(200, manager.preview_questions(body))
                if self.path == "/api/questions/prepare":
                    return self.send(200, manager.prepare_questions(body))
                if self.path == "/api/stop":
                    return self.send(200, manager.stop())
                if self.path == "/api/resume":
                    return self.send(200, manager.resume(body.get("run_id")))
                if self.path == "/api/shutdown":
                    with manager.lock:
                        if manager.active_id or manager.data_busy():
                            raise BusyError("请先停止并保存训练")
                        manager.closing = True
                    self.send(200, {"status": "closing"})
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                    return
                return self.send(404, {"error": "操作不存在"})
            except BusyError as error:
                return self.send(409, {"error": str(error)})
            except (ValueError, OSError, KeyError, TypeError) as error:
                return self.send(400, {"error": str(error)})

    class LocalServer(ThreadingHTTPServer):
        # On Windows SO_REUSEADDR can silently bind another app's listening port.
        allow_reuse_address = os.name != "nt"

    server = LocalServer(("127.0.0.1", port), Handler)
    server.manager = manager
    return server


def run_ui(workspace, port=8765, open_browser=True):
    if not 1 <= port <= 65535:
        raise ValueError("端口必须在 1 到 65535 之间")
    server = create_server(workspace, port)
    address = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"SmallAI 可视化训练界面：{address}", flush=True)
    if open_browser:
        webbrowser.open(address)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("正在停止训练并关闭界面服务。", flush=True)
    finally:
        server.manager.close()
        server.server_close()
