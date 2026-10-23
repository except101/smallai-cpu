import json
from pathlib import Path
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from smallai.model import ModelConfig
from smallai.training import load_checkpoint, train
from smallai.ui import BusyError, TrainingManager, create_server


def wait_until(predicate, timeout=25):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        result = predicate()
        if result:
            return result
        time.sleep(0.04)
    raise AssertionError("Timed out waiting for local training")


def test_stop_flag_saves_checkpoint_and_can_resume(local_data, tmp_path):
    root, tokenizer = local_data
    config = tmp_path / "config.json"
    config.write_text(json.dumps(ModelConfig(vocab_size=2048, hidden_size=16, intermediate_size=32,
                                             num_layers=1, num_heads=2, num_kv_heads=1,
                                             max_seq_len=32).to_dict()), encoding="utf-8")
    flag = tmp_path / "stop.flag"
    flag.touch()
    options = dict(stage="pretrain", data=root / "pretrain", tokenizer_path=tokenizer.path,
                   output=tmp_path / "run", config_path=config, steps=4, block_size=16,
                   eval_every=4, eval_batches=1, threads=2, stop_file=flag)
    stopped = train(**options)
    assert stopped["interrupted"] and stopped["step"] == 0
    assert Path(stopped["checkpoint"]).is_file()
    flag.unlink()
    result = train(**options, resume=stopped["checkpoint"])
    assert result["step"] == 4 and not result["interrupted"]
    assert load_checkpoint(result["checkpoint"])["step"] == 4


def test_manager_reports_real_progress_and_runs_sft(ui_workspace):
    manager = TrainingManager(ui_workspace)
    try:
        first = manager.start({"model": "smoke", "steps": 4, "block_size": 64, "threads": 2})
        with pytest.raises(BusyError):
            manager.start({"model": "smoke", "steps": 4})
        wait_until(lambda: manager.state()["active_id"] is None)
        state = manager.state()
        assert state["run"]["status"] == "completed", state
        assert state["run"]["step"] == 4 and state["run"]["progress"] == 100
        assert len(state["run"]["series"]) == 4
        assert state["run"]["tokens_per_second"] > 0
        assert state["run"]["checkpoint_ready"]
        assert manager.bootstrap()["checkpoints"]
        manager.start({"stage": "sft", "model": "smoke", "steps": 1, "threads": 2,
                       "init_from": first["output"] + "/last.pt"})
        wait_until(lambda: manager.state()["active_id"] is None)
        assert manager.state()["run"]["status"] == "completed"
        assert manager.state()["run"]["params"]["block_size"] == 1023
    finally:
        manager.close()


def test_manager_stop_and_resume_preserve_experiment(ui_workspace):
    manager = TrainingManager(ui_workspace)
    try:
        job = manager.start({"model": "smoke", "steps": 10000, "block_size": 64, "threads": 2})
        wait_until(lambda: manager.state()["run"]["step"] >= 2)
        manager.stop()
        wait_until(lambda: manager.state()["active_id"] is None)
        stopped = manager.state()["run"]
        assert stopped["status"] == "stopped"
        assert 0 < stopped["progress"] < 100 and stopped["checkpoint_ready"]
        resumed = manager.resume(job["id"])
        assert resumed["id"] == job["id"]
        wait_until(lambda: manager.state()["run"]["step"] > stopped["step"])
        manager.stop()
        wait_until(lambda: manager.state()["active_id"] is None)
        final = manager.state()["run"]
        assert final["status"] == "stopped" and final["step"] > stopped["step"]
        assert len(manager.state()["runs"]) == 1
        restored = TrainingManager(ui_workspace)
        assert restored.state()["run"]["id"] == job["id"]
        assert restored.state()["run"]["step"] == final["step"]
        assert restored.state()["run"]["series"]
    finally:
        manager.close()


def test_server_serves_ui_and_rejects_invalid_start_requests(ui_workspace):
    server = create_server(ui_workspace, port=0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urlopen(base + "/") as response:
            assert response.status == 200
            assert 'id="training-progress"' in response.read().decode()
        with urlopen(base + "/api/bootstrap") as response:
            bootstrap = json.load(response)
        assert bootstrap["tokenizer_ready"]
        assert len(bootstrap["datasets"]) == 2
        headers = {"Content-Type": "application/json"}
        with pytest.raises(HTTPError) as rejected:
            urlopen(Request(base + "/api/train", b"{}", headers=headers))
        assert rejected.value.code == 403
        headers["X-SmallAI-Token"] = bootstrap["csrf_token"]
        headers["Origin"] = "https://another-site.example"
        with pytest.raises(HTTPError) as rejected:
            urlopen(Request(base + "/api/train", b"{}", headers=headers))
        assert rejected.value.code == 403
        headers["Origin"] = base
        for body in ({"stage": "unknown"}, {"data": "../outside"}, {"steps": True}):
            with pytest.raises(HTTPError) as rejected:
                urlopen(Request(base + "/api/train", json.dumps(body).encode(), headers=headers))
            assert rejected.value.code == 400
    finally:
        server.shutdown()
        server.manager.close()
        server.server_close()
        worker.join(3)


def test_server_rejects_a_port_already_used_by_another_instance(ui_workspace):
    first = create_server(ui_workspace, port=0)
    try:
        with pytest.raises(OSError):
            second = create_server(ui_workspace, port=first.server_address[1])
            second.server_close()
    finally:
        first.manager.close()
        first.server_close()
