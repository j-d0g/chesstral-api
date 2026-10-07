import asyncio

import pytest
from fastapi.testclient import TestClient

from engines.catalog import ENGINE_CATALOG
from engines.anthropic_engine import AnthropicEngine
from engines.engine_factory import EngineFactory
from engines import nanogpt_paths
from models.download_checkpoint import resolve_checkpoint_name
import main


def test_nanogpt_checkpoint_paths_and_available_models(tmp_path, monkeypatch):
    monkeypatch.setattr(nanogpt_paths, "REPO_ROOT", tmp_path)
    downloaded = tmp_path / "models" / "checkpoints" / "small-8" / "ckpt.pt"
    downloaded.parent.mkdir(parents=True)
    downloaded.touch()
    legacy_downloaded_model = tmp_path / "models" / "small-8-ckpt.pt"
    legacy_downloaded_model.touch()
    legacy = tmp_path / "models" / "small-16-ckpt.pt"
    legacy.touch()
    unknown = tmp_path / "models" / "checkpoints" / "custom-model" / "ckpt.pt"
    unknown.parent.mkdir(parents=True)
    unknown.touch()

    assert nanogpt_paths.checkpoint_path("small-8") == downloaded
    assert nanogpt_paths.checkpoint_path("small-16") == legacy
    assert nanogpt_paths.available_models() == ["small-8", "small-16"]

    factory = EngineFactory()
    factory._engine_availability["nanogpt"] = False
    assert factory.is_engine_available("nanogpt")
    assert factory.get_available_engines()["nanogpt"]
    assert factory.get_engine_reason("nanogpt") == "Ready"


def test_nanogpt_factory_rejects_a_missing_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(nanogpt_paths, "REPO_ROOT", tmp_path)

    with pytest.raises(ValueError, match="checkpoint for 'small-8' was not found"):
        asyncio.run(EngineFactory().get_engine("nanogpt", "small-8"))


def test_nanogpt_factory_availability_does_not_load_a_model(monkeypatch):
    monkeypatch.setattr(nanogpt_paths, "available_models", lambda: ["small-24"])

    async def create_engine(engine_type, model=None):
        assert engine_type != "nanogpt"
        raise FileNotFoundError("test engine unavailable")

    factory = EngineFactory()
    monkeypatch.setattr(factory, "_create_engine", create_engine)
    asyncio.run(factory._test_engine_availability())

    assert factory.is_engine_available("nanogpt")
    assert factory.get_engine_reason("nanogpt") == "Ready"


def test_catalog_excludes_replicate_and_uses_valid_anthropic_model():
    assert "replicate" not in ENGINE_CATALOG
    assert ENGINE_CATALOG["anthropic"].models[0] == "claude-sonnet-4-20250514"
    assert AnthropicEngine().model == "claude-sonnet-4-20250514"
    factory_engine = asyncio.run(EngineFactory()._create_engine("anthropic"))
    assert factory_engine.model == "claude-sonnet-4-20250514"


def test_engines_endpoint_returns_status_reasons_and_no_replicate(monkeypatch):
    class Factory:
        def get_available_engines(self):
            return {name: False for name in ENGINE_CATALOG}

        def get_engine_reason(self, name):
            return (
                "No checkpoints in models/checkpoints (see models/README.md)"
                if name == "nanogpt"
                else f"Requires {ENGINE_CATALOG[name].required_env}"
                if ENGINE_CATALOG[name].required_env
                else "Stockfish binary not found"
            )

    monkeypatch.setattr(main, "engine_factory", Factory())
    monkeypatch.setattr(nanogpt_paths, "available_models", lambda: [])

    response = TestClient(main.app).get("/api/engines")

    assert response.status_code == 200
    engines = response.json()["engines"]
    assert all({"name", "models", "description", "status", "reason"} <= engine.keys() for engine in engines)
    assert "replicate" not in {engine["name"] for engine in engines}
    nanogpt = next(engine for engine in engines if engine["name"] == "nanogpt")
    assert nanogpt["models"] == []
    assert nanogpt["status"] == "unavailable"
    assert "models/README.md" in nanogpt["reason"]


def test_checkpoint_name_resolution_uses_highest_iteration_and_exact_names():
    files = [
        "checkpoints/small-8-120k_iters/ckpt.pt",
        "checkpoints/small-8-600k_iters/ckpt.pt",
        "checkpoints/small-8-300K_iters/ckpt.pt",
        "checkpoints/small-24-128K_iters/ckpt.pt",
    ]

    assert resolve_checkpoint_name("small-8", files) == "small-8-600k_iters"
    assert resolve_checkpoint_name("small-24", files) == "small-24-128K_iters"
    assert resolve_checkpoint_name("small-24-128K_iters", files) == "small-24-128K_iters"
    assert resolve_checkpoint_name("large-16", files) is None
