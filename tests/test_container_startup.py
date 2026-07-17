"""Static contracts for dependency-free container readiness configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_compose_uses_dependency_free_readiness_probe() -> None:
    compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "http://localhost:8000/readyz" in compose
    assert "urllib.request.urlopen" in compose
    assert '"curl"' not in compose


def test_frontend_public_api_url_is_available_at_build_time() -> None:
    dockerfile = (PROJECT_ROOT / "frontend" / "Dockerfile").read_text(encoding="utf-8")
    compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "ARG NEXT_PUBLIC_API_URL" in dockerfile
    assert "ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}" in dockerfile
    assert "args:" in compose
    assert "NEXT_PUBLIC_API_URL: ${NEXT_PUBLIC_API_URL:-http://localhost:8000}" in compose
