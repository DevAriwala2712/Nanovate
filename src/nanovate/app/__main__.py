"""``python -m nanovate.app [--port 8765] [--warm 50] [--pipeline-version v3]``: serve the API and the frontend on localhost only."""
from __future__ import annotations

import argparse
import logging


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m nanovate.app")
    ap.add_argument("--host", default="127.0.0.1", help="loopback only (127.0.0.1, localhost or ::1)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--warm", type=float, action="append", default=[], help="pre-fit the die-to-die rule at this dose (e-/px)")
    ap.add_argument("--pipeline-version", choices=("v2", "v3"), default="v3",
                    help="verifier pipeline: v3 (frozen v3 event rule, default) or v2 (post-hoc fixes)")
    return ap


def main(argv: list[str] | None = None) -> None:
    a = build_parser().parse_args(argv)
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit(f"refusing to bind {a.host!r}: the demonstrator serves localhost only")
    try:
        import uvicorn
    except ImportError as e:
        raise SystemExit('the app needs the optional extra: pip install -e ".[app]"') from e
    from .engine import Engine, check_repo_root

    check_repo_root()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from .api import create_app

    engine = Engine(version=a.pipeline_version)
    for d in a.warm:
        logging.getLogger("nanovate.app").info("fitting the die-to-die rule at dose %g on Manhattan val (one time)", d)
        engine.d2d_rules(d)
    print(f"Nanovate demonstrator ({a.pipeline_version} pipeline): http://{a.host}:{a.port}/  (synthetic data; Ctrl+C to stop)", flush=True)
    uvicorn.run(create_app(engine), host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()
