from __future__ import annotations

import argparse

import uvicorn

from aster.core.config import load_settings
from aster.core.lifecycle import create_application
from aster.core.process_title import build_aster_process_title, set_process_title


def main() -> None:
    parser = argparse.ArgumentParser(description="Aster inference runtime")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    bootstrap_settings = load_settings(args.config)
    set_process_title(build_aster_process_title(bootstrap_settings))

    app = create_application(args.config)
    settings = app.state.container.settings
    uvicorn.run(
        app,
        host=settings.api.host,
        port=settings.api.port,
        log_level=settings.logging.level.lower(),
    )


if __name__ == "__main__":
    main()
