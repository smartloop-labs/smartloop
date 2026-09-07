"""``python -m smartloop.model_host`` — run the device owner.

Spawned by the supervisor, or by hand when debugging.  A frozen build re-execs
itself as ``<slp binary> -m smartloop.model_host``, which ``main.py`` honours
the way ``python -m`` does.
"""

import argparse
import logging

from smartloop.model_host.host import serve


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="smartloop.model_host",
        description="Own the device and serve generation to project agents",
    )
    parser.add_argument(
        "--address",
        help="unix socket path, or host:port (default: $SLP_MODEL_HOST, "
             "else <home>/model-host.sock)",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    serve(args.address)


if __name__ == "__main__":
    main()
