import argparse
import logging
from pathlib import Path

import paste3
from paste3 import align

logger = logging.getLogger("paste3")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=paste3.__version__)

    modules = [align]

    subparsers = parser.add_subparsers(title="Choose a command")
    subparsers.required = True

    def get_str_name(module):
        return Path(module.__file__).stem

    for module in modules:
        this_parser = subparsers.add_parser(
            get_str_name(module), description=module.__doc__
        )
        this_parser.add_argument(
            "-v", "--verbose", action="store_true", help="Increase verbosity"
        )

        module.add_args(this_parser)
        this_parser.set_defaults(func=module.main)

    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    args.func(args)


if __name__ == "__main__":
    main()
