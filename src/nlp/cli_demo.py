import json
import sys

try:
    # Prefer package-relative import when the module is executed as part of a package
    from .command_parser import parse_command
except ImportError:
    # Fallback to direct import so the script can be run standalone (python cli_demo.py ...)
    from command_parser import parse_command


def main() -> int:
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        print("Enter command (Ctrl+C to exit):")
        try:
            text = input("> ")
        except KeyboardInterrupt:
            return 0
    result = parse_command(text)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


