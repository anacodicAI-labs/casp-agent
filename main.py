import sys


def main(argv: list[str]) -> int:
    message = argv[1] if len(argv) > 1 else "ship insulin Mumbai to Delhi"

    from agents.orchestrator import run_orchestrator

    print(run_orchestrator(message))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

