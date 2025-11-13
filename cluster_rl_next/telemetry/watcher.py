from __future__ import annotations

import argparse
import json
import os
import time


def _parse_args():
    parser = argparse.ArgumentParser(description="Tail telemetry events and print live summaries.")
    parser.add_argument("--events", required=True, help="Path to events JSONL file.")
    parser.add_argument("--poll", type=float, default=2.0, help="Polling interval seconds.")
    parser.add_argument("--once", action="store_true", help="Print current file once and exit.")
    return parser.parse_args()


def _print_event(ev):
    summary = {
        "ts": round(ev.get("ts", 0), 2),
        "actor": ev.get("actor_id"),
        "episode": ev.get("episode"),
        "step": ev.get("env_step"),
        "map": ev.get("map_id"),
        "pos": (ev.get("x"), ev.get("y")),
        "reward": round(ev.get("reward_delta", 0.0), 3),
        "eps": round(ev.get("epsilon", 0.0), 3),
    }
    print(json.dumps(summary))


def main():
    args = _parse_args()
    path = args.events
    if not os.path.exists(path):
        print(f"Events file {path} not found.")
        return
    position = 0
    while True:
        with open(path, "r", encoding="utf-8") as fh:
            fh.seek(position)
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                _print_event(event)
            position = fh.tell()
        if args.once:
            break
        time.sleep(max(0.5, args.poll))


if __name__ == "__main__":
    main()
