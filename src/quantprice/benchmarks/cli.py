"""Command line: ``quantprice-bench`` (or ``python -m quantprice.benchmarks``).

Subcommands::

    quantprice-bench generate --seed 2026 --per-topic 30 -o problems.jsonl
    quantprice-bench grade problems.jsonl answers.jsonl [--json] [--details]
    quantprice-bench answer-key problems.jsonl -o answers.jsonl
    quantprice-bench topics

``answer-key`` writes the verified answers in the answers format. Grading it must score
100%, which makes it a quick end-to-end check of a pipeline.
"""

import argparse
import json
import sys
from collections.abc import Sequence

from quantprice.benchmarks.generators import TOPICS, generate_all
from quantprice.benchmarks.grader import grade, load_answers
from quantprice.benchmarks.schema import read_jsonl, write_jsonl

__all__ = ["main"]


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="quantprice-bench",
        description="Generate and grade verified options-pricing benchmark problems.",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="write a seeded problem set as JSONL")
    gen.add_argument("--seed", type=int, required=True)
    gen.add_argument("--per-topic", type=int, required=True)
    gen.add_argument("--topics", nargs="+", choices=list(TOPICS), help="default: all topics")
    gen.add_argument("-o", "--output", required=True)

    gr = sub.add_parser("grade", help="score an answers file against a problem set")
    gr.add_argument("problems")
    gr.add_argument("answers")
    gr.add_argument("--json", action="store_true", help="print the report as JSON")
    gr.add_argument("--details", action="store_true", help="also list every non-correct item")

    key = sub.add_parser("answer-key", help="write the verified answers as an answers file")
    key.add_argument("problems")
    key.add_argument("-o", "--output", required=True)

    sub.add_parser("topics", help="list the available topics")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "generate":
            problems = generate_all(args.seed, args.per_topic, args.topics)
            n = write_jsonl(problems, args.output)
            print(f"wrote {n} problems to {args.output}")
        elif args.command == "grade":
            report = grade(read_jsonl(args.problems), load_answers(args.answers))
            if args.json:
                print(json.dumps(report.to_dict(), indent=2))
            else:
                print(report.format_table())
            if args.details:
                for r in report.results:
                    if r.status != "correct":
                        print(
                            f"{r.status:8s} {r.id}: answer {r.answer!r}, verified "
                            f"{r.verified_answer!r} +- {r.tolerance:.3g}"
                        )
        elif args.command == "answer-key":
            with open(args.output, "w", encoding="utf-8", newline="\n") as fh:
                count = 0
                for p in read_jsonl(args.problems):
                    fh.write(json.dumps({"id": p.id, "answer": p.verified_answer}) + "\n")
                    count += 1
            print(f"wrote {count} answers to {args.output}")
        else:
            for topic in TOPICS:
                print(topic)
    except (OSError, ValueError) as exc:
        print(f"quantprice-bench: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
