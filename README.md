# quantprice

Options pricing library with verified ground-truth problem sets for evaluating LLM
mathematical reasoning.

**Status: under construction (Phase 0, scaffolding only).** No pricing models are implemented
yet. The full README, including the LIMITATIONS section, arrives in Phase 6.

## Development

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
ruff check . && ruff format --check . && mypy && pytest
```
