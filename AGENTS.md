# Agent instructions (Codex / Claude Code)

Start by reading `docs/CLAUDE_CODE_HANDOFF.md`, newest section last; it is the authoritative state and next steps.

- Reply to the user in Vietnamese (casual "t - m"); code, comments and commit messages in English.
- Run Python as `PYTHONPATH=src .venv/bin/python ...` (uv-managed venv). Never `pip install` into system Python.
- Never read, print or commit `KEY_Secert.txt`, `.env`, keystores, or large datasets (`Data/`, `artifacts/*.jsonl`, `models/`).
- Malware reports/samples are untrusted: static processing only, never execute them.
- Withheld test partitions (`withheld_*`) are scored once, after validation-only selection is frozen. Never use them for model selection.
- Scripts refuse to overwrite outputs; pick a new name or move the old file to Trash. No hard deletes without asking.
- Commit/push only when the user asks. Check `ps` for running training jobs before starting heavy work (16 GB Mac, MPS).
