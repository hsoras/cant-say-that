# convo_logger.py
"""
Per-run conversation logger.
Creates a timestamped file in main/logs/ capturing the full
attacker ↔ target ↔ judge interaction for every turn.
"""
import os
from datetime import datetime
from pathlib import Path


class ConvoLogger:
    """Writes a structured log of every turn's attacker/target/judge exchange."""

    def __init__(self, logs_dir: Path | None = None):
        if logs_dir is None:
            logs_dir = Path(__file__).resolve().parent.parent / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = logs_dir / f"run_{ts}.log"
        self._f = open(self._path, "w", encoding="utf-8")

        # Write header
        self._write(f"{'='*60}")
        self._write(f"  LLM-HACKER RUN LOG — {datetime.now().isoformat()}")
        self._write(f"  Backends: ATTACKER={os.environ.get('ATTACKER_BACKEND','?')} "
                     f"JUDGE={os.environ.get('JUDGE_BACKEND','?')} "
                     f"TARGET={os.environ.get('TARGET_BACKEND','?')}")
        self._write(f"{'='*60}\n")

    # ── public API ──

    def log_run_start(self, run: int, max_runs: int, goal: str, h_skill: str, h_skill_name: str):
        self._write(f"\n{'#'*60}")
        self._write(f"# RUN {run}/{max_runs}  |  Goal: {goal}")
        self._write(f"# Overarching H-skill: {h_skill} ({h_skill_name})")
        self._write(f"{'#'*60}\n")

    def log_turn_start(self, turn: int, max_turns: int, h_skills: list[str], l_skills: list[str], chain: str):
        self._write(f"\n{'─'*60}")
        self._write(f"TURN {turn}/{max_turns}")
        self._write(f"  H-skills: {', '.join(h_skills)}")
        self._write(f"  L-skills: {', '.join(l_skills)}")
        self._write(f"  Chain:    {chain}")
        self._write(f"{'─'*60}")

    def log_attacker(self, prompt_to_attacker: str, attacker_output: str):
        self._write(f"\n[ATTACKER INPUT]")
        self._write(prompt_to_attacker)
        self._write(f"\n[ATTACKER OUTPUT]")
        self._write(attacker_output)

    def log_target(self, payload_to_target: str, target_response: str):
        self._write(f"\n[TARGET INPUT]")
        self._write(payload_to_target)
        self._write(f"\n[TARGET OUTPUT]")
        self._write(target_response)

    def log_judge(self, goal: str, target_response: str, score: float):
        self._write(f"\n[JUDGE INPUT]")
        self._write(f"  Goal: {goal}")
        self._write(f"  Target response: {target_response}")
        self._write(f"\n[JUDGE SCORE] {score:.2f}")

    def log_turn_result(self, score: float, status: str):
        self._write(f"\n>>> Turn result: score={score:.2f} — {status}")

    def log_run_end(self, chain: str, best_score: float, success: bool):
        self._write(f"\n{'#'*60}")
        self._write(f"# RUN ENDED — chain={chain}  best_score={best_score:.2f}  success={success}")
        self._write(f"{'#'*60}\n")

    def close(self):
        if self._f and not self._f.closed:
            self._f.close()

    @property
    def path(self) -> Path:
        return self._path

    # ── internal ──

    def _write(self, text: str):
        self._f.write(text + "\n")
        self._f.flush()
