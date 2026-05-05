def _print_turn_trace_header(turn: int) -> None:
    print("\n" + "=" * 60)
    print(f"🔎 TURN {turn} TRACE")
    print("=" * 60)


def _print_attacker_input(history: list[dict[str, str]], strategy_text: str) -> None:
    print("\n[INPUT TO ATTACKER LLM]")
    print("- History:")
    for idx, msg in enumerate(history, start=1):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        print(f"  {idx:02d}. {role.upper()}: {content}")
    print("\n- Strategy Text:")
    print(strategy_text)


def _manual_target(payload: str) -> str:
    print("\n" + "=" * 40)
    print("📋 AUTO-GENERATED PAYLOAD (PASTE TO TARGET):")
    print(payload)
    print("=" * 40)
    print("\nPaste target response below (Type 'DONE' on a new line when finished):")
    lines = []
    while True:
        line = input()
        if line.strip().upper() == "DONE":
            break
        lines.append(line)
    return "\n".join(lines).strip()