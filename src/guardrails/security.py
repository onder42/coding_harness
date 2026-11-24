import os

FORBIDDEN_TOKENS = [
    "rm -rf /",
    "rm -rf ~",
    "rm -rf $HOME",
    "dd if=",
    ":(){ :|:& };:",
    "chmod",
    "shutdown",
    "/testbed"
]


def is_within_workspace(target_path: str, allowed_root: str) -> bool:
    """Ensures file operations stay locked inside your designated sandbox folder."""
    abs_root = os.path.abspath(allowed_root)
    abs_target = os.path.abspath(target_path)
    return os.path.commonpath([abs_root, abs_target]) == abs_root


def check_command_safety(command: str) -> tuple[bool, str]:
    """Hard security block that triggers before the human prompt even appears."""
    clean_cmd = command.strip()
    for token in FORBIDDEN_TOKENS:
        if token in clean_cmd:
            return (
                False,
                f"Access denied: Blocked dangerous token '{token}' by local security policy.",
            )
    return True, "OK"


def request_human_approval(action_title: str, details: str) -> tuple[bool, str]:
    """Visual Interception Gate requiring explicit terminal confirmation."""
    print("\n" + "=" * 50)
    print(f"AGENT REQUESTING {action_title.upper()}")
    print(f"DETAILS: {details}")
    print("=" * 50)

    try:
        choice = input("Approve execution? (y/N): ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print("\nExecution canceled by user interface.")
        return False, "User cancelled execution."

    if choice != "y":
        user_input = input("Reason : ")
        print("Action Denied. Sending refusal message to the AI.")
        return False, f"Execution Denied: The user inspected this action and rejected it. New Instruction: {user_input}"

    return True, "Approved"
