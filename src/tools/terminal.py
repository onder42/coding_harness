import selectors
import subprocess
import time
from pydantic import BaseModel, Field
from src.guardrails.security import (
    check_command_safety,
    is_within_workspace,
    request_human_approval,
)
from src.tools.base import BaseTool


class PersistentBashSandbox:

    def __init__(self, initial_dir: str, timeout_seconds: float = 20.0):
        self.workspace_root = initial_dir
        self.timeout_seconds = timeout_seconds
        self.sentinel = "---COMMAND_FINISHED_SENTINEL---"
        self._start_process()

    def _start_process(self):
        self.process = subprocess.Popen(
            ["bash", "--noprofile", "--norc"],
            cwd=self.workspace_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)

    def _restart_process(self):
        self.close()
        self._start_process()

    def execute_command(self, command: str) -> str:
        command = command.strip()
        if not command:
            return "Error: Empty command."

        full_input = f'{command}\necho "$? {self.sentinel}"\n'
        self.process.stdin.write(full_input)
        self.process.stdin.flush()

        output_lines = []
        exit_code = 0
        deadline = time.monotonic() + self.timeout_seconds

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._restart_process()
                return (
                    f"Error: Command timed out after {self.timeout_seconds}s. "
                    "Background shell was restarted in workspace root."
                )

            events = self.selector.select(timeout=remaining)
            if not events:
                continue

            line = self.process.stdout.readline()
            if not line:
                self._restart_process()
                return "Error: Shell process terminated unexpectedly and was restarted."

            if self.sentinel in line:
                prefix = line.split(self.sentinel)[0].strip()
                if prefix.isdigit():
                    exit_code = int(prefix)
                break

            output_lines.append(line)

        output = "".join(output_lines).strip()

        current_dir = self._raw_pwd()
        if not is_within_workspace(current_dir, self.workspace_root):
            self._raw_cd(self.workspace_root)
            return (
                f"Security Warning: Command attempted to navigate outside workspace "
                f"('{current_dir}'). Shell reset to '{self.workspace_root}'."
            )

        if exit_code != 0:
            return f"[Exit Code: {exit_code}]\n{output}".strip()

        return output if output else "(Command succeeded with no output)"

    def _raw_pwd(self) -> str:
        self.process.stdin.write(f"pwd\necho '{self.sentinel}'\n")
        self.process.stdin.flush()
        lines = []
        while True:
            line = self.process.stdout.readline()
            if not line or self.sentinel in line:
                break
            lines.append(line.strip())
        return "".join(lines)

    def _raw_cd(self, target_dir: str):
        self.process.stdin.write(f"cd '{target_dir}'\necho '{self.sentinel}'\n")
        self.process.stdin.flush()
        while True:
            line = self.process.stdout.readline()
            if not line or self.sentinel in line:
                break

    def get_cwd(self) -> str:
        return self._raw_pwd()

    def close(self):
        if hasattr(self, "selector"):
            self.selector.unregister(self.process.stdout)
            self.selector.close()
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=2)


class BashArgs(BaseModel):
    command: str = Field(..., description="The bash shell command to execute.")


class BashTool(BaseTool):
    name = "run_bash"
    description = "Run a bash command in the persistent background shell."
    args_schema = BashArgs

    def __init__(self, sandbox: PersistentBashSandbox):
        self.sandbox = sandbox

    def execute(self, command: str) -> str:
        is_safe, safety_msg = check_command_safety(command)
        if not is_safe:
            print(f"\n[SECURITY BLOCK] {safety_msg}")
            return safety_msg

        approved, approval_msg = request_human_approval("Terminal Access", command)
        if not approved:
            return approval_msg

        print("🚀 Executing command...")
        return self.sandbox.execute_command(command)
