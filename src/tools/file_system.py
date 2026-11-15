import os
from pydantic import BaseModel, Field
from src.guardrails.security import is_within_workspace, request_human_approval
from src.tools.base import BaseTool
from src.tools.terminal import PersistentBashSandbox


class ReadFileArgs(BaseModel):
    file_path: str = Field(..., description="Relative or absolute path to the file.")


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read the text contents of a file inside the workspace."
    args_schema = ReadFileArgs

    def __init__(self, sandbox: PersistentBashSandbox, workspace_root: str):
        self.sandbox = sandbox
        self.workspace_root = workspace_root

    def execute(self, file_path: str) -> str:
        current_dir = self.sandbox.get_cwd()
        resolved_path = os.path.normpath(os.path.join(current_dir, file_path))

        if not is_within_workspace(resolved_path, self.workspace_root):
            return f"Security Error: '{file_path}' is outside the allowed workspace."

        if not os.path.exists(resolved_path):
            return f"Error: File '{file_path}' does not exist."

        with open(resolved_path, "r", encoding="utf-8") as f:
            return f.read()


class WriteFileArgs(BaseModel):
    file_path: str = Field(..., description="Path where the file should be written.")
    content: str = Field(..., description="Full text content to write into the file.")


class WriteFileTool(BaseTool):
    name = "write_file"
    description = "Create or overwrite a file with new content inside the workspace."
    args_schema = WriteFileArgs

    def __init__(self, sandbox: PersistentBashSandbox, workspace_root: str):
        self.sandbox = sandbox
        self.workspace_root = workspace_root

    def execute(self, file_path: str, content: str) -> str:
        current_dir = self.sandbox.get_cwd()
        resolved_path = os.path.normpath(os.path.join(current_dir, file_path))

        if not is_within_workspace(resolved_path, self.workspace_root):
            return f"Security Error: '{file_path}' is outside the allowed workspace."

        approved, approval_msg = request_human_approval(
            "File Write", f"Write {len(content)} chars to {resolved_path}"
        )
        if not approved:
            return approval_msg

        os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
        with open(resolved_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {resolved_path}"
