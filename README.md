# coding_harness
coding harness playground


Autonomous Coding Agent Harness

A progressive, build a production-grade terminal coding agent harness from scratch in Python. By building without high-level framework abstractions, this pathway strips away magic to expose the exact mechanics of ReAct tool execution loops, stateful process management, raw pseudo-terminal interactions, and automated guardrails.

5-Project Curriculum Profile


## Project 1: The Core ReAct Engine (The Brain)

• Goal: Build a functional, single-turn tool execution loop using non-persistent commands.
• What you will build: A script that takes a user prompt, connects to an LLM API (Anthropic or OpenAI), declares a basic tool schema (e.g., read_file), and successfully parses the model’s intent.
• Core Concepts to Master:
	• Structuring tool definitions using JSON schemas.
	• Capturing the model's structured tool_use blocks.
	• Manually mapping tool names to actual Python functions.
• Success Metric: The AI can successfully state, "I need to read config.json," invoke your local function, read the text, and summarize it back to you.

## Project 2: The Persistent Workspace (The Memory)

• Goal: Upgrade from disconnected, one-off commands to a live, continuous terminal environment.
• What you will build: Replace subprocess.run with a persistent background process using subprocess.Popen.
• Core Concepts to Master:
	• Keeping a single background bash or python interpreter process open across multiple LLM turns.
	• Implementing a sentinel delimiter (e.g., echo "FINISHED") so your Python code knows exactly when a background command is done printing.
	• Handling state retention (e.g., ensuring cd my_folder in Turn 1 keeps the AI inside that folder for Turn 2).
• Success Metric: The agent can run mkdir project && cd project, and a subsequent command like pwd correctly shows it is inside the new directory.

## Project 3: Terminal Realism & Human Gates (The PTY & Safety Layer)

• Goal: Fool complex tools into working properly and build a security checkpoint for the user.
• What you will build: Wrap your persistent process inside a Pseudo-Terminal (pty) and introduce a manual confirmation gate for high-risk commands.
• Core Concepts to Master:
	• Using Python’s pty or os.openpty() to generate a master/slave terminal pair.
	• Preserving terminal styling, ANSI colors, and preventing tools like git or pytest from crashing due to a lack of a TTY.
	• Intercepting the loop to prompt the human user with a [Y/n] choice before running modifying commands (like rm or git commit).
• Success Metric: Running a test command yields full terminal color outputs (red/green test indicators) inside the agent's log, and trying to delete a file pauses the loop to ask for your permission.

## Project 4: Context Management & The Gatekeeper (Sanitization)

• Goal: Prevent the agent from breaking token limits or leaking sensitive secrets over the internet.
• What you will build: Build an automated string-scrubbing middleware and basic text-truncation logic.
• Core Concepts to Master:
	• Context window pruning: Auto-summarizing or trimming massive terminal printouts (like a 10,000-line npm error log) before appending them to the AI's history.
	• Regex text masking: Scanning the raw TTY string for patterns like Password: or API keys, executing them locally, but replacing them with [REDACTED] before sending them to the API server.
• Success Metric: A command that outputs a massive file doesn't crash your LLM API context window, and typing a test password locally never appears in your outbound API logs.

## Project 5: The Autonomous Supervisor (Timeouts & Infinite Loops)

• Goal: Prevent your harness from freezing when the AI runs background servers or bad code.
• What you will build: A process monitor that checks on the background terminal's health and can force-terminate stuck jobs.
• Core Concepts to Master:
	• Asynchronous stream reading or polling loops to track how long a terminal command has been running.
	• Sending OS-level signals (like SIGINT / Ctrl+C) down the PTY line to kill runaway processes (like an infinite while true loop or an active web server).
	• Formatting interruption messages back to the AI (e.g., "Command timed out after 10 seconds. Process killed. Here is the partial log...").
• Success Metric: The agent can issue a command like python3 -m http.server or create an accidental infinite loop, and your harness cleanly cuts it off after a few seconds, hands the logs back to the AI, and keeps the conversation moving.
Which project should we target first? I can help you draft a clean skeleton script for Project 1's tool parsing loop, or we can sketch out a text-based layout showing how the sentinel string pattern fits inside the Project 2 runtime loop. Let me know how you'd like to begin!
