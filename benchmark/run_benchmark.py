import subprocess
import sys
import time
from pathlib import Path

from sensai.util import logging
from sensai.util.logging import datetime_tag

LOG_FILE = "session_log.txt"
ERROR_LOG_FILE = "error_log.txt"
TMUX_SESSION_PREFIX = "tianshou_"

log = logging.getLogger("benchmark")


def find_script_paths(benchmark_type: str) -> list[str]:
    """Return all Python scripts ending in _hl.py under examples/<benchmark_type>."""
    base_dir = Path(__file__).parent.parent / "examples" / benchmark_type
    glob_filter = "**/*_hl.py"
    if not base_dir.exists():
        raise FileNotFoundError(f"Directory '{base_dir}' does not exist.")

    scripts = sorted(str(p) for p in base_dir.glob(glob_filter))
    if not scripts:
        raise FileNotFoundError(f"Did not find any scripts matching '*_hl.py' in '{base_dir}'.")
    return scripts


def get_current_tmux_sessions() -> list[str]:
    """List active tmux sessions starting with 'job_'."""
    try:
        output = subprocess.check_output(["tmux", "list-sessions"], stderr=subprocess.DEVNULL)
        sessions = [
            line.split(b":")[0].decode()
            for line in output.splitlines()
            if line.startswith(TMUX_SESSION_PREFIX.encode())
        ]
        return sessions
    except subprocess.CalledProcessError:
        return []


def start_tmux_session(script_path: str) -> bool:
    """Start a tmux session running the given Python script, returning True on success."""
    # Normalize paths for Git Bash / Windows compatibility
    python_exec = sys.executable.replace("\\", "/")
    script_path = script_path.replace("\\", "/")
    session_name = TMUX_SESSION_PREFIX + Path(script_path).name.replace("_hl.py", "")
    num_experiments = 5  # always 5 experiments to get rliable evaluations

    cmd = [
        "tmux",
        "new-session",
        "-d",
        "-s",
        session_name,
        f"{python_exec} {script_path} --num_experiments {num_experiments}; echo 'Finished {script_path}'; tmux kill-session -t {session_name}",
    ]
    try:
        subprocess.run(cmd, check=True)
        log.info(
            f"Started {script_path} in session '{session_name}'. Attach with:\ntmux attach -t {session_name}"
        )
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"Failed to start {script_path} (session {session_name}): {e}")
        return False


def main(max_concurrent_sessions: int = 2, benchmark_type: str = "mujoco"):
    """
     Run the benchmarking by executing each high level script in its default configuration
     (apart from num_experiments, which will be set to 5) in its own tmux session.
     Note that if you have unclosed tmux sessions from previous runs, those will count
     towards the max_concurrent_sessions limit. You can terminate all sessions with
    `tmux kill-server`.

     :param max_concurrent_sessions: how many scripts to run in parallel, each script will
         run in a tmux session
     :param benchmark_type: mujoco or atari
     :return:
    """
    log_file = Path(__file__).parent / "logs" / f"benchmarking_{datetime_tag()}.txt"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.add_file_logger(log_file, append=False)

    log.info(
        f"=== Starting benchmark batch for '{benchmark_type}' with {max_concurrent_sessions} concurrent jobs ==="
    )
    scripts = find_script_paths(benchmark_type)
    log.info(f"Found {len(scripts)} scripts to run.")

    for i, script in enumerate(scripts, start=1):
        # Wait for free slot
        has_printed_waiting = False
        while len(get_current_tmux_sessions()) >= max_concurrent_sessions:
            if not has_printed_waiting:
                log.info(
                    f"Max concurrent sessions reached ({max_concurrent_sessions}). "
                    f"Current sessions:\n{get_current_tmux_sessions()}\nWaiting for a free slot..."
                )
                has_printed_waiting = True
            time.sleep(5)

        log.info(f"Starting script {i}/{len(scripts)}")
        session_started = start_tmux_session(script)
        if session_started:
            time.sleep(2)  # Give tmux a moment to start the session

    log.info("All jobs have been started.")
    log.info("Use 'tmux ls' to list all active sessions.")
    log.info("Use 'tmux attach -t <session_name>' to attach to a running session.")
    log.info("===============================================================")


if __name__ == "__main__":
    logging.run_cli(main)
