import json
import os
import subprocess
import sys
import time
from pathlib import Path

from sensai.util import logging
from sensai.util.logging import datetime_tag

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


def start_tmux_session(script_path: str, persistence_base_dir: Path | str, num_experiments: int) -> bool:
    """Start a tmux session running the given Python script, returning True on success."""
    # Normalize paths for Git Bash / Windows compatibility
    python_exec = sys.executable.replace("\\", "/")
    script_path = script_path.replace("\\", "/")
    persistence_base_dir = str(persistence_base_dir).replace("\\", "/")
    session_name = TMUX_SESSION_PREFIX + Path(script_path).name.replace("_hl.py", "")

    cmd = [
        "tmux", "new-session", "-d", "-s",
        session_name,
        f"{python_exec} {script_path} --num_experiments {num_experiments} --persistence_base_dir {persistence_base_dir}; "
        f"echo 'Finished {script_path}'; "
        f"tmux kill-session -t {session_name}",
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

def aggregate_results(benchmark_results_dir: str) -> None:
    environments = [d for d in os.listdir(benchmark_results_dir) if os.path.isdir(os.path.join(benchmark_results_dir, d))]
    for env in environments:
        experiment_dirs = [d for d in os.listdir(os.path.join(benchmark_results_dir, env)) if os.path.isdir(os.path.join(benchmark_results_dir, env, d))]
        aggregated_results = []
        for experiment_dir in experiment_dirs:
            agent_name = experiment_dir.split("Experiment")[0]
            with open(os.path.join(benchmark_results_dir, env, experiment_dir, "rliable_evaluation_test.json"), "r") as f:
                result_entries = json.load(f)
            for result_entry in result_entries:
                result_entry["agent"] = agent_name
                aggregated_results.append(result_entry)
        aggregated_results_path = os.path.join(benchmark_results_dir, env, "results.json")
        with open(aggregated_results_path, "w") as f:
            json.dump(aggregated_results, f, indent=4)
        log.info(f"Aggregated results for environment '{env}' written to '{aggregated_results_path}'.")


def main(max_concurrent_sessions: int = 2, benchmark_type: str = "mujoco", num_experiments: int = 10, max_scripts: int = -1) -> None:
    """
     Run the benchmarking by executing each high level script in its default configuration
     (apart from num_experiments, which will be set to 5) in its own tmux session.
     Note that if you have unclosed tmux sessions from previous runs, those will count
     towards the max_concurrent_sessions limit. You can terminate all sessions with
    `tmux kill-server`.

     :param max_concurrent_sessions: how many scripts to run in parallel, each script will
         run in a tmux session
     :param benchmark_type: mujoco or atari
     :param num_experiments: number of experiments to run per script
     :param max_scripts: maximum number of scripts to run, -1 for all. Set this to a low number for testing.
     :return:
    """
    persistence_base_dir = Path(__file__).parent / "logs" / benchmark_type / datetime_tag()

    # file logger for the global benchmarking logs, each individual experiment will log to its own file
    log_file = persistence_base_dir / f"benchmarking_run.txt"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.add_file_logger(log_file, append=False)

    scripts = find_script_paths(benchmark_type)
    if max_scripts > 0:
        log.info(f"Limiting to first {max_scripts}/{len(scripts)} scripts.")
        scripts = scripts[:max_scripts]

    log.info(
        f"=== Starting benchmark batch for '{benchmark_type}' ({len(scripts)} scripts) "
        f"with {max_concurrent_sessions} concurrent jobs ==="
    )
    for i, script in enumerate(scripts, start=1):
        # Wait for free slot
        has_printed_waiting_message = False
        while len(get_current_tmux_sessions()) >= max_concurrent_sessions:
            if not has_printed_waiting_message:
                log.info(
                    f"Max concurrent sessions reached ({max_concurrent_sessions}). "
                    f"Current sessions:\n{get_current_tmux_sessions()}\nWaiting for a free slot..."
                )
                has_printed_waiting_message = True
            time.sleep(5)

        log.info(f"Starting script {i}/{len(scripts)}")
        session_started = start_tmux_session(script, persistence_base_dir=persistence_base_dir, num_experiments=num_experiments)
        if session_started:
            time.sleep(2)  # Give tmux a moment to start the session

    has_printed_final_waiting_message = False
    # Wait for all sessions to complete
    while len(get_current_tmux_sessions()) > 0:
        if not has_printed_final_waiting_message:
            log.info(
                f"All scripts have now been started, waiting for completion of remaining tmux sessions:\n"
                f"{get_current_tmux_sessions()}"
            )
            has_printed_final_waiting_message = True
        time.sleep(10)
    log.info("All tmux sessions have completed.")
    # Aggregate results
    aggregate_results(str(persistence_base_dir))



if __name__ == "__main__":
    logging.run_cli(main)
