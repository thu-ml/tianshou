import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

from sensai.util import logging
from sensai.util.logging import datetime_tag

TMUX_SESSION_PREFIX = "tianshou_"

# Sleep durations in seconds
TMUX_SESSION_START_DELAY = 2
SESSION_CHECK_INTERVAL = 5
COMPLETION_CHECK_INTERVAL = 10

log = logging.getLogger("benchmark")

# Default tasks for each benchmark type
DEFAULT_TASKS = {
    "mujoco": [
        "Ant-v4",
        "HalfCheetah-v4",
        "Hopper-v4",
        "Humanoid-v4",
        "InvertedDoublePendulum-v4",
        "InvertedPendulum-v4",
        "Reacher-v4",
        "Swimmer-v4",
        "Walker2d-v4",
    ],
    "atari": [
        "PongNoFrameskip-v4",
        "BreakoutNoFrameskip-v4",
        "EnduroNoFrameskip-v4",
        "QbertNoFrameskip-v4",
        "MsPacmanNoFrameskip-v4",
        "SeaquestNoFrameskip-v4",
        "SpaceInvadersNoFrameskip-v4",
    ],
}


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


def get_current_tmux_sessions(benchmark_type: str) -> list[str]:
    """List active tmux sessions starting with TMUX_SESSION_PREFIX."""
    try:
        output = subprocess.check_output(["tmux", "list-sessions"], stderr=subprocess.DEVNULL)
        sessions = [
            line.split(b":")[0].decode()
            for line in output.splitlines()
            if line.startswith(f"{TMUX_SESSION_PREFIX}_{benchmark_type}".encode())
        ]
        return sessions
    except subprocess.CalledProcessError:
        return []


def start_tmux_session(
    script_path: str,
    persistence_base_dir: Path | str,
    num_experiments: int,
    benchmark_type: str,
    task: str,
    max_epochs: int | None = None,
    epoch_num_steps: int | None = None,
    experiment_launcher: Literal["sequential", "joblib"] | None = None,
) -> bool:
    """Start a tmux session running the given experiment script, returning True on success."""
    # Normalize paths for Git Bash / Windows compatibility
    python_exec = sys.executable.replace("\\", "/")
    script_path = script_path.replace("\\", "/")
    persistence_base_dir = str(persistence_base_dir).replace("\\", "/")

    # Include task name in session to avoid collisions when running multiple tasks
    script_name = Path(script_path).name.replace("_hl.py", "")
    # Remove benchmark_type from name since we add it explicitly below
    script_name = script_name.replace(benchmark_type, "").strip("_")

    session_name = f"{TMUX_SESSION_PREFIX}_{benchmark_type}_{task}_{script_name}"

    # Build command with optional max_epochs and epoch_num_steps
    cmd_args = f"{python_exec} {script_path} --num_experiments {num_experiments} --persistence_base_dir {persistence_base_dir} --task {task}"
    if max_epochs is not None:
        cmd_args += f" --max_epochs {max_epochs}"
    if epoch_num_steps is not None:
        cmd_args += f" --epoch_num_steps {epoch_num_steps}"
    if experiment_launcher is not None:
        cmd_args += f" --experiment_launcher {experiment_launcher}"

    cmd = [
        "tmux",
        "new-session",
        "-d",
        "-s",
        session_name,
        f"{cmd_args}; echo 'Finished {script_path}'; tmux kill-session -t {session_name}",
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


def aggregate_rliable_results(task_results_dir: str | Path) -> None:
    """Aggregate rliable results from all experiments into a single results.json per environment.

    This form is expected by `benchmark.js` in the docs.
    """
    task_results_dir = Path(task_results_dir)
    if not task_results_dir.exists():
        log.warning(f"Benchmark results directory does not exist: '{task_results_dir}'")
        return

    experiment_dirs = [d for d in task_results_dir.iterdir() if d.is_dir()]
    aggregated_results = []
    for experiment_dir in experiment_dirs:
        agent_name = experiment_dir.name.split("Experiment")[0]
        if not agent_name:
            log.warning(
                f"Could not extract agent name from directory: '{experiment_dir.name}', skipping..."
            )
            continue

        rliable_file = experiment_dir / "rliable_evaluation_test.json"
        if not rliable_file.exists():
            log.warning(f"Missing rliable results file: '{rliable_file}', skipping...")
            continue

        try:
            with open(rliable_file) as f:
                result_entries = json.load(f)
            for result_entry in result_entries:
                result_entry["agent"] = agent_name
                aggregated_results.append(result_entry)
        except (OSError, json.JSONDecodeError) as e:
            log.error(f"Failed to read or parse '{rliable_file}': {e}")
            continue

    if not aggregated_results:
        log.warning(f"No results to aggregate for directory '{task_results_dir}'")
        return

    aggregated_results_path = task_results_dir / "results.json"
    try:
        with open(aggregated_results_path, "w") as f:
            json.dump(aggregated_results, f, indent=4)
        log.info(f"Aggregated {len(aggregated_results)} results to '{aggregated_results_path}'.")
    except OSError as e:
        log.error(f"Failed to write aggregated results to '{aggregated_results_path}': {e}")


def main(
    max_concurrent_sessions: int | None = None,
    benchmark_type: Literal["mujoco", "atari"] = "mujoco",
    num_experiments: int = 10,
    max_scripts: int = -1,
    tasks: list[str] | None = None,
    max_tasks: int = -1,
    max_epochs: int | None = None,
    epoch_num_steps: int | None = None,
    experiment_launcher: Literal["sequential", "joblib"] | None = None,
) -> None:
    """
     Run the benchmarking by executing each high level script in its default configuration
     (apart from num_experiments, which defaults to 10) in its own tmux session.
     Note that if you have unclosed tmux sessions from previous runs, those will count
     towards the max_concurrent_sessions limit. You can terminate all sessions with
    `tmux kill-server`.

     :param max_concurrent_sessions: optionally restrict how many tmux sessions to open in parallel, each script will
         run in a tmux session
     :param benchmark_type: mujoco or atari
     :param num_experiments: number of experiments to run per script
     :param max_scripts: maximum number of scripts to run, -1 for all. Set this to a low number for testing.
     :param tasks: optional list of task names to run benchmarks on. If None, uses default tasks for the benchmark_type.
     :param max_tasks: maximum number of tasks to run, -1 for all. Set this to a low number for testing.
     :param max_epochs: optional maximum number of training epochs to pass to all scripts. If None, uses script defaults.
     :param epoch_num_steps: optional number of environment steps per epoch to pass to all scripts. If None, uses script defaults.
     :param experiment_launcher: type of experiment launcher to use, only has an effect if `num_experiments>1`.
        By default, will use the experiment launchers defined in the individual scripts.
     :return:
    """
    # Use default tasks if none provided
    if tasks is None:
        tasks = DEFAULT_TASKS.get(benchmark_type, [])
        if not tasks:
            raise ValueError(
                f"No default tasks found for benchmark_type '{benchmark_type}'. Please provide tasks manually."
            )

    # Limit number of tasks if specified
    if max_tasks > 0:
        log.info(f"Limiting to first {max_tasks}/{len(tasks)} tasks.")
        tasks = tasks[:max_tasks]

    log.info(f"Running benchmarks for {len(tasks)} task(s): {tasks}")

    persistence_base_dir = Path(__file__).parent / "logs" / benchmark_type / datetime_tag()

    # file logger for the global benchmarking logs, each individual experiment will log to its own file
    log_file = persistence_base_dir / "benchmarking_run.txt"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.add_file_logger(log_file, append=False)

    scripts = find_script_paths(benchmark_type)
    if max_scripts > 0:
        log.info(f"Limiting to first {max_scripts}/{len(scripts)} scripts.")
        scripts = scripts[:max_scripts]
    if max_concurrent_sessions is None:
        max_concurrent_sessions = len(scripts)

    # Run benchmarks for each task
    for i_task, task in enumerate(tasks, 1):
        log.info(
            f"=== Starting benchmark batch for '{benchmark_type}' on task '{task}' ({i_task}/{len(tasks)}) "
            f"for {len(scripts)} scripts with {max_concurrent_sessions} concurrent jobs ==="
        )
        for i_script, script in enumerate(scripts, start=1):
            # Wait for free slot
            has_printed_waiting_message = False
            while len(get_current_tmux_sessions(benchmark_type)) >= max_concurrent_sessions:
                if not has_printed_waiting_message:
                    log.info(
                        f"Max concurrent sessions reached ({max_concurrent_sessions}). "
                        f"Current sessions:\n{get_current_tmux_sessions(benchmark_type)}\nWaiting for a free slot..."
                    )
                    has_printed_waiting_message = True
                time.sleep(SESSION_CHECK_INTERVAL)

            log.info(f"Starting script {i_script}/{len(scripts)} for task '{task}'")
            session_started = start_tmux_session(
                script,
                benchmark_type=benchmark_type,
                persistence_base_dir=persistence_base_dir,
                num_experiments=num_experiments,
                task=task,
                max_epochs=max_epochs,
                epoch_num_steps=epoch_num_steps,
                experiment_launcher=experiment_launcher,
            )
            if session_started:
                time.sleep(TMUX_SESSION_START_DELAY)  # Give tmux a moment to start the session

        has_printed_final_waiting_message = False
        # Wait for all sessions to complete before moving to next task
        while len(get_current_tmux_sessions(benchmark_type)) > 0:
            if not has_printed_final_waiting_message:
                log.info(
                    f"All scripts for task '{task}' have been started, waiting for completion of remaining tmux sessions:\n"
                    f"{get_current_tmux_sessions(benchmark_type)}"
                )
                has_printed_final_waiting_message = True
            time.sleep(COMPLETION_CHECK_INTERVAL)
        log.info(f"All tmux sessions for task '{task}' have completed.")
        # Aggregate results for this specific task (scripts create task-named directory automatically)
        task_results_dir = persistence_base_dir / task
        log.info(f"Aggregating results for task '{task}' from directory: {task_results_dir}")
        try:
            aggregate_rliable_results(str(task_results_dir))
        except Exception as e:
            log.error(f"Failed to aggregate rliable results for task '{task}': {e}\nContinuing...")

    log.info(
        f"=== Benchmark batch completed for all {len(scripts)} scripts and all {len(tasks)} task(s) ==="
    )


if __name__ == "__main__":
    logging.run_cli(main)
