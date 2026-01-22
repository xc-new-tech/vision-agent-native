#!/usr/bin/env python3
"""
Cross-platform script to run the FastAPI backend and React frontend.
Works on Windows, macOS, and Linux without any additional dependencies.
"""

import os
import sys
import time
import signal
import webbrowser
import subprocess
import platform
import shutil
from pathlib import Path
from dotenv import load_dotenv, dotenv_values

load_dotenv()

# Configuration
PORT_BACKEND = os.getenv("PORT_BACKEND", "8888")
PORT_FRONTEND = os.getenv("PORT_FRONTEND", "3000")
BACKEND_HOST = os.getenv("BACKEND_HOST", "localhost")
PROCESSES = []


def is_windows():
    """Check if the current platform is Windows"""
    return platform.system() == "Windows"


def find_process_by_port(port):
    """Find process using a specific port"""
    if is_windows():
        try:
            # Windows - use netstat
            output = subprocess.check_output(
                f"netstat -ano | findstr :{port}", shell=True
            ).decode()
            if output:
                for line in output.splitlines():
                    if f":{port}" in line and "LISTENING" in line:
                        parts = line.strip().split()
                        return int(parts[-1])
        except subprocess.CalledProcessError:
            pass
    else:
        try:
            # Unix-like - use lsof
            output = (
                subprocess.check_output(f"lsof -i :{port} -t", shell=True)
                .decode()
                .strip()
            )
            if output:
                return int(output)
        except subprocess.CalledProcessError:
            pass
    return None


def kill_process(pid):
    """Kill a process by its PID"""
    if pid:
        try:
            if is_windows():
                subprocess.run(f"taskkill /F /PID {pid}", shell=True, check=False)
            else:
                os.kill(pid, signal.SIGKILL)
            print(f"[INFO] Killed process {pid}")
            return True
        except (subprocess.SubprocessError, OSError) as e:
            print(f"[WARNING] Failed to kill process {pid}: {e}")
    return False


def cleanup_port(port):
    """Clean up a specific port by killing any process using it"""
    pid = find_process_by_port(port)
    if pid:
        print(f"[INFO] Port {port} is in use by process {pid}, attempting to free...")
        kill_process(pid)
        # Wait briefly to ensure the port is released
        time.sleep(1)
        return True
    return False


def cleanup():
    """Clean up all running processes and ports"""
    print("\n[INFO] Shutting down all processes...")

    # First try to terminate processes gracefully
    for proc in PROCESSES:
        if proc and proc.poll() is None:  # Check if process is still running
            try:
                if is_windows():
                    proc.terminate()
                else:
                    proc.send_signal(signal.SIGTERM)
                print(f"[INFO] Terminating process: {proc.pid}")
            except Exception as e:
                print(f"[WARNING] Error terminating process {proc.pid}: {e}")

    # Give processes time to shut down gracefully
    time.sleep(2)

    # Force kill any remaining processes
    for proc in PROCESSES:
        if proc and proc.poll() is None:  # Check if process is still running
            try:
                proc.kill()
                print(f"[INFO] Force killed process: {proc.pid}")
            except Exception as e:
                print(f"[WARNING] Error killing process {proc.pid}: {e}")

    # Clean up ports as a last resort
    cleanup_port(PORT_BACKEND)
    cleanup_port(PORT_FRONTEND)

    print("[INFO] All processes terminated, ports released.")


def signal_handler(sig, frame):
    """Handle interrupt signals (Ctrl+C)"""
    print("\n[INFO] Received interrupt signal. Shutting down...")
    cleanup()
    sys.exit(0)


def run_command(cmd, cwd=None, shell=False, env=None):
    """Run a command in a subprocess with appropriate platform considerations"""
    if is_windows() and not shell:
        if isinstance(cmd, str):
            return subprocess.Popen(cmd, cwd=cwd, shell=True, env=env)
        return subprocess.Popen(cmd, cwd=cwd, env=env)
    return subprocess.Popen(cmd, cwd=cwd, shell=shell, env=env)


def main():
    """Main function to start services"""
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Set environment variable
    os.environ["REPORT_TOOL_TRACES"] = "1"

    print("[INFO] Starting services. Press Ctrl+C to stop all processes.")

    # Clean up ports if they're already in use
    cleanup_port(PORT_BACKEND)
    cleanup_port(PORT_FRONTEND)

    try:
        # Start FastAPI backend
        backend_env = os.environ.copy()
        backend_env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
        repo_env = dotenv_values(str(Path(__file__).resolve().parents[2] / ".env"))
        backend_env.update({k: v for k, v in repo_env.items() if v})

        if is_windows():
            backend_proc = run_command(
                f"python -m fastapi dev app.py --port {PORT_BACKEND}",
                env=backend_env,
            )
        else:
            backend_proc = run_command(
                f"fastapi dev app.py --port {PORT_BACKEND}",
                shell=True,
                env=backend_env,
            )

        PROCESSES.append(backend_proc)
        print(f"[INFO] Started FastAPI backend with process ID: {backend_proc.pid}")

        # Start React frontend
        chat_app_dir = Path("chat-app").absolute()
        if not chat_app_dir.exists():
            print(f"[ERROR] Directory not found: {chat_app_dir}")
            cleanup()
            sys.exit(1)

        frontend_env = os.environ.copy()
        frontend_env["PORT"] = str(PORT_FRONTEND)
        frontend_env["NEXT_PUBLIC_PORT_BACKEND"] = str(PORT_BACKEND)
        frontend_env["NEXT_PUBLIC_BACKEND_HOST"] = BACKEND_HOST

        npm_path = shutil.which("npm")
        if not npm_path:
            print("[ERROR] 未找到 npm，请先安装 Node.js。")
            cleanup()
            sys.exit(1)

        if is_windows():
            frontend_proc = run_command(
                f"{npm_path} run dev",
                cwd=str(chat_app_dir),
                shell=True,
                env=frontend_env,
            )
        else:
            frontend_proc = subprocess.Popen(
                [npm_path, "run", "dev"],
                cwd=str(chat_app_dir),
                env=frontend_env,
                shell=False,
            )

        PROCESSES.append(frontend_proc)
        print(f"[INFO] Started React frontend with process ID: {frontend_proc.pid}")

        # Give services a moment to start
        time.sleep(2)

        # Open browser
        print("[INFO] Opening browser to http://localhost:3000")
        try:
            webbrowser.open(f"http://localhost:{PORT_FRONTEND}")
        except Exception as e:
            print(f"[WARNING] Could not open browser: {e}")
            print(f"[INFO] Please manually open: http://localhost:{PORT_FRONTEND}")

        # Keep the script running until Ctrl+C
        print("[INFO] Services are running. Press Ctrl+C to stop.")
        while all(proc.poll() is None for proc in PROCESSES if proc):
            time.sleep(1)

        # If we get here, one of the processes has ended
        for proc in PROCESSES:
            if proc and proc.poll() is not None:
                print(
                    f"[WARNING] Process {proc.pid} exited with code {proc.returncode}"
                )

        # Clean up remaining processes
        cleanup()

    except Exception as e:
        print(f"[ERROR] {e}")
        cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()
