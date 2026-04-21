import hashlib
import json
import os
import sys
import subprocess
import tempfile
import threading
import tkinter as tk
import urllib.request
import zipfile
from tkinter import messagebox, ttk

HOOMINER_VERSION = "0.4.1"
DOWNLOAD_URL = f"https://github.com/Hoosat-Oy/hoominer/releases/download/{HOOMINER_VERSION}/hoominer-{HOOMINER_VERSION}-windows.zip"


miner_process = None


def _get_cache_dir() -> str:
    if getattr(sys, 'frozen', False):
        # Running as executable
        exe_dir = os.path.dirname(sys.executable)
    else:
        # Running as script
        exe_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(exe_dir, f"hoominer-{HOOMINER_VERSION}")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _sha1_file(file_path: str) -> str:
    digest = hashlib.sha1()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _maybe_verify_zip(zip_path: str) -> None:
    """Verify zip SHA1 if a matching .sha1sum file is present next to this script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sha1_path = os.path.join(script_dir, f"hoominer-{HOOMINER_VERSION}-windows.zip.sha1sum")
    if not os.path.exists(sha1_path):
        return

    expected = None
    with open(sha1_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Typical format: <sha1>  <filename>
            expected = line.split()[0]
            break

    if not expected:
        return

    actual = _sha1_file(zip_path)
    if actual.lower() != expected.lower():
        raise RuntimeError(
            "Downloaded zip failed SHA1 verification. "
            "Delete the cached zip and try again."
        )


def _download_if_needed(download_url: str, zip_path: str) -> None:
    if os.path.exists(zip_path) and os.path.getsize(zip_path) > 0:
        return
    urllib.request.urlretrieve(download_url, zip_path)
    _maybe_verify_zip(zip_path)


def _find_exe(search_root: str, exe_name: str) -> str:
    for root_dir, _, files in os.walk(search_root):
        for name in files:
            if name.lower() == exe_name.lower():
                return os.path.join(root_dir, name)
    raise FileNotFoundError(f"{exe_name} not found under: {search_root}")


def _extract_if_needed(zip_path: str, extract_dir: str) -> None:
    marker = os.path.join(extract_dir, ".extracted")
    if os.path.exists(marker):
        return
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    with open(marker, "w", encoding="utf-8") as f:
        f.write("ok")


def _build_command(
    miner_exe: str,
    stratum: str,
    wallet: str,
    device_mode: str,
    disable_cuda: bool,
    algorithm: str,
) -> list[str]:
    cmd = [
        miner_exe,
        "--algorithm",
        algorithm,
        "--stratum",
        stratum,
        "--user",
        wallet,
        "--password",
        "x",
    ]

    # Device selection
    # - CPU: disable all GPU backends
    # - GPU: disable CPU
    if device_mode == "cpu":
        cmd.extend(["--disable-gpu", "--disable-opencl", "--disable-cuda"])
    elif device_mode == "gpu":
        cmd.append("--disable-cpu")

    # Optional GPU backend toggle
    if disable_cuda and device_mode != "cpu":
        cmd.append("--disable-cuda")

    return cmd


def _get_settings_file() -> str:
    """Get the path to the settings file."""
    if getattr(sys, 'frozen', False):
        # Running as executable
        exe_dir = os.path.dirname(sys.executable)
    else:
        # Running as script
        exe_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(exe_dir, "hoosat-miner-settings.json")


def _save_settings() -> None:
    """Save current settings to file."""
    settings = {
        "stratum": stratum_entry.get().strip(),
        "wallet": wallet_entry.get().strip(),
        "algorithm": algorithm_var.get(),
        "device_mode": device_mode_var.get(),
        "disable_cuda": bool(disable_cuda_var.get()),
    }
    try:
        with open(_get_settings_file(), "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
    except Exception:
        # Silently ignore save errors
        pass


def _load_settings() -> None:
    """Load settings from file and apply them."""
    settings_file = _get_settings_file()
    if not os.path.exists(settings_file):
        return

    try:
        with open(settings_file, "r", encoding="utf-8") as f:
            settings = json.load(f)

        # Apply loaded settings
        if "stratum" in settings and settings["stratum"]:
            stratum_entry.delete(0, tk.END)
            stratum_entry.insert(0, settings["stratum"])

        if "wallet" in settings and settings["wallet"]:
            wallet_entry.delete(0, tk.END)
            wallet_entry.insert(0, settings["wallet"])

        if "algorithm" in settings:
            algorithm_var.set(settings["algorithm"])

        if "device_mode" in settings:
            device_mode_var.set(settings["device_mode"])

        if "disable_cuda" in settings:
            disable_cuda_var.set(settings["disable_cuda"])

    except Exception:
        # Silently ignore load errors
        pass


def _set_busy(is_busy: bool, status_text: str | None = None) -> None:
    state = (tk.DISABLED if is_busy else tk.NORMAL)
    start_button.config(state=state)
    stratum_entry.config(state=state)
    wallet_entry.config(state=state)
    for rb in algorithm_radiobuttons:
        rb.config(state=state)
    for rb in device_mode_radiobuttons:
        rb.config(state=state)
    disable_cuda_check.config(state=state)
    for rb in disable_mode_radiobuttons:
        rb.config(state=state)

    if is_busy:
        progress_bar.start(10)
    else:
        progress_bar.stop()
    if status_text is not None:
        status_var.set(status_text)


def _start_mining_worker(stratum: str, wallet: str, device_mode: str, disable_cuda: bool, algorithm: str) -> None:
    global miner_process
    try:
        cache_dir = _get_cache_dir()
        zip_path = os.path.join(cache_dir, f"hoominer-{HOOMINER_VERSION}-windows.zip")
        extract_dir = os.path.join(cache_dir, "extracted")

        _download_if_needed(DOWNLOAD_URL, zip_path)
        _extract_if_needed(zip_path, extract_dir)
        miner_exe = _find_exe(extract_dir, "hoominer.exe")

        cmd = _build_command(miner_exe, stratum, wallet, device_mode, disable_cuda, algorithm)

        creationflags = 0
        # if os.name == "nt" and hasattr(subprocess, "CREATE_NEW_CONSOLE"):
        #     creationflags = subprocess.CREATE_NEW_CONSOLE

        miner_process = subprocess.Popen(cmd, cwd=os.path.dirname(miner_exe), creationflags=creationflags)

        def on_success() -> None:
            status_var.set("Mining")
            start_button.config(text="Stop Mining", command=stop_mining, state=tk.NORMAL)
            stratum_entry.config(state=tk.DISABLED)
            wallet_entry.config(state=tk.DISABLED)
            for rb in algorithm_radiobuttons:
                rb.config(state=tk.DISABLED)
            for rb in device_mode_radiobuttons:
                rb.config(state=tk.DISABLED)
            disable_cuda_check.config(state=tk.DISABLED)
            for rb in disable_mode_radiobuttons:
                rb.config(state=tk.DISABLED)

        root.after(0, on_success)
    except Exception as e:
        def on_error() -> None:
            _set_busy(False, "Idle")
            messagebox.showerror("Error", f"Failed to download or run hoominer: {e}")

        root.after(0, on_error)


def stop_mining():
    global miner_process
    if miner_process and miner_process.poll() is None:
        miner_process.terminate()
        try:
            miner_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            miner_process.kill()
        miner_process = None
    start_button.config(text="Start Mining", command=start_mining)
    status_var.set("Stopped")
    stratum_entry.config(state=tk.NORMAL)
    wallet_entry.config(state=tk.NORMAL)
    for rb in algorithm_radiobuttons:
        rb.config(state=tk.NORMAL)
    for rb in device_mode_radiobuttons:
        rb.config(state=tk.NORMAL)
    disable_cuda_check.config(state=tk.NORMAL)
    for rb in disable_mode_radiobuttons:
        rb.config(state=tk.NORMAL)


def start_mining() -> None:
    global miner_process
    if miner_process and miner_process.poll() is None:
        messagebox.showinfo("Info", "Miner is already running.")
        return

    stratum = stratum_entry.get().strip()
    wallet = wallet_entry.get().strip()
    device_mode = device_mode_var.get()
    disable_cuda = bool(disable_cuda_var.get())
    algorithm = algorithm_var.get()

    # Save current settings
    _save_settings()

    if not stratum or not wallet:
        messagebox.showerror("Error", "Stratum address and wallet address are required.")
        return
    if "://" not in stratum or ":" not in stratum:
        if not messagebox.askyesno(
            "Confirm",
            "Stratum address doesn't look like a URL (e.g. stratum+tcp://pool:port).\n\nStart anyway?",
        ):
            return

    _set_busy(True, "Downloading / starting…")
    thread = threading.Thread(
        target=_start_mining_worker,
        args=(stratum, wallet, device_mode, disable_cuda, algorithm),
        daemon=True,
    )
    thread.start()

# GUI setup
root = tk.Tk()
root.title("Hoosat Miner Launcher")

root.minsize(520, 260)
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

style = ttk.Style()
try:
    style.theme_use("vista")
except Exception:
    pass

main = ttk.Frame(root, padding=14)
main.grid(row=0, column=0, sticky="nsew")
main.columnconfigure(0, weight=1)

connection_frame = ttk.LabelFrame(main, text="Connection", padding=10)
connection_frame.grid(row=0, column=0, sticky="ew")
connection_frame.columnconfigure(0, weight=1)

ttk.Label(connection_frame, text="Stratum address").grid(row=0, column=0, sticky="w")
stratum_entry = ttk.Entry(connection_frame)
stratum_entry.grid(row=1, column=0, sticky="ew", pady=(2, 8))

ttk.Label(connection_frame, text="Hoosat deposit address (wallet)").grid(row=2, column=0, sticky="w")
wallet_entry = ttk.Entry(connection_frame)
wallet_entry.grid(row=3, column=0, sticky="ew", pady=(2, 0))

options_frame = ttk.LabelFrame(main, text="Options", padding=10)
options_frame.grid(row=1, column=0, sticky="ew", pady=(12, 0))
options_frame.columnconfigure(0, weight=1)

ttk.Label(options_frame, text="Algorithm:").grid(row=0, column=0, sticky="w")
algorithm_var = tk.StringVar(value="hoohash")
algorithm_radiobuttons: list[tk.Radiobutton] = []
rb_alg_hoohash = tk.Radiobutton(options_frame, text="Hoohash", variable=algorithm_var, value="hoohash")
rb_alg_pepepow = tk.Radiobutton(options_frame, text="Pepepow", variable=algorithm_var, value="pepepow")
algorithm_radiobuttons.extend([rb_alg_hoohash, rb_alg_pepepow])
rb_alg_hoohash.grid(row=1, column=0, sticky="w")
rb_alg_pepepow.grid(row=2, column=0, sticky="w")

ttk.Separator(options_frame).grid(row=3, column=0, sticky="ew", pady=(8, 8))

ttk.Label(options_frame, text="Device:").grid(row=4, column=0, sticky="w")
device_mode_var = tk.StringVar(value="both")
device_mode_radiobuttons: list[tk.Radiobutton] = []
rb_dev_both = tk.Radiobutton(options_frame, text="CPU + GPU (default)", variable=device_mode_var, value="both")
rb_dev_cpu = tk.Radiobutton(options_frame, text="CPU only (--disable-gpu)", variable=device_mode_var, value="cpu")
rb_dev_gpu = tk.Radiobutton(options_frame, text="GPU only (--disable-cpu)", variable=device_mode_var, value="gpu")
device_mode_radiobuttons.extend([rb_dev_both, rb_dev_cpu, rb_dev_gpu])
rb_dev_both.grid(row=5, column=0, sticky="w")
rb_dev_cpu.grid(row=6, column=0, sticky="w")
rb_dev_gpu.grid(row=7, column=0, sticky="w")

ttk.Separator(options_frame).grid(row=8, column=0, sticky="ew", pady=(8, 8))

ttk.Label(options_frame, text="GPU options:").grid(row=9, column=0, sticky="w")
disable_cuda_var = tk.BooleanVar(value=False)
disable_cuda_check = ttk.Checkbutton(options_frame, text="Disable CUDA (--disable-cuda)", variable=disable_cuda_var)
disable_cuda_check.grid(row=10, column=0, sticky="w")

disable_mode_radiobuttons: list[tk.Radiobutton] = []

footer = ttk.Frame(main)
footer.grid(row=2, column=0, sticky="ew", pady=(12, 0))
footer.columnconfigure(0, weight=1)

status_var = tk.StringVar(value="Idle")
ttk.Label(footer, textvariable=status_var).grid(row=0, column=0, sticky="w")
progress_bar = ttk.Progressbar(footer, mode="indeterminate", length=140)
progress_bar.grid(row=0, column=1, sticky="e", padx=(10, 10))

start_button = ttk.Button(footer, text="Start Mining", command=start_mining)
start_button.grid(row=0, column=2, sticky="e")

def _on_enter(_: object) -> None:
    start_mining()

root.bind("<Return>", _on_enter)
stratum_entry.focus_set()

# Load saved settings
_load_settings()


def _on_algorithm_change(*args) -> None:
    """Automatically disable CUDA for pepepow algorithm since it only supports OpenCL and CPU."""
    if algorithm_var.get() == "pepepow":
        disable_cuda_var.set(True)
    else:
        disable_cuda_var.set(False)


# Add trace after variables are defined
algorithm_var.trace_add("write", _on_algorithm_change)
_on_algorithm_change()  # Set initial state

root.mainloop()