#!/usr/bin/env python3
"""run-all.py — end‑to‑end *setup + benchmark* runner (no plotting)

Use `--help` to see all options.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Sequence

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _header(msg: str) -> None:
    print("\n" + "=" * 80)
    print(msg)
    print("=" * 80, flush=True)


def _run(cmd: Sequence[str] | str, *, env: Dict[str, str] | None = None, cwd: str | Path | None = None) -> None:
    disp = " ".join(cmd) if isinstance(cmd, (list, tuple)) else cmd
    print(f"+ {disp}")
    subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd, env=env, check=True)


def _sudo(cmd: Sequence[str] | str, **kw):
    full = ["sudo", "-E", *cmd] if isinstance(cmd, (list, tuple)) else f"sudo -E {cmd}"
    _run(full, **kw)


def _cargo_clean(repo_dir: Path) -> None:
    _run(["cargo", "clean"], cwd=repo_dir)


def _reset_dirs(*dirs: Path) -> None:
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Repository & dataset paths
# ---------------------------------------------------------------------------
# Determine if we are already inside a git repo named 'ebi'
def _is_inside_ebi_repo() -> bool:
    try:
        top_level = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL)
        return Path(top_level.decode().strip()).name == "ebi"
    except subprocess.CalledProcessError:
        return False

REPO_URL = "git@github.com:lemolatoon/ebi.git"
EBI_DIR = Path(".").resolve()
DATA_DIR = EBI_DIR / "data"
DATA_GENERAL = DATA_DIR / "general"
DATA_XOR = DATA_DIR / "xor_dataset"
DATA_EMB = DATA_DIR / "embeddings"
DATA_UCR = DATA_DIR / "UCRArchive_2018"
TPCH_DATA = EBI_DIR / "tools/tpch_data"

# ---------------------------------------------------------------------------
# Setup utilities
# ---------------------------------------------------------------------------

def _ensure_tool(cmd: str, installer: Sequence[str] | str) -> None:
    if shutil.which(cmd):
        return
    _header(f"[Setup] Installing {cmd}")
    _run(installer)
    if not shutil.which(cmd):
        sys.exit(f"❌ Failed to install {cmd} or not in PATH.")


def _extract_tar(tar_path: Path, dest_dir: Path) -> None:
    _header(f"[Setup] Extracting {tar_path} → {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    _run(["tar", "xvf", str(tar_path), "-C", str(dest_dir)])

# ---------------------------------------------------------------------------
# Environment / data setup
# ---------------------------------------------------------------------------

def setup_environment(data_tar: Path | None, ucr_tar: Path | None) -> None:
    _header("[Setup] Environment & data")

    if _is_inside_ebi_repo():
        EBI_DIR = Path(".").resolve()
        print("✅ Already inside ebi repo – using current directory")
    else:
        sys.exit("❌ Please run this script inside the ebi repo.")

    # 2) ensure tools
    _ensure_tool("cargo", "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")
    _ensure_tool("uv", "curl -LsSf https://astral.sh/uv/install.sh | sh")
    if shutil.which("clang") is None:
        _header("[Setup] Installing clang via apt")
        _sudo(["apt", "update"])
        _sudo(["apt", "install", "-y", "clang"])
    else:
        print("clang already present – skipping apt install")

    # 3) submodules
    _run(["git", "submodule", "init"], cwd=EBI_DIR)
    _run(["git", "submodule", "update"], cwd=EBI_DIR)

    # 4) Handle optional tar extractions
    if not DATA_GENERAL.exists():
        if data_tar is None:
            sys.exit(f"❌ {DATA_GENERAL} not found and --data-tar not provided.")
        if not data_tar.is_file():
            sys.exit(f"❌ --data-tar path {data_tar} does not exist.")
        _extract_tar(data_tar, DATA_DIR)
        # Some archives extract into data/data/…; if so, rename
        extracted = DATA_DIR / "data"
        if extracted.exists() and not DATA_GENERAL.exists():
            extracted.rename(DATA_GENERAL)
    else:
        print(f"{DATA_GENERAL} exists – skipping data_tar extraction")

    if not DATA_UCR.exists() and ucr_tar is not None:
        if not ucr_tar.is_file():
            sys.exit(f"❌ --ucrarchive2018-tar path {ucr_tar} does not exist.")
        _extract_tar(ucr_tar, DATA_DIR)
    elif DATA_UCR.exists():
        print(f"{DATA_UCR} exists – skipping UCR tar extraction")
    elif ucr_tar is None:
        sys.exit(f"❌ {DATA_UCR} not found and --ucrarchive2018-tar not provided.")

    # 5) csv2bin for general
    if not (DATA_GENERAL / "binary").exists():
        _header("[Setup] csv2bin → data/general")
        _run(["cargo", "run", "--bin", "csv2bin", "--release", "--", "data/general/*"], cwd=EBI_DIR)
    else:
        print("binary/ for general already present – skipping csv2bin")

    # 6) create embedding & xor datasets
    DATA_EMB.mkdir(parents=True, exist_ok=True)
    DATA_XOR.mkdir(parents=True, exist_ok=True)

    precision_json = DATA_EMB / "precision_data.json"
    bin_files = list(DATA_EMB.glob("*.bin"))

    if precision_json.exists() and bin_files:
        print("Embedding datasets already present – skipping creation")
    else:
        _run(["uv", "run", "create_embed_datasets.py", "../data/embeddings"], cwd=EBI_DIR / "tools")

    xor_csv_files = list(DATA_XOR.glob("xor_dataset_*.csv"))
    if xor_csv_files:
        print("XOR datasets already present – skipping creation")
    else:
        _run(["uv", "run", "create_xor_datasets.py", "../data/xor_dataset"], cwd=EBI_DIR / "tools")

    # 7) csv2bin for xor
    if not (DATA_XOR / "binary").exists():
        _header("[Setup] csv2bin → data/xor_dataset")
        _run(["cargo", "run", "--bin", "csv2bin", "--release", "--", "data/xor_dataset/*"], cwd=EBI_DIR)
    else:
        print("binary/ for xor_dataset already present – skipping csv2bin")

# ---------------------------------------------------------------------------
# Repo‑scoped helpers
# ---------------------------------------------------------------------------

def _repo_run(cmd: Sequence[str] | str, **kw):
    cwd = kw.pop("cwd", EBI_DIR)
    _run(cmd, cwd=cwd, **kw)


def _repo_sudo(cmd: Sequence[str] | str, **kw):
    cwd = kw.pop("cwd", EBI_DIR)
    _sudo(cmd, cwd=cwd, **kw)

# ---------------------------------------------------------------------------
# Benchmark functions (exact Bash parity)
# ---------------------------------------------------------------------------

def compression_and_queries_general(n: int) -> None:
    _header("[General] Compression/query (4 configs)")
    save_dir = EBI_DIR / "save"
    comp_cfg = EBI_DIR / "compressor_configs"
    filter_cfg = DATA_GENERAL / "filter_config"
    result_root = DATA_GENERAL / "result" / "all"
    comp_cfg.mkdir(exist_ok=True)
    filter_cfg.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    def _run_once(cpu: str, in_mem: bool):
        _header(f"  → cpu={cpu}  in_memory={in_mem}")
        _cargo_clean(EBI_DIR)
        env = os.environ.copy(); env.update({"RUSTFLAGS": f"-C target-cpu={cpu}", "TARGET_CPU": cpu})
        _repo_run(["cargo", "build", "--release", "--bin", "experimenter"], env=env)
        _repo_sudo(["target/release/experimenter",
                   "all", "-c", str(comp_cfg), "--exact-precision",
                   "-f", str(filter_cfg), "-b", str(DATA_GENERAL / "binary"),
                   "--create-config", "--n", str(n), *( ["--in-memory"] if in_mem else [] ),
                   "-s", str(save_dir)], env=env)

    config_sets = [("haswell", False), ("native", False), ("haswell", True), ("native", True)]
    for cpu, mem in config_sets:
        _reset_dirs(comp_cfg, filter_cfg, save_dir)
        _run_once(cpu, mem)

    print(f"[General] Results base directory: {result_root}")
    for i, (cpu, mem) in enumerate(config_sets):
        print(f"{i}  → cpu={cpu}  in_memory={mem}")


def xor_bench(n: int) -> None:
    _header("[XOR] Compression/query (Haswell, in-memory)")
    save_dir = EBI_DIR / "save"
    comp_cfg = EBI_DIR / "compressor_configs"
    _reset_dirs(save_dir, comp_cfg, DATA_XOR / "filter_config")
    _cargo_clean(EBI_DIR)
    env = os.environ.copy(); env.update({"RUSTFLAGS": "-C target-cpu=haswell", "TARGET_CPU": "haswell"})
    _repo_run(["cargo", "build", "--release", "--bin", "experimenter"], env=env)
    _repo_sudo(["target/release/experimenter", "all", "-c", "compressor_configs",
                "--exact-precision", "-f", str(DATA_XOR / "filter_config"),
                "-b", str(DATA_XOR / "binary"), "--create-config",
                "--in-memory", "--n", str(n), "-s", "save"], env=env)
    print(f"[XOR] Results base directory: {DATA_XOR / 'result' / 'all'}")


def embeddings_bench() -> None:
    _header("[Embeddings] Benchmark")
    _cargo_clean(EBI_DIR)
    (EBI_DIR / "embeddings").mkdir(exist_ok=True)
    env = os.environ.copy(); env.update({"RUSTFLAGS": "-C target-cpu=haswell", "TARGET_CPU": "haswell"})
    _repo_run(["cargo", "build", "--release", "--bin", "experimenter"], env=env)
    _repo_sudo(["target/release/experimenter", "-i", str(DATA_EMB), "embedding", "-o", "embeddings"], env=env)
    print(f"[Embeddings] Results base directory: {(EBI_DIR / 'embeddings' / 'result' / 'embedding')}" )


def ucr_bench() -> None:
    _header("[UCR] 1‑NN benchmark")
    _cargo_clean(EBI_DIR)
    (EBI_DIR / "ucr2018").mkdir(exist_ok=True)
    env = os.environ.copy(); env.update({"RUSTFLAGS": "-C target-cpu=haswell", "TARGET_CPU": "haswell"})
    _repo_run(["cargo", "build", "--release", "--bin", "experimenter"], env=env)
    _repo_sudo(["target/release/experimenter", "-i", str(DATA_UCR), "ucr2018", "-o", "ucr2018"], env=env)
    print(f"[UCR] Results base directory: {DATA_DIR / 'result' / 'ucr2018'}")


def tpch_bench() -> None:
    _header("[TPC-H] Benchmark")
    _repo_run(["uv", "run", "gen-tpch.py"], cwd=EBI_DIR / "tools")
    _repo_run(["uv", "run", "gen-tpch.py", "-r"], cwd=EBI_DIR / "tools")
    _cargo_clean(EBI_DIR)
    env = os.environ.copy(); env.update({"RUSTFLAGS": "-C target-cpu=haswell", "TARGET_CPU": "haswell"})
    _repo_run(["cargo", "build", "--release", "--bin", "experimenter"], env=env)
    (EBI_DIR / "tpch").mkdir(exist_ok=True)
    _repo_sudo(["target/release/experimenter", "tpch", "-i", str(TPCH_DATA), "-o", "tpch"], env=env)
    print(f"[TPC‑H] Results base directory: {(EBI_DIR / 'tpch' / 'result' / 'tpch')}")


def matmul_cuda() -> None:
    _header("[MatMul] CUDA benchmark")
    _cargo_clean(EBI_DIR)
    env = os.environ.copy(); env.update({"RUSTFLAGS": "-C target-cpu=native", "TARGET_CPU": "native"})
    _repo_run(["cargo", "build", "--release", "--features=cuda"], env=env, cwd=EBI_DIR / "experimenter")
    _repo_sudo(["../target/release/experimenter", "matrix-cuda", "-o", "matrix_cuda"], env=env, cwd=EBI_DIR / "experimenter")
    print(f"[MatMul] Results base directory: {(EBI_DIR/'experimenter'/'matrix_cuda'/'result'/'matrix_cuda')}")

# ---------------------------------------------------------------------------
# Repo‑scoped helpers & benchmarks (unchanged logic)
# ---------------------------------------------------------------------------
# … (KEEP ALL EXISTING BENCHMARK FUNCTIONS AS‑IS) …

# For brevity, benchmark functions from previous version are unchanged and
# assumed to be present below this comment.
# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------

def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EBI setup + benchmark runner (no plotting)")
    p.add_argument("--n", type=int, default=5, help="Repetitions for compression benchmarks (default: 5)")
    p.add_argument("--only-setup", action="store_true", help="Run environment/data setup only, no benchmarks")
    p.add_argument("--only-matmul", action="store_true", help="Run *only* the CUDA matmul benchmark and exit")
    p.add_argument("--skip-matmul", action="store_true", help="Skip CUDA matmul stage in full pipeline")
    p.add_argument("--data-tar", type=Path, help="Tarball containing the general dataset")
    p.add_argument("--ucrarchive2018-tar", type=Path, help="Tarball containing the UCRArchive_2018 dataset")
    return p.parse_args()


def main() -> None:
    args = parse_cli()

    # 1) Setup
    setup_environment(args.data_tar, args.ucrarchive2018_tar)

    if args.only_setup:
        _header("✔︎ Environment setup complete (only-setup mode)")
        return

    if args.only_matmul:
        matmul_cuda()
        _header("✔︎ MatMul benchmark complete (only-matmul mode)")
        return

    # 2) Benchmarks
    compression_and_queries_general(args.n)
    xor_bench(args.n)
    embeddings_bench()
    ucr_bench()
    tpch_bench()
    if not args.skip_matmul:
        matmul_cuda()

    _header("✔︎ All requested benchmarks completed")


if __name__ == "__main__":
    main()

