#!/usr/bin/env python3

import os
import subprocess
import sys
import shutil
import time
import re
import argparse
import matplotlib.pyplot as plt
from statistics import mean

ENZYME_PATH = "/home/brant/sync/Enzyme/build/Enzyme/ClangEnzyme-15.so"
LLVM_PATH = "/home/brant/llvms/llvm15/build/bin"
CXX = os.path.join(LLVM_PATH, "clang++")

CXXFLAGS = [
    "-O3",
    "-I/home/brant/include",
    "-L/home/brant/lib",
    "-I/usr/include/c++/11",
    "-I/usr/include/x86_64-linux-gnu/c++/11",
    "-L/usr/lib/gcc/x86_64-linux-gnu/11",
    "-fno-exceptions",
    f"-fpass-plugin={ENZYME_PATH}",
    "-Xclang",
    "-load",
    "-Xclang",
    ENZYME_PATH,
    "-lmpfr",
    "-ffast-math",
    "-fuse-ld=lld",
]

FPOPTFLAGS_BASE = [
    "-mllvm",
    "--enzyme-enable-fpopt",
    "-mllvm",
    "--enzyme-print-herbie",
    "-mllvm",
    "--enzyme-print-fpopt",
    "-mllvm",
    "--fpopt-log-path=example.txt",
    "-mllvm",
    "--fpopt-target-func-regex=example",
    "-mllvm",
    "--fpopt-enable-solver",
    "-mllvm",
    "--fpopt-enable-pt",
    "-mllvm",
    "--fpopt-comp-cost-budget=0",
    "-mllvm",
    "--fpopt-num-samples=1000",
    "-mllvm",
    "--fpopt-cost-model-path=../microbm/cm.csv",
]

SRC = "example.c"
LOGGER = "fp-logger.cpp"
EXE = ["example.exe", "example-logged.exe", "example-fpopt.exe"]
NUM_RUNS = 10
DRIVER_NUM_SAMPLES = 10000000


def run_command(command, description, capture_output=False, output_file=None, verbose=True):
    print(f"=== {description} ===")
    print("Running:", " ".join(command))
    try:
        if capture_output and output_file:
            with open(output_file, "w") as f:
                subprocess.check_call(command, stdout=f, stderr=subprocess.STDOUT)
        elif capture_output:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            return result.stdout
        else:
            if verbose:
                subprocess.check_call(command)
            else:
                subprocess.check_call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"Error during: {description}")
        if capture_output and output_file:
            print(f"Check the output file: {output_file} for details.")
        else:
            print(e)
        sys.exit(e.returncode)


def clean(tmp_dir, logs_dir, prefix):
    print("=== Cleaning up generated files ===")
    exe_files = [
        os.path.join(tmp_dir, f"{prefix}example.exe"),
        os.path.join(tmp_dir, f"{prefix}example-logged.exe"),
        os.path.join(tmp_dir, f"{prefix}example-baseline.exe"),
        os.path.join(tmp_dir, f"{prefix}example-fpopt.exe"),
    ]
    cpp_files = [
        os.path.join(tmp_dir, f"{prefix}example.cpp"),
        os.path.join(tmp_dir, f"{prefix}example-logged.cpp"),
        os.path.join(tmp_dir, f"{prefix}example-baseline.cpp"),
    ]
    log_files = [
        os.path.join(logs_dir, f"{prefix}compile_fpopt.log"),
    ]
    output_txt = os.path.join(tmp_dir, f"{prefix}example.txt")
    runtime_plot = os.path.join("plots", f"runtime_plot_{prefix[:-1]}.png")

    for exe in exe_files:
        if os.path.exists(exe):
            os.remove(exe)
            print(f"Removed {exe}")
    for cpp in cpp_files:
        if os.path.exists(cpp):
            os.remove(cpp)
            print(f"Removed {cpp}")
    for log in log_files:
        if os.path.exists(log):
            os.remove(log)
            print(f"Removed {log}")
    if os.path.exists(output_txt):
        os.remove(output_txt)
        print(f"Removed {output_txt}")
    if os.path.exists(runtime_plot):
        os.remove(runtime_plot)
        print(f"Removed {runtime_plot}")


def generate_example_cpp(tmp_dir, prefix):
    script = "fpopt-original-driver-generator.py"
    print(f"=== Running {script} ===")
    src_prefixed = os.path.join(tmp_dir, f"{prefix}{SRC}")
    dest_prefixed = os.path.join(tmp_dir, f"{prefix}example.cpp")
    run_command(
        ["python3", script, src_prefixed, dest_prefixed, "example", str(DRIVER_NUM_SAMPLES)],
        f"Generating example.cpp from {SRC}",
    )
    if not os.path.exists(dest_prefixed):
        print(f"Failed to generate {dest_prefixed}.")
        sys.exit(1)
    print(f"Generated {dest_prefixed} successfully.")


def generate_example_logged_cpp(tmp_dir, prefix):
    script = "fpopt-logged-driver-generator.py"
    print(f"=== Running {script} ===")
    src_prefixed = os.path.join(tmp_dir, f"{prefix}{SRC}")
    dest_prefixed = os.path.join(tmp_dir, f"{prefix}example-logged.cpp")
    run_command(
        ["python3", script, src_prefixed, dest_prefixed, "example", str(DRIVER_NUM_SAMPLES)],
        f"Generating example-logged.cpp from {SRC}",
    )
    if not os.path.exists(dest_prefixed):
        print(f"Failed to generate {dest_prefixed}.")
        sys.exit(1)
    print(f"Generated {dest_prefixed} successfully.")


def generate_example_baseline_cpp(tmp_dir, prefix):
    script = "fpopt-baseline-generator.py"
    print(f"=== Running {script} ===")
    src_prefixed = os.path.join(tmp_dir, f"{prefix}{SRC}")
    dest_prefixed = os.path.join(tmp_dir, f"{prefix}example-baseline.cpp")
    run_command(
        ["python3", script, src_prefixed, dest_prefixed, "example", str(DRIVER_NUM_SAMPLES)],
        f"Generating example-baseline.cpp from {SRC}",
    )
    if not os.path.exists(dest_prefixed):
        print(f"Failed to generate {dest_prefixed}.")
        sys.exit(1)
    print(f"Generated {dest_prefixed} successfully.")


def compile_example_exe(tmp_dir, prefix):
    source = os.path.join(tmp_dir, f"{prefix}example.cpp")
    output = os.path.join(tmp_dir, f"{prefix}example.exe")
    cmd = [CXX, source] + CXXFLAGS + ["-o", output]
    run_command(cmd, f"Compiling {output}")


def compile_example_logged_exe(tmp_dir, prefix):
    source = os.path.join(tmp_dir, f"{prefix}example-logged.cpp")
    output = os.path.join(tmp_dir, f"{prefix}example-logged.exe")
    cmd = [CXX, source, LOGGER] + CXXFLAGS + ["-o", output]
    run_command(cmd, f"Compiling {output}")


def compile_example_baseline_exe(tmp_dir, prefix):
    source = os.path.join(tmp_dir, f"{prefix}example-baseline.cpp")
    output = os.path.join(tmp_dir, f"{prefix}example-baseline.exe")
    cmd = [CXX, source] + CXXFLAGS + ["-o", output]
    run_command(cmd, f"Compiling {output}")


def generate_example_txt(tmp_dir, prefix):
    exe = os.path.join(tmp_dir, f"{prefix}example-logged.exe")
    output = os.path.join(tmp_dir, f"{prefix}example.txt")
    if not os.path.exists(exe):
        print(f"Executable {exe} not found. Cannot generate {output}.")
        sys.exit(1)
    with open(output, "w") as f:
        print(f"=== Running {exe} to generate {output} ===")
        try:
            subprocess.check_call([exe], stdout=f)
        except subprocess.CalledProcessError as e:
            print(f"Error running {exe}")
            sys.exit(e.returncode)


def compile_example_fpopt_exe(tmp_dir, prefix, fpoptflags, output="example-fpopt.exe", verbose=True):
    source = os.path.join(tmp_dir, f"{prefix}example.cpp")
    output_path = os.path.join(tmp_dir, f"{prefix}{output}")
    cmd = [CXX, source] + CXXFLAGS + fpoptflags + ["-o", output_path]
    log_path = os.path.join("logs", f"{prefix}compile_fpopt.log")
    if output == "example-fpopt.exe":
        run_command(
            cmd,
            f"Compiling {output_path} with FPOPTFLAGS",
            capture_output=True,
            output_file=log_path,
            verbose=verbose,
        )
    else:
        run_command(
            cmd,
            f"Compiling {output_path} with FPOPTFLAGS",
            verbose=verbose,
        )


def parse_critical_comp_costs(tmp_dir, prefix, log_path="compile_fpopt.log"):
    print(f"=== Parsing critical computation costs from {log_path} ===")
    full_log_path = os.path.join("logs", f"{prefix}{log_path}")
    if not os.path.exists(full_log_path):
        print(f"Log file {full_log_path} does not exist.")
        sys.exit(1)
    with open(full_log_path, "r") as f:
        content = f.read()

    pattern = r"\*\*\* Critical Computation Costs \*\*\*(.*?)\*\*\* End of Critical Computation Costs \*\*\*"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        print("Critical Computation Costs block not found in the log.")
        sys.exit(1)

    costs_str = match.group(1).strip()
    costs = [int(cost) for cost in costs_str.split(",") if re.fullmatch(r"-?\d+", cost.strip())]
    print(f"Parsed computation costs: {costs}")
    return costs


def measure_runtime(tmp_dir, prefix, executable, num_runs=NUM_RUNS):
    print(f"=== Measuring runtime for {executable} ===")
    runtimes = []
    exe_path = os.path.join(tmp_dir, f"{prefix}{executable}")
    for i in range(1, num_runs + 1):
        try:
            result = subprocess.run([exe_path], capture_output=True, text=True, check=True)
            output = result.stdout
            match = re.search(r"Total runtime: ([\d\.]+) seconds", output)
            if match:
                runtime = float(match.group(1))
                runtimes.append(runtime)
                print(f"Run {i}: {runtime:.6f} seconds")
            else:
                print(f"Could not parse runtime from output on run {i}")
                sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"Error running {exe_path} on run {i}")
            sys.exit(e.returncode)
    average_runtime = mean(runtimes)
    print(f"Average runtime for {executable}: {average_runtime:.6f} seconds")
    return average_runtime


def plot_results(plots_dir, prefix, budgets, runtimes, example_adjusted_runtime=None):
    plot_filename = os.path.join(plots_dir, f"runtime_plot_{prefix[:-1]}.png")
    print(f"=== Plotting results to {plot_filename} ===")
    plt.figure(figsize=(10, 6))
    plt.plot(budgets, runtimes, marker="o", linestyle="-", label="FPOPT Adjusted Runtime")

    if example_adjusted_runtime is not None:
        plt.axhline(y=example_adjusted_runtime, color="r", linestyle="--", label="example.exe Adjusted Runtime")

    plt.xlabel("Computation Cost Budget")
    plt.ylabel("Average Runtime (seconds)")
    plt.title("Computation Cost Budget vs. Denoised Average Runtime")
    plt.grid(True)
    plt.legend()
    plt.savefig(plot_filename)
    plt.close()
    print(f"Plot saved to {plot_filename}")


def build_all(tmp_dir, logs_dir, prefix):
    generate_example_cpp(tmp_dir, prefix)
    generate_example_logged_cpp(tmp_dir, prefix)
    generate_example_baseline_cpp(tmp_dir, prefix)
    compile_example_exe(tmp_dir, prefix)
    compile_example_logged_exe(tmp_dir, prefix)
    compile_example_baseline_exe(tmp_dir, prefix)
    generate_example_txt(tmp_dir, prefix)
    fpoptflags = []
    for flag in FPOPTFLAGS_BASE:
        if flag.startswith("--fpopt-log-path="):
            fpoptflags.append(f"--fpopt-log-path=tmp/{prefix}example.txt")
        else:
            fpoptflags.append(flag)
    compile_example_fpopt_exe(tmp_dir, prefix, fpoptflags, output="example-fpopt.exe")
    print("=== Initial build process completed successfully ===")


def measure_baseline_runtime(tmp_dir, prefix, num_runs=NUM_RUNS):
    executable = f"example-baseline.exe"
    exe_path = os.path.join(tmp_dir, executable)
    avg_runtime = measure_runtime(tmp_dir, prefix, executable, num_runs)
    return avg_runtime


def benchmark(tmp_dir, logs_dir, prefix, plots_dir):
    costs = parse_critical_comp_costs(tmp_dir, prefix)

    baseline_runtime = measure_baseline_runtime(tmp_dir, prefix, NUM_RUNS)
    print(f"Baseline average runtime: {baseline_runtime:.6f} seconds")

    print("\n=== Measuring adjusted runtime for example.exe ===")
    avg_runtime_example = measure_runtime(tmp_dir, prefix, "example.exe", NUM_RUNS)
    adjusted_runtime_example = avg_runtime_example - baseline_runtime

    budgets = []
    runtimes = []

    for cost in costs:
        print(f"\n=== Processing computation cost budget: {cost} ===")
        fpoptflags = []
        for flag in FPOPTFLAGS_BASE:
            if flag.startswith("--fpopt-comp-cost-budget="):
                fpoptflags.append(f"--fpopt-comp-cost-budget={cost}")
            elif flag.startswith("--fpopt-log-path="):
                fpoptflags.append(f"--fpopt-log-path=tmp/{prefix}example.txt")
            else:
                fpoptflags.append(flag)

        output_binary = f"example-fpopt-{cost}.exe"

        compile_example_fpopt_exe(tmp_dir, prefix, fpoptflags, output=output_binary, verbose=False)

        avg_runtime = measure_runtime(tmp_dir, prefix, output_binary, NUM_RUNS)

        adjusted_runtime = avg_runtime - baseline_runtime

        budgets.append(cost)
        runtimes.append(adjusted_runtime)

    plot_results(plots_dir, prefix, budgets, runtimes, example_adjusted_runtime=adjusted_runtime_example)


def build_with_benchmark(tmp_dir, logs_dir, plots_dir, prefix):
    build_all(tmp_dir, logs_dir, prefix)
    benchmark(tmp_dir, logs_dir, prefix, plots_dir)


def main():
    parser = argparse.ArgumentParser(description="Run the example C code with prefix handling.")
    parser.add_argument("--prefix", type=str, required=True, help="Prefix for intermediate files (e.g., rosa-ex23-)")
    parser.add_argument("--clean", action="store_true", help="Clean up generated files")
    parser.add_argument("--build", action="store_true", help="Build all components")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--all", action="store_true", help="Build and run benchmark")
    args = parser.parse_args()

    prefix = args.prefix
    if not prefix.endswith("-"):
        prefix += "-"

    tmp_dir = "tmp"
    logs_dir = "logs"
    plots_dir = "plots"

    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    if args.clean:
        clean(tmp_dir, logs_dir, prefix)
        sys.exit(0)
    elif args.build:
        build_all(tmp_dir, logs_dir, prefix)
        sys.exit(0)
    elif args.benchmark:
        benchmark(tmp_dir, logs_dir, prefix, plots_dir)
        sys.exit(0)
    elif args.all:
        build_with_benchmark(tmp_dir, logs_dir, plots_dir, prefix)
        sys.exit(0)
    else:
        build_with_benchmark(tmp_dir, logs_dir, plots_dir, prefix)


if __name__ == "__main__":
    main()
