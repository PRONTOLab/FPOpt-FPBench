#!/usr/bin/env python3

import os
import subprocess
import sys
import shutil
import time
import re
import argparse
import matplotlib.pyplot as plt
import scipy.stats.mstats as ssm
import math
import random
import numpy as np
from statistics import mean
import pickle

from tqdm import tqdm, trange
from matplotlib import rcParams
from concurrent.futures import ProcessPoolExecutor, as_completed

HOME = "/home/sbrantq"
ENZYME_PATH = os.path.join(HOME, "sync/Enzyme/build-release/Enzyme/ClangEnzyme-16.so")
LLVM_PATH = os.path.join(HOME, "llvms/llvm16/build-release/bin")
CXX = os.path.join(LLVM_PATH, "clang++")

CXXFLAGS = [
    "-O3",
    "-I" + os.path.join(HOME, "include"),
    "-L" + os.path.join(HOME, "lib"),
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
    "-fno-finite-math-only",
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
    "--herbie-num-threads=8",
    "-mllvm",
    "--herbie-timeout=1000",
    "-mllvm",
    "--fpopt-num-samples=1024",
    "-mllvm",
    "--fpopt-cost-model-path=/home/sbrantq/sync/FPBench/microbm/cm.csv",
    "-mllvm",
    "-fpopt-cache-path=cache",
]

SRC = "example.c"
LOGGER = "fp-logger.cpp"
EXE = ["example.exe", "example-logged.exe", "example-fpopt.exe"]
NUM_RUNS = 10
DRIVER_NUM_SAMPLES = 10000000
LOG_NUM_SAMPLES = 10000
MAX_TESTED_COSTS = 999


def geomean(values):
    assert len(values) > 0, "Cannot compute geometric mean of an empty list"
    sum_log = 0.0
    nonzero_count = 0

    for x in values:
        if x != 0:
            sum_log += math.log(x)
            nonzero_count += 1

    if nonzero_count == 0:
        return 0.0

    return math.exp(sum_log / nonzero_count)


def run_command(command, description, capture_output=False, output_file=None, verbose=True, timeout=None):
    print(f"=== {description} ===")
    print("Running:", " ".join(command))
    try:
        if capture_output and output_file:
            with open(output_file, "w") as f:
                subprocess.check_call(command, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
        elif capture_output:
            result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=timeout)
            return result.stdout
        else:
            if verbose:
                subprocess.check_call(command, timeout=timeout)
            else:
                subprocess.check_call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"Command '{' '.join(command)}' timed out after {timeout} seconds.")
        return
    except subprocess.CalledProcessError as e:
        print(f"Error during: {description}")
        if capture_output and output_file:
            print(f"Check the output file: {output_file} for details.")
        else:
            print(e)
        sys.exit(e.returncode)


def clean(tmp_dir, logs_dir, plots_dir):
    print("=== Cleaning up generated files ===")
    directories = [tmp_dir, logs_dir, plots_dir]
    for directory in directories:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"Removed directory: {directory}")


def clean_tmp_except_pkl(tmp_dir):
    for entry in os.listdir(tmp_dir):
        full_path = os.path.join(tmp_dir, entry)
        if os.path.isfile(full_path) and not full_path.endswith(".pkl"):
            os.remove(full_path)
            print(f"Removed file: {full_path}")
        elif os.path.isdir(full_path):
            shutil.rmtree(full_path)
            print(f"Removed directory: {full_path}")


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
        ["python3", script, src_prefixed, dest_prefixed, "example", str(LOG_NUM_SAMPLES)],
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
        except subprocess.TimeoutExpired:
            print(f"Execution of {exe} timed out.")
            if os.path.exists(exe):
                os.remove(exe)
                print(f"Removed executable {exe} due to timeout.")
            return
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


def parse_critical_comp_costs(tmp_dir, prefix):
    budgets_file = os.path.join("cache", "budgets.txt")
    print(f"=== Reading critical computation costs from {budgets_file} ===")
    if not os.path.exists(budgets_file):
        print(f"Budgets file {budgets_file} does not exist.")
        sys.exit(1)
    with open(budgets_file, "r") as f:
        content = f.read().strip()
    if not content:
        print(f"Budgets file {budgets_file} is empty.")
        sys.exit(1)
    try:
        costs = [int(cost.strip()) for cost in content.split(",") if cost.strip() != ""]
    except ValueError as e:
        print(f"Error parsing budgets from file {budgets_file}: {e}")
        sys.exit(1)
    print(f"Read computation costs: {costs}")
    if not costs:
        print("No valid computation costs found in budgets.txt.")
        sys.exit(1)
    num_to_sample = min(MAX_TESTED_COSTS, len(costs))
    sampled_costs = random.sample(costs, num_to_sample)
    sampled_costs_sorted = sorted(sampled_costs)
    print(f"Sampled computation costs (sorted): {sampled_costs_sorted}")
    return sampled_costs_sorted


def measure_runtime(tmp_dir, prefix, executable, num_runs=NUM_RUNS):
    print(f"=== Measuring runtime for {executable} ===")
    runtimes = []
    exe_path = os.path.join(tmp_dir, f"{prefix}{executable}")
    for i in trange(1, num_runs + 1):
        try:
            result = subprocess.run([exe_path], capture_output=True, text=True, check=True, timeout=300)
            output = result.stdout
            match = re.search(r"Total runtime: ([\d\.]+) seconds", output)
            if match:
                runtime = float(match.group(1))
                runtimes.append(runtime)
            else:
                print(f"Could not parse runtime from output on run {i}")
                sys.exit(1)
        except subprocess.TimeoutExpired:
            print(f"Execution of {exe_path} timed out on run {i}")
            if os.path.exists(exe_path):
                os.remove(exe_path)
                print(f"Removed executable {exe_path} due to timeout.")
            return None
        except subprocess.CalledProcessError as e:
            print(f"Error running {exe_path} on run {i}")
            sys.exit(e.returncode)
    if runtimes:
        average_runtime = mean(runtimes)
        print(f"Average runtime for {prefix}{executable}: {average_runtime:.6f} seconds")
        return average_runtime
    else:
        print(f"No successful runs for {prefix}{executable}")
        return None


def get_values_file_path(tmp_dir, prefix, binary_name):
    return os.path.join(tmp_dir, f"{prefix}{binary_name}-values.txt")


def generate_example_values(tmp_dir, prefix):
    binary_name = "example.exe"
    exe = os.path.join(tmp_dir, f"{prefix}{binary_name}")
    output_values_file = get_values_file_path(tmp_dir, prefix, binary_name)
    cmd = [exe, "--output-path", output_values_file]
    run_command(cmd, f"Generating function values from {binary_name}", verbose=False, timeout=300)


def generate_values(tmp_dir, prefix, binary_name):
    exe = os.path.join(tmp_dir, f"{prefix}{binary_name}")
    values_file = get_values_file_path(tmp_dir, prefix, binary_name)
    cmd = [exe, "--output-path", values_file]
    run_command(cmd, f"Generating function values from {binary_name}", verbose=False, timeout=300)


def compile_golden_exe(tmp_dir, prefix):
    source = os.path.join(tmp_dir, f"{prefix}golden.cpp")
    output = os.path.join(tmp_dir, f"{prefix}golden.exe")
    cmd = [CXX, source] + CXXFLAGS + ["-o", output]
    run_command(cmd, f"Compiling {output}")


def generate_golden_values(tmp_dir, prefix):
    script = "fpopt-golden-driver-generator.py"
    src_prefixed = os.path.join(tmp_dir, f"{prefix}{SRC}")
    dest_prefixed = os.path.join(tmp_dir, f"{prefix}golden.cpp")
    cur_prec = 128
    max_prec = 4096
    PREC_step = 128
    prev_output = None
    output_values_file = get_values_file_path(tmp_dir, prefix, "golden.exe")
    while cur_prec <= max_prec:
        run_command(
            ["python3", script, src_prefixed, dest_prefixed, str(cur_prec), "example", str(DRIVER_NUM_SAMPLES)],
            f"Generating golden.cpp with PREC={cur_prec}",
        )
        if not os.path.exists(dest_prefixed):
            print(f"Failed to generate {dest_prefixed}.")
            sys.exit(1)
        print(f"Generated {dest_prefixed} successfully.")

        compile_golden_exe(tmp_dir, prefix)

        exe = os.path.join(tmp_dir, f"{prefix}golden.exe")
        cmd = [exe, "--output-path", output_values_file]
        run_command(cmd, f"Generating golden values with PREC={cur_prec}", verbose=False)

        if not os.path.exists(output_values_file):
            print(f"Failed to generate golden values at PREC={cur_prec} due to timeout.")
            return  # Assume golden values do not exist

        with open(output_values_file, "r") as f:
            output = f.read()

        if output == prev_output:
            print(f"Golden values converged at PREC={cur_prec}")
            break
        else:
            prev_output = output
            cur_prec += PREC_step
    else:
        print(f"Failed to converge golden values up to PREC={max_prec}")
        sys.exit(1)


def get_avg_rel_error(tmp_dir, prefix, golden_values_file, binaries):
    with open(golden_values_file, "r") as f:
        golden_values = [float(line.strip()) for line in f]

    errors = {}
    for binary in binaries:
        values_file = get_values_file_path(tmp_dir, prefix, binary)
        if not os.path.exists(values_file):
            print(f"Values file {values_file} does not exist. Skipping error calculation for {binary}.")
            errors[binary] = None
            continue
        with open(values_file, "r") as f:
            values = [float(line.strip()) for line in f]
        if len(values) != len(golden_values):
            print(f"Number of values in {values_file} does not match golden values")
            sys.exit(1)

        valid_errors = []
        for v, g in zip(values, golden_values):
            if math.isnan(v) or math.isnan(g):
                continue
            if g == 0:
                continue
            error = max(abs((v - g) / g), abs(math.ulp(g) / g))
            valid_errors.append(error)

        if not valid_errors:
            print(f"No valid data to compute rel error for binary {binary}. Setting rel error to None.")
            errors[binary] = None
            continue

        try:
            errors[binary] = geomean(valid_errors)
        except OverflowError:
            print(
                f"Overflow error encountered while computing geometric mean for binary {binary}. Setting rel error to None."
            )
            errors[binary] = None
        except ZeroDivisionError:
            print(f"No valid errors to compute geometric mean for binary {binary}. Setting rel error to None.")
            errors[binary] = None

    return errors


def plot_results(
    plots_dir,
    prefix,
    budgets,
    runtimes,
    errors,
    original_runtime=None,
    original_error=None,
    output_format="png",
):
    print(f"=== Plotting results to {output_format.upper()} file ===")

    # Filter out entries where runtimes or errors are None
    data = list(zip(budgets, runtimes, errors))
    filtered_data = [(b, r, e) for b, r, e in data if r is not None and e is not None]

    if not filtered_data:
        print("No valid data to plot.")
        return

    budgets, runtimes, errors = zip(*filtered_data)

    rcParams["font.size"] = 20
    rcParams["axes.titlesize"] = 24
    rcParams["axes.labelsize"] = 20
    rcParams["xtick.labelsize"] = 18
    rcParams["ytick.labelsize"] = 18
    rcParams["legend.fontsize"] = 18

    if output_format.lower() == "pdf":
        # First Plot: Computation Cost Budget vs Runtime and Relative Error
        fig1, ax1 = plt.subplots(figsize=(10, 8))  # Adjust size as needed

        color_runtime = "tab:blue"
        ax1.set_xlabel("Computation Cost Budget")
        ax1.set_ylabel("Runtimes (seconds)", color=color_runtime)
        (line1,) = ax1.step(
            budgets, runtimes, marker="o", linestyle="-", label="Optimized Runtimes", color=color_runtime, where="post"
        )
        if original_runtime is not None:
            line2 = ax1.axhline(y=original_runtime, color=color_runtime, linestyle="--", label="Original Runtime")
        ax1.tick_params(axis="y", labelcolor=color_runtime)

        ax2 = ax1.twinx()
        color_error = "tab:green"
        ax2.set_ylabel("Relative Errors", color=color_error)
        (line3,) = ax2.step(
            budgets,
            errors,
            marker="s",
            linestyle="-",
            label="Optimized Relative Errors",
            color=color_error,
            where="post",
        )
        if original_error is not None:
            line4 = ax2.axhline(y=original_error, color=color_error, linestyle="--", label="Original Relative Error")
        ax2.tick_params(axis="y", labelcolor=color_error)
        ax2.set_yscale("symlog", linthresh=1e-14)
        ax2.set_ylim(bottom=0)

        ax1.set_title(f"Computation Cost Budget vs Runtime\nand Relative Error ({prefix[:-1]})")
        ax1.grid(True)

        lines = [line1, line3]
        labels = [line.get_label() for line in lines]
        if original_runtime is not None:
            lines.append(line2)
            labels.append(line2.get_label())
        if original_error is not None:
            lines.append(line4)
            labels.append(line4.get_label())

        ax1.legend(
            lines,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=2,
            borderaxespad=0.0,
            frameon=False,
        )

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        plot_filename1 = os.path.join(plots_dir, f"runtime_plot_{prefix[:-1]}.{output_format}")
        plt.savefig(plot_filename1, bbox_inches="tight", dpi=300)
        plt.close(fig1)
        print(f"First plot saved to {plot_filename1}")

        # Second Plot: Pareto Front of Optimized Programs
        fig2, ax3 = plt.subplots(figsize=(10, 8))  # Adjust size as needed

        ax3.set_xlabel("Runtimes (seconds)")
        ax3.set_ylabel("Relative Errors")
        ax3.set_title(f"Pareto Front of Optimized Programs ({prefix[:-1]})")

        scatter1 = ax3.scatter(runtimes, errors, label="Optimized Programs", color="blue")

        if original_runtime is not None and original_error is not None:
            scatter2 = ax3.scatter(
                original_runtime,
                original_error,
                marker="x",
                color="red",
                s=100,
                label="Original Program",
            )

        # Calculate Pareto Front
        points = np.array(list(zip(runtimes, errors)))
        sorted_indices = np.argsort(points[:, 0])
        sorted_points = points[sorted_indices]

        pareto_front = [sorted_points[0]]
        for point in sorted_points[1:]:
            if point[1] < pareto_front[-1][1]:
                pareto_front.append(point)

        pareto_front = np.array(pareto_front)

        (line_pareto,) = ax3.step(
            pareto_front[:, 0], pareto_front[:, 1], linestyle="-", color="purple", label="Pareto Front", where="post"
        )
        ax3.set_yscale("log")

        ax3.grid(True)

        pareto_lines = [scatter1, line_pareto]
        pareto_labels = [scatter1.get_label(), line_pareto.get_label()]
        if original_runtime is not None and original_error is not None:
            pareto_lines.append(scatter2)
            pareto_labels.append(scatter2.get_label())

        ax3.legend(
            pareto_lines,
            pareto_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=len(pareto_lines),
            borderaxespad=0.0,
            frameon=False,
        )

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)

        plot_filename2 = os.path.join(plots_dir, f"pareto_front_plot_{prefix[:-1]}.{output_format}")
        plt.savefig(plot_filename2, bbox_inches="tight", dpi=300)
        plt.close(fig2)
        print(f"Second plot saved to {plot_filename2}")
    else:
        # Existing behavior for non-PDF formats
        plot_filename = os.path.join(plots_dir, f"runtime_error_plot_{prefix[:-1]}.{output_format}")

        fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(20, 8))

        # First Plot: Computation Cost Budget vs Runtime and Relative Error
        color_runtime = "tab:blue"
        ax1.set_xlabel("Computation Cost Budget")
        ax1.set_ylabel("Runtimes (seconds)", color=color_runtime)
        (line1,) = ax1.step(
            budgets, runtimes, marker="o", linestyle="-", label="Optimized Runtimes", color=color_runtime, where="post"
        )
        if original_runtime is not None:
            line2 = ax1.axhline(y=original_runtime, color=color_runtime, linestyle="--", label="Original Runtime")
        ax1.tick_params(axis="y", labelcolor=color_runtime)

        ax2 = ax1.twinx()
        color_error = "tab:green"
        ax2.set_ylabel("Relative Errors", color=color_error)
        (line3,) = ax2.step(
            budgets,
            errors,
            marker="s",
            linestyle="-",
            label="Optimized Relative Errors",
            color=color_error,
            where="post",
        )
        if original_error is not None:
            line4 = ax2.axhline(y=original_error, color=color_error, linestyle="--", label="Original Relative Error")
        ax2.tick_params(axis="y", labelcolor=color_error)
        ax2.set_yscale("log")

        ax1.set_title(f"Computation Cost Budget vs Runtime and Relative Error ({prefix[:-1]})")
        ax1.grid(True)

        lines = [line1, line3]
        labels = [line.get_label() for line in lines]
        if original_runtime is not None:
            lines.append(line2)
            labels.append(line2.get_label())
        if original_error is not None:
            lines.append(line4)
            labels.append(line4.get_label())

        ax1.legend(
            lines,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=len(lines),
            borderaxespad=0.0,
            frameon=False,
        )

        # Second Plot: Pareto Front of Optimized Programs
        ax3.set_xlabel("Runtimes (seconds)")
        ax3.set_ylabel("Relative Errors")
        ax3.set_title(f"Pareto Front of Optimized Programs ({prefix[:-1]})")

        scatter1 = ax3.scatter(runtimes, errors, label="Optimized Programs", color="blue")

        if original_runtime is not None and original_error is not None:
            scatter2 = ax3.scatter(
                original_runtime,
                original_error,
                marker="x",
                color="red",
                s=100,
                label="Original Program",
            )

        # Calculate Pareto Front
        points = np.array(list(zip(runtimes, errors)))
        sorted_indices = np.argsort(points[:, 0])
        sorted_points = points[sorted_indices]

        pareto_front = [sorted_points[0]]
        for point in sorted_points[1:]:
            if point[1] <= pareto_front[-1][1]:
                pareto_front.append(point)

        pareto_front = np.array(pareto_front)

        (line_pareto,) = ax3.step(
            pareto_front[:, 0], pareto_front[:, 1], linestyle="-", color="purple", label="Pareto Front", where="post"
        )
        ax3.set_yscale("log")

        ax3.grid(True)

        pareto_lines = [scatter1, line_pareto]
        pareto_labels = [scatter1.get_label(), line_pareto.get_label()]
        if original_runtime is not None and original_error is not None:
            pareto_lines.append(scatter2)
            pareto_labels.append(scatter2.get_label())

        ax3.legend(
            pareto_lines,
            pareto_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=len(pareto_lines),
            borderaxespad=0.0,
            frameon=False,
        )

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)

        plt.savefig(plot_filename, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Plot saved to {plot_filename}")


def build_all(tmp_dir, logs_dir, prefix):
    generate_example_cpp(tmp_dir, prefix)
    generate_example_logged_cpp(tmp_dir, prefix)
    # generate_example_baseline_cpp(tmp_dir, prefix)
    compile_example_exe(tmp_dir, prefix)
    compile_example_logged_exe(tmp_dir, prefix)
    # compile_example_baseline_exe(tmp_dir, prefix)
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
    avg_runtime = measure_runtime(tmp_dir, prefix, executable, num_runs)
    return avg_runtime


def process_cost(args):
    cost, tmp_dir, prefix = args

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

    generate_values(tmp_dir, prefix, output_binary)

    return cost, output_binary


def benchmark(tmp_dir, logs_dir, prefix, plots_dir, num_parallel=1):
    costs = parse_critical_comp_costs(tmp_dir, prefix)

    original_avg_runtime = measure_runtime(tmp_dir, prefix, "example.exe", NUM_RUNS)
    original_runtime = original_avg_runtime

    if original_runtime is None:
        print("Original binary timed out. Proceeding as if it doesn't exist.")
        return

    generate_example_values(tmp_dir, prefix)

    generate_golden_values(tmp_dir, prefix)

    golden_values_file = get_values_file_path(tmp_dir, prefix, "golden.exe")
    example_binary = "example.exe"
    rel_errs_example = get_avg_rel_error(tmp_dir, prefix, golden_values_file, [example_binary])
    rel_err_example = rel_errs_example[example_binary]
    print(f"Average Rel Error for {prefix}example.exe: {rel_err_example}")

    data_tuples = []

    args_list = [(cost, tmp_dir, prefix) for cost in costs]

    if num_parallel == 1:
        for args in args_list:
            cost, output_binary = process_cost(args)
            data_tuples.append((cost, output_binary))
    else:
        with ProcessPoolExecutor(max_workers=num_parallel) as executor:
            future_to_cost = {executor.submit(process_cost, args): args[0] for args in args_list}
            for future in as_completed(future_to_cost):
                cost = future_to_cost[future]
                try:
                    cost_result, output_binary = future.result()
                    data_tuples.append((cost_result, output_binary))
                except Exception as exc:
                    print(f"Cost {cost} generated an exception: {exc}")

    data_tuples_sorted = sorted(data_tuples, key=lambda x: x[0])
    sorted_budgets, sorted_optimized_binaries = zip(*data_tuples_sorted) if data_tuples_sorted else ([], [])

    # Measure runtimes serially based on sorted budgets
    sorted_runtimes = []
    for cost, output_binary in zip(sorted_budgets, sorted_optimized_binaries):
        avg_runtime = measure_runtime(tmp_dir, prefix, output_binary, NUM_RUNS)
        if avg_runtime is not None:
            sorted_runtimes.append(avg_runtime)
        else:
            print(f"Skipping cost {cost} due to runtime measurement failure.")
            sorted_runtimes.append(None)

    errors_dict = get_avg_rel_error(tmp_dir, prefix, golden_values_file, sorted_optimized_binaries)
    sorted_errors = []
    for binary in sorted_optimized_binaries:
        sorted_errors.append(errors_dict.get(binary))
        print(f"Average rel error for {binary}: {errors_dict.get(binary)}")

    sorted_budgets = list(sorted_budgets)
    sorted_runtimes = list(sorted_runtimes)
    sorted_errors = list(sorted_errors)

    data = {
        "budgets": sorted_budgets,
        "runtimes": sorted_runtimes,
        "errors": sorted_errors,
        "original_runtime": original_runtime,
        "original_error": rel_err_example,
    }
    data_file = os.path.join(tmp_dir, f"{prefix}benchmark_data.pkl")
    with open(data_file, "wb") as f:
        pickle.dump(data, f)
    print(f"Benchmark data saved to {data_file}")

    plot_results(
        plots_dir,
        prefix,
        sorted_budgets,
        sorted_runtimes,
        sorted_errors,
        original_runtime=original_runtime,
        original_error=rel_err_example,
    )


def plot_from_data(tmp_dir, plots_dir, prefix, output_format="png"):
    data_file = os.path.join(tmp_dir, f"{prefix}benchmark_data.pkl")
    if not os.path.exists(data_file):
        print(f"Data file {data_file} does not exist. Cannot plot.")
        sys.exit(1)
    with open(data_file, "rb") as f:
        data = pickle.load(f)
    plot_results(
        plots_dir,
        prefix,
        data["budgets"],
        data["runtimes"],
        data["errors"],
        original_runtime=data["original_runtime"],
        original_error=data["original_error"],
        output_format=output_format,
    )


def analyze_all_data(tmp_dir, thresholds=None):
    prefixes = []
    data_list = []

    # Load all benchmark data files
    for filename in os.listdir(tmp_dir):
        if filename.endswith("benchmark_data.pkl"):
            data_file = os.path.join(tmp_dir, filename)
            with open(data_file, "rb") as f:
                data = pickle.load(f)
            prefix = filename[: -len("benchmark_data.pkl")]
            prefixes.append(prefix)
            data_list.append((prefix, data))

    print("Number of tested FPBench functions: ", len(data_list))
    if not data_list:
        print("No benchmark data files found in the tmp directory.")
        return

    print(f"Analyzing data for prefixes: {', '.join(prefixes)}\n")

    if thresholds is None:
        thresholds = [
            0,
            1e-15,
            1e-14,
            1e-13,
            1e-12,
            1e-11,
            1e-10,
            1e-9,
            1e-8,
            1e-7,
            1e-6,
            1e-5,
            5e-5,
            1e-4,
            1e-3,
            1e-2,
            1e-1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.9,
            1,
        ]

    max_accuracy_improvements = {}  # Per benchmark
    min_runtime_ratios = {threshold: {} for threshold in thresholds}

    original_digits = []
    for prefix, data in data_list:
        budgets = data["budgets"]
        runtimes = data["runtimes"]
        errors = data["errors"]
        original_runtime = data["original_runtime"]
        original_error = data["original_error"]
        # print(f"Analyzing {prefix}...")

        # print(f"Original runtime: {original_runtime}")
        # print(f"Original error: {original_error}")
        # for budget, runtime, error in zip(budgets, runtimes, errors):
        #     print(f"Budget: {budget}, Runtime: {runtime}, Error: {error}")
        # print()

        # For each threshold, find the minimum runtime ratio for this benchmark
        for threshold in thresholds:
            min_ratio = None
            for err, runtime in zip(errors, runtimes):
                if err is not None and runtime is not None and err <= threshold:
                    runtime_ratio = runtime / original_runtime
                    if min_ratio is None or runtime_ratio < min_ratio:
                        min_ratio = runtime_ratio
            if min_ratio is not None:
                min_runtime_ratios[threshold][prefix] = min_ratio

    overall_runtime_improvements = {}
    for threshold in thresholds:
        ratios = min_runtime_ratios[threshold].values()
        if ratios:
            log_sum = sum(math.log(min(1, ratio)) for ratio in ratios)
            geo_mean_ratio = math.exp(log_sum / len(ratios))
            percentage_improvement = (1 - geo_mean_ratio) * 100
            overall_runtime_improvements[threshold] = percentage_improvement
        else:
            overall_runtime_improvements[threshold] = None

    print("\nGeometric average percentage of runtime improvements while allowing some level of relative error:")
    for threshold in thresholds:
        percentage_improvement = overall_runtime_improvements[threshold]
        if percentage_improvement is not None:
            print(
                f"Allowed relative error ≤ {threshold} ({len(min_runtime_ratios[threshold])} benchmarks): "
                f"{percentage_improvement:.2f}% runtime reduction / {1 / (1 - percentage_improvement / 100):.2f}x average speedup"
            )
        else:
            print(f"Allowed relative error ≤ {threshold}: No data")

    max_speedups = {}
    max_speedup_prefixes = {}
    for threshold in thresholds:
        if min_runtime_ratios[threshold]:
            best_prefix = min(min_runtime_ratios[threshold], key=min_runtime_ratios[threshold].get)
            best_ratio = min_runtime_ratios[threshold][best_prefix]
            max_speedup = 1 / best_ratio if best_ratio > 0 else float("inf")
            max_speedups[threshold] = max_speedup
            max_speedup_prefixes[threshold] = best_prefix
        else:
            max_speedups[threshold] = None
            max_speedup_prefixes[threshold] = None

    print("\nMaximum speedup on a single benchmark for each threshold:")
    for threshold in thresholds:
        prefix = max_speedup_prefixes[threshold]
        if prefix is not None:
            print(f"Allowed relative error ≤ {threshold}: {max_speedups[threshold]:.2f}x speedup ({prefix})")
        else:
            print(f"Allowed relative error ≤ {threshold}: No data")

    max_accuracy_improvement_ratio = 0.0
    max_accuracy_improvement_prefix = None
    better_accuracy_count = 0

    for prefix, data in data_list:
        orig_err = data["original_error"]
        # Consider only positive non-None errors
        valid_optimized_errors = [err for err in data["errors"] if err is not None and err > 0]
        if valid_optimized_errors and orig_err is not None:
            best_optimized_error = min(valid_optimized_errors)
            if best_optimized_error < orig_err:
                better_accuracy_count += 1
                improvement_ratio = orig_err / best_optimized_error
                if improvement_ratio > max_accuracy_improvement_ratio:
                    max_accuracy_improvement_ratio = improvement_ratio
                    max_accuracy_improvement_prefix = prefix

    if max_accuracy_improvement_prefix is not None:
        print(
            f"\nMaximum accuracy improvement ratio: {max_accuracy_improvement_ratio:.2f}x (in benchmark: {max_accuracy_improvement_prefix})"
        )
    else:
        print("\nNo accuracy improvements found.")

    print(f"\nNumber of benchmarks where we can get better accuracy: {better_accuracy_count}")

    # Geometric mean of original relative errors
    original_errors = [
        data["original_error"]
        for _, data in data_list
        if data["original_error"] is not None and data["original_error"] > 0
    ]

    if original_errors:
        log_sum = sum(math.log(err) for err in original_errors)
        geomean_error = math.exp(log_sum / len(original_errors))
        print("Geometric mean of original relative errors:", geomean_error)
    else:
        print("No valid original errors found.")

    # Compute the geometric mean of error reduction ratios
    reduction_ratios = []
    for prefix, data in data_list:
        orig_err = data["original_error"]
        valid_optimized_errors = [err for err in data["errors"] if err is not None and err > 0]
        if valid_optimized_errors and orig_err is not None and orig_err > 0:
            best_optimized_error = min(valid_optimized_errors)
            if best_optimized_error < orig_err:
                reduction_ratio = best_optimized_error / orig_err
                reduction_ratios.append(reduction_ratio)

    if reduction_ratios:
        log_sum = sum(math.log(ratio) for ratio in reduction_ratios)
        geomean_reduction = math.exp(log_sum / len(reduction_ratios))
        improvement_factor = 1 / geomean_reduction
        print("\nGeometric mean of error reduction:", geomean_reduction)
        print("Average improvement factor in error:", improvement_factor, "x")
    else:
        print("\nNo error improvements found.")


def build_with_benchmark(tmp_dir, logs_dir, plots_dir, prefix, num_parallel=1):
    build_all(tmp_dir, logs_dir, prefix)
    benchmark(tmp_dir, logs_dir, prefix, plots_dir, num_parallel)


def remove_cache_dir():
    cache_dir = "cache"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print("=== Removed existing cache directory ===")


def main():
    parser = argparse.ArgumentParser(description="Run the example C code with prefix handling.")
    parser.add_argument("--prefix", type=str, required=True, help="Prefix for intermediate files (e.g., rosa-ex23-)")
    parser.add_argument("--clean", action="store_true", help="Clean up generated files")
    parser.add_argument("--build", action="store_true", help="Build all components")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--all", action="store_true", help="Build and run benchmark")
    parser.add_argument("--plot-only", action="store_true", help="Plot results from existing data")
    parser.add_argument("--output-format", type=str, default="png", help="Output format for plots (e.g., png, pdf)")
    parser.add_argument("--analytics", action="store_true", help="Run analytics on saved data")
    parser.add_argument("--disable-preopt", action="store_true", help="Disable Enzyme preoptimization")
    parser.add_argument(
        "--num-parallel", type=int, default=16, help="Number of parallel processes to use (default: 16)"
    )
    args = parser.parse_args()

    global FPOPTFLAGS_BASE
    if args.disable_preopt:
        FPOPTFLAGS_BASE.extend(["-mllvm", "--enzyme-preopt=0"])

    prefix = args.prefix
    if not prefix.endswith("-"):
        prefix += "-"

    tmp_dir = "tmp"
    logs_dir = "logs"
    plots_dir = "plots"

    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    remove_cache_dir()

    if args.clean:
        clean(tmp_dir, logs_dir, plots_dir)
        sys.exit(0)
    elif args.build:
        build_all(tmp_dir, logs_dir, prefix)
        sys.exit(0)
    elif args.benchmark:
        benchmark(tmp_dir, logs_dir, prefix, plots_dir, num_parallel=args.num_parallel)
        clean_tmp_except_pkl(tmp_dir)
        sys.exit(0)
    elif args.plot_only:
        plot_from_data(tmp_dir, plots_dir, prefix, output_format=args.output_format)
        sys.exit(0)
    elif args.analytics:
        analyze_all_data(tmp_dir)
        sys.exit(0)
    elif args.all:
        build_with_benchmark(tmp_dir, logs_dir, plots_dir, prefix, num_parallel=args.num_parallel)
        clean_tmp_except_pkl(tmp_dir)
        sys.exit(0)
    else:
        build_with_benchmark(tmp_dir, logs_dir, plots_dir, prefix, num_parallel=args.num_parallel)
        clean_tmp_except_pkl(tmp_dir)


if __name__ == "__main__":
    main()
