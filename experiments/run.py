#!/usr/bin/env python3

import os
import subprocess
import sys
import shutil
import time
import re
import argparse
import matplotlib.pyplot as plt
import math
import numpy as np
from statistics import mean

from tqdm import tqdm, trange

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
    "--fpopt-num-samples=1000",
    "-mllvm",
    "--fpopt-cost-model-path=../microbm/cm.csv",
]

SRC = "example.c"
LOGGER = "fp-logger.cpp"
EXE = ["example.exe", "example-logged.exe", "example-fpopt.exe"]
NUM_RUNS = 100
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


def clean(tmp_dir, logs_dir, plots_dir):
    print("=== Cleaning up generated files ===")
    directories = [tmp_dir, logs_dir, plots_dir]
    for directory in directories:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"Removed directory: {directory}")


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
    for i in trange(1, num_runs + 1):
        try:
            result = subprocess.run([exe_path], capture_output=True, text=True, check=True)
            output = result.stdout
            match = re.search(r"Total runtime: ([\d\.]+) seconds", output)
            if match:
                runtime = float(match.group(1))
                runtimes.append(runtime)
            else:
                print(f"Could not parse runtime from output on run {i}")
                sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"Error running {exe_path} on run {i}")
            sys.exit(e.returncode)
    average_runtime = mean(runtimes)
    print(f"Average runtime for {executable}: {average_runtime:.6f} seconds")
    return average_runtime


def get_values_file_path(tmp_dir, prefix, binary_name):
    return os.path.join(tmp_dir, f"{prefix}{binary_name}-values.txt")


def generate_example_values(tmp_dir, prefix):
    binary_name = "example.exe"
    exe = os.path.join(tmp_dir, f"{prefix}{binary_name}")
    output_values_file = get_values_file_path(tmp_dir, prefix, binary_name)
    cmd = [exe, "--output-path", output_values_file]
    run_command(cmd, f"Generating function values from {binary_name}", verbose=False)


def generate_values(tmp_dir, prefix, binary_name):
    exe = os.path.join(tmp_dir, f"{prefix}{binary_name}")
    values_file = get_values_file_path(tmp_dir, prefix, binary_name)
    cmd = [exe, "--output-path", values_file]
    run_command(cmd, f"Generating function values from {binary_name}", verbose=False)


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
            print(f"Values file {values_file} does not exist. Generating it now.")
            generate_values(tmp_dir, prefix, binary)
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
            error = abs((v - g) / g) * 100
            valid_errors.append(error)

        if not valid_errors:
            print(f"No valid data to compute rel error for binary {binary}. Setting rel error to None.")
            errors[binary] = None
            continue

        try:
            log_sum = sum(math.log1p(e) for e in valid_errors)
            geo_mean = math.expm1(log_sum / len(valid_errors))
            errors[binary] = geo_mean
        except OverflowError:
            print(
                f"Overflow error encountered while computing geometric mean for binary {binary}. Setting rel error to None."
            )
            errors[binary] = None
        except ZeroDivisionError:
            print(f"No valid errors to compute geometric mean for binary {binary}. Setting rel error to None.")
            errors[binary] = None

    return errors


def plot_results(plots_dir, prefix, budgets, runtimes, errors, example_adjusted_runtime=None, example_rel_err=None):
    plot_filename = os.path.join(plots_dir, f"runtime_error_plot_{prefix[:-1]}.png")
    print(f"=== Plotting results to {plot_filename} ===")

    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(20, 6))

    # First Plot: Computation Cost Budget vs Runtime and Relative Error
    color_runtime = "tab:blue"
    ax1.set_xlabel("Computation Cost Budget")
    ax1.set_ylabel("Runtimes (seconds)", color=color_runtime)
    (line1,) = ax1.plot(budgets, runtimes, marker="o", linestyle="-", label="Optimized Runtimes", color=color_runtime)
    if example_adjusted_runtime is not None:
        line2 = ax1.axhline(y=example_adjusted_runtime, color=color_runtime, linestyle=":", label="Original Runtime")
    ax1.tick_params(axis="y", labelcolor=color_runtime)

    ax2 = ax1.twinx()
    color_error = "tab:green"
    ax2.set_ylabel("Relative Errors (%)", color=color_error)
    (line3,) = ax2.plot(
        budgets, errors, marker="s", linestyle="-", label="Optimized Relative Errors", color=color_error
    )
    if example_rel_err is not None:
        line4 = ax2.axhline(y=example_rel_err, color=color_error, linestyle=":", label="Original Relative Error")
    ax2.tick_params(axis="y", labelcolor=color_error)
    ax2.set_yscale("log")

    ax1.set_title(f"Computation Cost Budget vs Runtime and Relative Error ({prefix[:-1]})")
    ax1.grid(True)

    lines = [line1, line3]
    labels = [line.get_label() for line in lines]
    if example_adjusted_runtime is not None:
        lines.append(line2)
        labels.append(line2.get_label())
    if example_rel_err is not None:
        lines.append(line4)
        labels.append(line4.get_label())

    ax1.legend(lines, labels, loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.0)

    # Second Plot: Pareto Front of Optimized Programs
    ax3.set_xlabel("Runtimes (seconds)")
    ax3.set_ylabel("Relative Errors (%)")
    ax3.set_title(f"Pareto Front of Optimized {prefix[:-1]} Binaries")

    scatter1 = ax3.scatter(runtimes, errors, label="Optimized Programs", color="blue")

    if example_adjusted_runtime is not None and example_rel_err is not None:
        scatter2 = ax3.scatter(
            example_adjusted_runtime, example_rel_err, marker="x", color="red", s=100, label="Original Program"
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

    (line_pareto,) = ax3.plot(
        pareto_front[:, 0], pareto_front[:, 1], linestyle="-", color="purple", label="Pareto Front"
    )
    ax3.set_yscale("log")

    ax3.grid(True)

    pareto_lines = [scatter1, line_pareto]
    pareto_labels = [scatter1.get_label(), line_pareto.get_label()]
    if example_adjusted_runtime is not None and example_rel_err is not None:
        pareto_lines.append(scatter2)
        pareto_labels.append(scatter2.get_label())

    ax3.legend(pareto_lines, pareto_labels, loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.0)

    plt.tight_layout()
    plt.subplots_adjust(right=0.75)

    plt.savefig(plot_filename, bbox_inches="tight")
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
    avg_runtime = measure_runtime(tmp_dir, prefix, executable, num_runs)
    return avg_runtime


def benchmark(tmp_dir, logs_dir, prefix, plots_dir):
    costs = parse_critical_comp_costs(tmp_dir, prefix)

    baseline_runtime = measure_baseline_runtime(tmp_dir, prefix, NUM_RUNS)
    print(f"Baseline average runtime: {baseline_runtime:.6f} seconds")

    avg_runtime_example = measure_runtime(tmp_dir, prefix, "example.exe", NUM_RUNS)
    adjusted_runtime_example = avg_runtime_example - baseline_runtime

    generate_example_values(tmp_dir, prefix)

    generate_golden_values(tmp_dir, prefix)

    golden_values_file = get_values_file_path(tmp_dir, prefix, "golden.exe")
    example_binary = "example.exe"
    rel_errs_example = get_avg_rel_error(tmp_dir, prefix, golden_values_file, [example_binary])
    rel_err_example = rel_errs_example[example_binary]
    print(f"Average Rel Error for example.exe: {rel_err_example}")

    budgets = []
    runtimes = []
    errors = []

    optimized_binaries = []

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

        generate_values(tmp_dir, prefix, output_binary)
        optimized_binaries.append(output_binary)

    errors_dict = get_avg_rel_error(tmp_dir, prefix, golden_values_file, optimized_binaries)
    for binary in optimized_binaries:
        errors.append(errors_dict[binary])
        print(f"Average rel error for {binary}: {errors_dict[binary]}")

    plot_results(
        plots_dir,
        prefix,
        budgets,
        runtimes,
        errors,
        example_adjusted_runtime=adjusted_runtime_example,
        example_rel_err=rel_err_example,
    )


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
        clean(tmp_dir, logs_dir, plots_dir)
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
