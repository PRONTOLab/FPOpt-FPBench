#!/usr/bin/env python3

import os
import subprocess
import sys
import shutil
import re
import argparse
import matplotlib.pyplot as plt
import math
import random
import numpy as np
from statistics import mean
import pickle
from tqdm import trange
from matplotlib import rcParams
from concurrent.futures import ProcessPoolExecutor, as_completed

HOME = "/home/sbrantq"
ENZYME_PATH = os.path.join(HOME, "sync/Enzyme/build/Enzyme/ClangEnzyme-15.so")
LLVM_PATH = os.path.join(HOME, "llvms/llvm15/build/bin")
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

FPOPTFLAGS_BASE_TEMPLATE = [
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
    "--fpopt-cost-model-path=../microbm/cm.csv",
    "-mllvm",
    "-fpopt-cache-path=cache",
]

SRC = "example.c"
LOGGER = "fp-logger.cpp"
NUM_RUNS = 10
DRIVER_NUM_SAMPLES = 10000000
LOG_NUM_SAMPLES = 10000
MAX_TESTED_COSTS = 999


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


def generate_example_cpp(tmp_dir, original_prefix, prefix):
    script = "fpopt-original-driver-generator.py"
    print(f"=== Running {script} ===")
    src_prefixed = os.path.join(tmp_dir, f"{original_prefix}{SRC}")
    dest_prefixed = os.path.join(tmp_dir, f"{prefix}example.cpp")
    run_command(
        ["python3", script, src_prefixed, dest_prefixed, "example", str(DRIVER_NUM_SAMPLES)],
        f"Generating example.cpp from {SRC}",
    )
    if not os.path.exists(dest_prefixed):
        print(f"Failed to generate {dest_prefixed}.")
        sys.exit(1)
    print(f"Generated {dest_prefixed} successfully.")


def generate_example_logged_cpp(tmp_dir, original_prefix, prefix):
    script = "fpopt-logged-driver-generator.py"
    print(f"=== Running {script} ===")
    src_prefixed = os.path.join(tmp_dir, f"{original_prefix}{SRC}")
    dest_prefixed = os.path.join(tmp_dir, f"{prefix}example-logged.cpp")
    run_command(
        ["python3", script, src_prefixed, dest_prefixed, "example", str(LOG_NUM_SAMPLES)],
        f"Generating example-logged.cpp from {SRC}",
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

    if not costs:
        print("No valid computation costs found to sample.")
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


def generate_golden_values(tmp_dir, original_prefix, prefix):
    script = "fpopt-golden-driver-generator.py"
    src_prefixed = os.path.join(tmp_dir, f"{original_prefix}{SRC}")
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
            return

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


def build_all(tmp_dir, logs_dir, original_prefix, prefix, fpoptflags, example_txt_path):
    generate_example_cpp(tmp_dir, original_prefix, prefix)
    generate_example_logged_cpp(tmp_dir, original_prefix, prefix)
    compile_example_exe(tmp_dir, prefix)
    compile_example_logged_exe(tmp_dir, prefix)
    if not os.path.exists(example_txt_path):
        generate_example_txt(tmp_dir, original_prefix)
    compile_example_fpopt_exe(tmp_dir, prefix, fpoptflags, output="example-fpopt.exe")
    print("=== Initial build process completed successfully ===")


def process_cost(args):
    cost, tmp_dir, prefix, fpoptflags = args

    print(f"\n=== Processing computation cost budget: {cost} ===")
    fpoptflags_cost = []
    for flag in fpoptflags:
        if flag.startswith("--fpopt-comp-cost-budget="):
            fpoptflags_cost.append(f"--fpopt-comp-cost-budget={cost}")
        else:
            fpoptflags_cost.append(flag)

    output_binary = f"example-fpopt-{cost}.exe"

    compile_example_fpopt_exe(tmp_dir, prefix, fpoptflags_cost, output=output_binary, verbose=False)

    generate_values(tmp_dir, prefix, output_binary)

    return cost, output_binary


def benchmark(tmp_dir, logs_dir, original_prefix, prefix, plots_dir, fpoptflags, num_parallel=1):
    costs = parse_critical_comp_costs(tmp_dir, prefix)

    original_avg_runtime = measure_runtime(tmp_dir, prefix, "example.exe", NUM_RUNS)
    original_runtime = original_avg_runtime

    if original_runtime is None:
        print("Original binary timed out. Proceeding as if it doesn't exist.")
        return

    generate_example_values(tmp_dir, prefix)

    generate_golden_values(tmp_dir, original_prefix, prefix)

    golden_values_file = get_values_file_path(tmp_dir, prefix, "golden.exe")
    example_binary = "example.exe"
    rel_errs_example = get_avg_rel_error(tmp_dir, prefix, golden_values_file, [example_binary])
    rel_err_example = rel_errs_example[example_binary]
    print(f"Average Rel Error for {prefix}example.exe: {rel_err_example}")

    data_tuples = []

    args_list = [(cost, tmp_dir, prefix, fpoptflags) for cost in costs]

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

    return data


def build_with_benchmark(
    tmp_dir, logs_dir, plots_dir, original_prefix, prefix, fpoptflags, example_txt_path, num_parallel=1
):
    build_all(tmp_dir, logs_dir, original_prefix, prefix, fpoptflags, example_txt_path)
    data = benchmark(tmp_dir, logs_dir, original_prefix, prefix, plots_dir, fpoptflags, num_parallel)
    return data


def remove_cache_dir():
    cache_dir = "cache"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print("=== Removed existing cache directory ===")


def plot_ablation_results(tmp_dir, plots_dir, original_prefix, prefix, output_format="png"):
    ablation_data_file = os.path.join(tmp_dir, f"{prefix}ablation-widen-range.pkl")
    if not os.path.exists(ablation_data_file):
        print(f"Ablation data file {ablation_data_file} does not exist. Cannot plot.")
        sys.exit(1)
    with open(ablation_data_file, "rb") as f:
        all_data = pickle.load(f)

    rcParams["font.size"] = 20
    rcParams["axes.titlesize"] = 24
    rcParams["axes.labelsize"] = 20
    rcParams["xtick.labelsize"] = 18
    rcParams["ytick.labelsize"] = 18
    rcParams["legend.fontsize"] = 18

    plt.figure(figsize=(10, 8))

    colors = ["blue", "green", "red", "purple", "orange", "brown", "pink", "gray", "olive", "cyan"]
    color_iter = iter(colors)

    for X, data in sorted(all_data.items()):
        budgets = data["budgets"]
        runtimes = data["runtimes"]
        errors = data["errors"]
        original_runtime = data["original_runtime"]
        original_error = data["original_error"]

        data_points = list(zip(runtimes, errors))
        filtered_data = [(r, e) for r, e in data_points if r is not None and e is not None]
        if not filtered_data:
            print(f"No valid data to plot for widen-range={X}.")
            continue
        runtimes_filtered, errors_filtered = zip(*filtered_data)
        color = next(color_iter)
        plt.scatter(runtimes_filtered, errors_filtered, label=f"widen-range={X}", color=color)
        points = np.array(filtered_data)
        sorted_indices = np.argsort(points[:, 0])
        sorted_points = points[sorted_indices]

        pareto_front = [sorted_points[0]]
        for point in sorted_points[1:]:
            if point[1] < pareto_front[-1][1]:
                pareto_front.append(point)

        pareto_front = np.array(pareto_front)

        plt.step(
            pareto_front[:, 0],
            pareto_front[:, 1],
            where="post",
            linestyle="-",
            color=color,
        )

    plt.scatter(original_runtime, original_error, marker="x", color="black", s=100, label="Original Program")

    plt.xlabel("Runtimes (seconds)")
    plt.ylabel("Relative Errors (%)")
    plt.title("Pareto Fronts for Different widen-range Values")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)

    plot_filename = os.path.join(plots_dir, f"{prefix}ablation_widen_range_pareto_front.{output_format}")
    plt.savefig(plot_filename, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Ablation plot saved to {plot_filename}")


def plot_ablation_results_cost_model(tmp_dir, plots_dir, original_prefix, prefix, output_format="png"):
    ablation_data_file = os.path.join(tmp_dir, f"{prefix}ablation-cost-model.pkl")
    if not os.path.exists(ablation_data_file):
        print(f"Ablation data file {ablation_data_file} does not exist. Cannot plot.")
        sys.exit(1)
    with open(ablation_data_file, "rb") as f:
        all_data = pickle.load(f)

    rcParams["font.size"] = 20
    rcParams["axes.titlesize"] = 24
    rcParams["axes.labelsize"] = 20
    rcParams["xtick.labelsize"] = 18
    rcParams["ytick.labelsize"] = 18
    rcParams["legend.fontsize"] = 18

    plt.figure(figsize=(10, 8))

    colors = ["blue", "green"]
    labels = ["With Cost Model", "Without Cost Model"]

    for idx, key in enumerate(["with_cost_model", "without_cost_model"]):
        data = all_data[key]
        budgets = data["budgets"]
        runtimes = data["runtimes"]
        errors = data["errors"]
        original_runtime = data["original_runtime"]
        original_error = data["original_error"]

        data_points = list(zip(runtimes, errors))
        filtered_data = [(r, e) for r, e in data_points if r is not None and e is not None]
        if not filtered_data:
            print(f"No valid data to plot for {labels[idx]}.")
            continue
        runtimes_filtered, errors_filtered = zip(*filtered_data)
        color = colors[idx]
        plt.scatter(runtimes_filtered, errors_filtered, label=labels[idx], color=color)
        points = np.array(filtered_data)
        sorted_indices = np.argsort(points[:, 0])
        sorted_points = points[sorted_indices]

        pareto_front = [sorted_points[0]]
        for point in sorted_points[1:]:
            if point[1] < pareto_front[-1][1]:
                pareto_front.append(point)

        pareto_front = np.array(pareto_front)

        plt.step(
            pareto_front[:, 0],
            pareto_front[:, 1],
            where="post",
            linestyle="-",
            color=color,
        )

    plt.scatter(original_runtime, original_error, marker="x", color="black", s=100, label="Original Program")

    plt.xlabel("Runtimes (seconds)")
    plt.ylabel("Relative Errors (%)")
    plt.title("Pareto Fronts for Cost Model Ablation")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)

    plot_filename = os.path.join(plots_dir, f"{prefix}ablation_cost_model_pareto_front.{output_format}")
    plt.savefig(plot_filename, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Ablation plot saved to {plot_filename}")


def remove_mllvm_flag(flags_list, flag_prefix):
    new_flags = []
    i = 0
    while i < len(flags_list):
        if flags_list[i] == "-mllvm" and i + 1 < len(flags_list) and flags_list[i + 1].startswith(flag_prefix):
            i += 2
        else:
            new_flags.append(flags_list[i])
            i += 1
    return new_flags


def main():
    parser = argparse.ArgumentParser(description="Run the ablation study with widen-range parameter.")
    parser.add_argument("--prefix", type=str, required=True, help="Prefix for intermediate files (e.g., rosa-ex23-)")
    parser.add_argument("--clean", action="store_true", help="Clean up generated files")
    parser.add_argument("--plot-only", action="store_true", help="Plot results from existing data")
    parser.add_argument("--output-format", type=str, default="png", help="Output format for plots (e.g., png, pdf)")
    parser.add_argument(
        "--num-parallel", type=int, default=16, help="Number of parallel processes to use (default: 16)"
    )
    parser.add_argument(
        "--ablation-type",
        type=str,
        choices=["widen-range", "cost-model"],
        default="widen-range",
        help="Type of ablation study to perform (default: widen-range)",
    )
    args = parser.parse_args()

    original_prefix = args.prefix
    if not original_prefix.endswith("-"):
        original_prefix += "-"

    tmp_dir = "tmp"
    logs_dir = "logs"
    plots_dir = "plots"

    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    example_txt_path = os.path.join(tmp_dir, f"{original_prefix}example.txt")

    if args.clean:
        clean(tmp_dir, logs_dir, plots_dir)
        sys.exit(0)
    elif args.plot_only:
        if args.ablation_type == "widen-range":
            plot_ablation_results(tmp_dir, plots_dir, original_prefix, original_prefix, args.output_format)
        elif args.ablation_type == "cost-model":
            plot_ablation_results_cost_model(tmp_dir, plots_dir, original_prefix, original_prefix, args.output_format)
        sys.exit(0)
    else:
        if not os.path.exists(example_txt_path):
            generate_example_logged_cpp(tmp_dir, original_prefix, original_prefix)
            compile_example_logged_exe(tmp_dir, original_prefix)
            generate_example_txt(tmp_dir, original_prefix)

        if args.ablation_type == "widen-range":
            widen_ranges = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
            all_data = {}
            for X in widen_ranges:
                print(f"=== Running ablation study with widen-range={X} ===")
                remove_cache_dir()
                FPOPTFLAGS_BASE = FPOPTFLAGS_BASE_TEMPLATE.copy()
                for idx, flag in enumerate(FPOPTFLAGS_BASE):
                    if flag.startswith("--fpopt-log-path="):
                        FPOPTFLAGS_BASE[idx] = f"--fpopt-log-path={example_txt_path}"
                FPOPTFLAGS_BASE.extend(["-mllvm", f"--fpopt-widen-range={X}"])

                prefix_with_x = f"{original_prefix}abl-widen-range-{X}-"

                data = build_with_benchmark(
                    tmp_dir,
                    logs_dir,
                    plots_dir,
                    original_prefix,
                    prefix_with_x,
                    FPOPTFLAGS_BASE,
                    example_txt_path,
                    num_parallel=args.num_parallel,
                )

                all_data[X] = data

            ablation_data_file = os.path.join(tmp_dir, f"{original_prefix}ablation-widen-range.pkl")
            with open(ablation_data_file, "wb") as f:
                pickle.dump(all_data, f)
            print(f"Ablation data saved to {ablation_data_file}")

            plot_ablation_results(tmp_dir, plots_dir, original_prefix, original_prefix, args.output_format)

        if args.ablation_type == "cost-model":
            print("=== Running cost-model ablation study ===")
            remove_cache_dir()
            FPOPTFLAGS_WITH_CM = FPOPTFLAGS_BASE_TEMPLATE.copy()
            for idx, flag in enumerate(FPOPTFLAGS_WITH_CM):
                if flag.startswith("--fpopt-log-path="):
                    FPOPTFLAGS_WITH_CM[idx] = f"--fpopt-log-path={example_txt_path}"
            prefix_with_cm = f"{original_prefix}abl-with-cost-model-"

            data_with_cm = build_with_benchmark(
                tmp_dir,
                logs_dir,
                plots_dir,
                original_prefix,
                prefix_with_cm,
                FPOPTFLAGS_WITH_CM,
                example_txt_path,
                num_parallel=args.num_parallel,
            )

            remove_cache_dir()
            FPOPTFLAGS_NO_CM = remove_mllvm_flag(FPOPTFLAGS_WITH_CM, "--fpopt-cost-model-path=")
            prefix_without_cm = f"{original_prefix}abl-without-cost-model-"

            data_without_cm = build_with_benchmark(
                tmp_dir,
                logs_dir,
                plots_dir,
                original_prefix,
                prefix_without_cm,
                FPOPTFLAGS_NO_CM,
                example_txt_path,
                num_parallel=args.num_parallel,
            )

            all_data = {
                "with_cost_model": data_with_cm,
                "without_cost_model": data_without_cm,
            }

            ablation_data_file = os.path.join(tmp_dir, f"{original_prefix}ablation-cost-model.pkl")
            with open(ablation_data_file, "wb") as f:
                pickle.dump(all_data, f)
            print(f"Ablation data saved to {ablation_data_file}")

            plot_ablation_results_cost_model(tmp_dir, plots_dir, original_prefix, original_prefix, args.output_format)


if __name__ == "__main__":
    main()
