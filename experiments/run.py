#!/usr/bin/env python3

import os
import subprocess
import sys
import shutil
import time
import re
import matplotlib.pyplot as plt
from statistics import mean

ENZYME_PATH = "/home/brant/sync/Enzyme/build/Enzyme/ClangEnzyme-15.so"
LLVM_PATH = "/home/brant/llvms/llvm15/build/bin"
CXX = os.path.join(LLVM_PATH, "clang++")

CXXFLAGS = [
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
    "-O3",
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
NUM_RUNS = 1000
DRIVER_NUM_SAMPLES = 1000


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


def clean():
    print("=== Cleaning up generated files ===")
    for exe in EXE:
        if os.path.exists(exe):
            os.remove(exe)
            print(f"Removed {exe}")
    for cpp in ["example.cpp", "example-logged.cpp"]:
        if os.path.exists(cpp):
            os.remove(cpp)
            print(f"Removed {cpp}")
    for file in ["example.txt", "output.txt", "runtime_plot.png", "compile_fpopt.log"]:
        if os.path.exists(file):
            os.remove(file)
            print(f"Removed {file}")


def generate_example_cpp():
    script = "fpopt-original-driver-generator.py"
    output = "example.cpp"
    run_command(["python3", script, SRC, "example", str(DRIVER_NUM_SAMPLES)], f"Generating {output} from {SRC}")


def generate_example_logged_cpp():
    script = "fpopt-logged-driver-generator.py"
    output = "example-logged.cpp"
    run_command(["python3", script, SRC, "example", str(DRIVER_NUM_SAMPLES)], f"Generating {output} from {SRC}")


def compile_example_exe():
    source = "example.cpp"
    output = "example.exe"
    cmd = [CXX, "-Wall", "-O3", source] + CXXFLAGS + ["-o", output]
    run_command(cmd, f"Compiling {output}")


def compile_example_logged_exe():
    sources = ["example-logged.cpp", LOGGER]
    output = "example-logged.exe"
    cmd = [CXX, "-Wall", "-O3"] + sources + CXXFLAGS + ["-o", output]
    run_command(cmd, f"Compiling {output}")


def generate_example_txt():
    exe = "./example-logged.exe"
    output = "example.txt"
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


def compile_example_fpopt_exe(fpoptflags, output="example-fpopt.exe", verbose=True):
    source = "example.cpp"
    cmd = [CXX, source] + CXXFLAGS + fpoptflags + ["-o", output]
    if output == "example-fpopt.exe":
        run_command(
            cmd,
            f"Compiling {output} with FPOPTFLAGS",
            capture_output=True,
            output_file="compile_fpopt.log",
            verbose=verbose,
        )
    else:
        run_command(
            cmd,
            f"Compiling {output} with FPOPTFLAGS",
            verbose=verbose,
        )


def parse_critical_comp_costs(log_path="compile_fpopt.log"):
    print(f"=== Parsing critical computation costs from {log_path} ===")
    if not os.path.exists(log_path):
        print(f"Log file {log_path} does not exist.")
        sys.exit(1)
    with open(log_path, "r") as f:
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


def measure_runtime(executable, num_runs=NUM_RUNS):
    print(f"=== Measuring runtime for {executable} ===")
    runtimes = []
    for i in range(1, num_runs + 1):
        # print(f"Run {i}/{num_runs}")
        start_time = time.perf_counter()
        try:
            subprocess.check_call([f"./{executable}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"Error running {executable} on run {i}")
            sys.exit(e.returncode)
        end_time = time.perf_counter()
        runtime = end_time - start_time
        runtimes.append(runtime)
        # print(f"Run {i}: {runtime:.4f} seconds")
    average_runtime = mean(runtimes)
    print(f"Average runtime for {executable}: {average_runtime:.4f} seconds")
    return average_runtime


def plot_results(budgets, runtimes, output_file="runtime_plot.png"):
    print(f"=== Plotting results to {output_file} ===")
    plt.figure(figsize=(10, 6))
    plt.plot(budgets, runtimes, marker="o", linestyle="-")
    plt.xlabel("Computation Cost Budget")
    plt.ylabel("Average Runtime (seconds)")
    plt.title("Computation Cost Budget vs. Average Runtime")
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()
    print(f"Plot saved to {output_file}")


def build_all():
    generate_example_cpp()
    generate_example_logged_cpp()
    compile_example_exe()
    compile_example_logged_exe()
    generate_example_txt()
    compile_example_fpopt_exe(FPOPTFLAGS_BASE, output="example-fpopt.exe")
    print("=== Initial build process completed successfully ===")


def benchmark():
    if not os.path.exists("example-fpopt.exe"):
        print("example-fpopt.exe not found. Please build it first.")
        sys.exit(1)

    print("=== Running example-fpopt.exe and saving output to output.txt ===")
    run_command(
        ["./example-fpopt.exe"],
        "Running example-fpopt.exe and saving output to output.txt",
        capture_output=True,
        output_file="output.txt",
    )

    costs = parse_critical_comp_costs("compile_fpopt.log")

    budgets = []
    runtimes = []

    for cost in costs:
        print(f"\n=== Processing computation cost budget: {cost} ===")
        fpoptflags = []
        for flag in FPOPTFLAGS_BASE:
            if flag.startswith("--fpopt-comp-cost-budget="):
                fpoptflags.append(f"--fpopt-comp-cost-budget={cost}")
            else:
                fpoptflags.append(flag)

        output_binary = f"example-fpopt-{cost}.exe"

        compile_example_fpopt_exe(fpoptflags, output=output_binary, verbose=False)

        avg_runtime = measure_runtime(output_binary, NUM_RUNS)

        budgets.append(cost)
        runtimes.append(avg_runtime)

    plot_results(budgets, runtimes)


def build_with_benchmark():
    build_all()
    benchmark()


def main():
    if len(sys.argv) > 1:
        target = sys.argv[1]
        if target == "clean":
            clean()
            sys.exit(0)
        elif target == "build":
            build_all()
            sys.exit(0)
        elif target == "benchmark":
            benchmark()
            sys.exit(0)
        elif target == "all":
            build_with_benchmark()
            sys.exit(0)
        else:
            print(f"Unknown target: {target}")
            print("Usage: build_fpopt.py [build|clean|benchmark|all]")
            sys.exit(1)
    else:
        build_with_benchmark()


if __name__ == "__main__":
    main()
