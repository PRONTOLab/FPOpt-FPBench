#!/usr/bin/env python3

import os
import subprocess
import sys
import shutil

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

FPOPTFLAGS = [
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


def run_command(command, description):
    print(f"=== {description} ===")
    print("Running:", " ".join(command))
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as e:
        print(f"Error during: {description}")
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


def generate_example_cpp():
    script = "fpopt-original-driver-generator.py"
    output = "example.cpp"
    run_command(["python3", script, SRC, "example"], f"Generating {output} from {SRC}")


def generate_example_logged_cpp():
    script = "fpopt-logged-driver-generator.py"
    output = "example-logged.cpp"
    run_command(["python3", script, SRC, "example"], f"Generating {output} from {SRC}")


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


def compile_example_fpopt_exe():
    source = "example.cpp"
    output = "example-fpopt.exe"
    cmd = [CXX, source] + CXXFLAGS + FPOPTFLAGS + ["-o", output]
    run_command(cmd, f"Compiling {output}")


def build_all():
    generate_example_cpp()
    generate_example_logged_cpp()
    compile_example_exe()
    compile_example_logged_exe()
    generate_example_txt()
    compile_example_fpopt_exe()
    print("=== Build process completed successfully ===")


def main():
    if len(sys.argv) > 1:
        target = sys.argv[1]
        if target == "clean":
            clean()
            sys.exit(0)
        elif target == "build":
            build_all()
            sys.exit(0)
        else:
            print(f"Unknown target: {target}")
            print("Usage: build_fpopt.py [build|clean]")
            sys.exit(1)
    else:
        build_all()


if __name__ == "__main__":
    main()
