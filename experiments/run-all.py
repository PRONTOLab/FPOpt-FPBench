import os
import re
import subprocess
import glob
import multiprocessing
import argparse


def extract_functions_from_c_file(content, func_regex="ex\\d+"):
    functions = []
    lines = content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        func_def_pattern = re.compile(rf"^\s*(.*?)\s+({func_regex})\s*\((.*?)\)\s*\{{\s*$")
        match = func_def_pattern.match(line)
        if match:
            return_type = match.group(1).strip()
            func_name = match.group(2)
            params = match.group(3).strip()
            comments = []
            j = i - 1
            while j >= 0:
                prev_line = lines[j]
                if prev_line.strip().startswith("//"):
                    comments.insert(0, prev_line)
                    j -= 1
                elif prev_line.strip() == "":
                    j -= 1
                else:
                    break
            func_body_lines = [line]
            brace_level = line.count("{") - line.count("}")
            i += 1
            while i < len(lines) and brace_level > 0:
                func_line = lines[i]
                func_body_lines.append(func_line)
                brace_level += func_line.count("{")
                brace_level -= func_line.count("}")
                i += 1
            func_body = "\n".join(func_body_lines)
            comments_str = "\n".join(comments)
            functions.append(
                {
                    "comments": comments_str,
                    "return_type": return_type,
                    "func_name": func_name,
                    "params": params,
                    "func_body": func_body,
                }
            )
        else:
            i += 1
    return functions


def process_function_task(func, base_name, plot_only):
    func_name = func["func_name"]
    return_type = func["return_type"]
    params = func["params"]
    comments = func["comments"]
    print(f"Processing function: {func_name}")

    prefix = f"{base_name}-{func_name}-"

    example_c_filename = f"{prefix}example.c"
    example_c_filepath = os.path.join("tmp", example_c_filename)

    func_body_lines = func["func_body"].split("\n")

    func_signature_line = f"__attribute__((noinline))\n{return_type} example({params}) {{"

    func_body_lines[0] = func_signature_line
    func_code = comments + "\n" + "\n".join(func_body_lines)
    includes = "#include <math.h>\n#include <stdint.h>\n#define TRUE 1\n#define FALSE 0\n"
    example_c_content = includes + "\n" + func_code
    with open(example_c_filepath, "w") as f:
        f.write(example_c_content)
    
    base_command = ["python3", "run.py", "--prefix", prefix]
    
    if plot_only:
        base_command.append("--plot-only")
    
    try:
        subprocess.check_call(base_command)
    except subprocess.CalledProcessError:
        print(f"Error running run.py for function {func_name} in base {base_name}. Retrying with --disable-preopt.")
        retry_command = ["python3", "run.py", "--prefix", prefix, "--disable-preopt"]
        if plot_only:
            retry_command.append("--plot-only")
        try:
            subprocess.check_call(retry_command)
        except subprocess.CalledProcessError:
            print(f"Error running run.py with --disable-preopt for function {func_name} in base {base_name}")


def main():
    parser = argparse.ArgumentParser(description="Process functions from .fpcore files.")
    parser.add_argument("-j", type=int, help="Number of parallel tasks", default=1)
    parser.add_argument("--regen", action="store_true", help="Force regeneration of .c files")
    parser.add_argument("--plot-only", action="store_true", help="Only plot the results")
    args = parser.parse_args()

    num_parallel_tasks = args.j
    force_regen = args.regen
    plot_only = args.plot_only

    source_dir = "../benchmarks"
    exported_dir = "exported"
    racket_script = "../export.rkt"
    fpcore_files = glob.glob(os.path.join(source_dir, "*.fpcore"))

    if not fpcore_files:
        print("No .fpcore files found in the benchmarks directory.")
        return

    os.makedirs("tmp", exist_ok=True)
    os.makedirs("exported", exist_ok=True)

    tasks = []

    for fpcore_file in fpcore_files:
        filename = os.path.basename(fpcore_file)
        base_name = os.path.splitext(filename)[0]
        c_filename = f"{base_name}.fpcore.c"
        c_filepath = os.path.join(exported_dir, c_filename)

        if not force_regen and os.path.exists(c_filepath):
            print(f"{c_filename} already exists. Skipping generation.")
        else:
            print(f"Generating {c_filename} using Racket script...")
            try:
                print("Running command: ", " ".join(["racket", racket_script, fpcore_file, c_filepath]))
                subprocess.check_call(["racket", racket_script, fpcore_file, c_filepath])
            except subprocess.CalledProcessError:
                print(f"Error running export.rkt on {filename}")
                continue

        print(f"Processing generated .c file: {c_filename}")
        with open(c_filepath, "r") as f:
            content = f.read()
        functions = extract_functions_from_c_file(content)
        if not functions:
            print(f"No ex functions found in {c_filename}")
            continue
        for func in functions:
            func_name = func["func_name"]
            print(f"Found function: {func_name}")
            task = (func, base_name, plot_only)
            tasks.append(task)

    if tasks:
        if num_parallel_tasks == 1:
            for task in tasks:
                process_function_task(*task)
        else:
            with multiprocessing.Pool(num_parallel_tasks) as pool:
                pool.starmap(process_function_task, tasks)
    else:
        print("No functions to process.")

    print("Processing completed.")


if __name__ == "__main__":
    main()
