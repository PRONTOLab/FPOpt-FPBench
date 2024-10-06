import os
import re
import subprocess
import glob


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


def main():
    benchmark_dir = "../benchmarks"
    racket_script = "../export.rkt"
    fpcore_files = glob.glob(os.path.join(benchmark_dir, "*.fpcore"))

    if not fpcore_files:
        print("No .fpcore files found in the benchmarks directory.")
        return

    for fpcore_file in fpcore_files:
        filename = os.path.basename(fpcore_file)
        base_name = os.path.splitext(filename)[0]
        print(f"Processing .fpcore file: {filename}")

        c_filename = f"{base_name}.fpcore.c"
        c_filepath = os.path.join(benchmark_dir, c_filename)
        print(f"Generating {c_filename} using Racket script...")
        try:
            subprocess.check_call(["racket", racket_script, fpcore_file, c_filepath])
        except subprocess.CalledProcessError as e:
            print(f"Error running export.rkt on {filename}")
            continue

        if not os.path.exists(c_filepath):
            print(f"Generated .c file {c_filename} not found.")
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
            return_type = func["return_type"]
            params = func["params"]
            comments = func["comments"]

            print(f"Processing function: {func_name}")
            func_body_lines = func["func_body"].split("\n")
            
            func_signature_line = f"__attribute__((noinline))\n{return_type} example({params}) {{"
            
            func_body_lines[0] = func_signature_line
            func_code = comments + "\n" + "\n".join(func_body_lines)
            includes = "#include <math.h>\n#include <stdint.h>\n#define TRUE 1\n#define FALSE 0\n"
            example_c_content = includes + "\n" + func_code
            with open("example.c", "w") as f:
                f.write(example_c_content)
            try:
                # print("Target func: ", example_c_content)
                subprocess.check_call(["python3", "run.py"])
                plot_filename = f"runtime_plot_{base_name}_{func_name}.png"
                if os.path.exists("runtime_plot.png"):
                    os.rename("runtime_plot.png", plot_filename)
                    print(f"Plot saved to {plot_filename}")
                else:
                    print("Plot not found after running run.py")
            except subprocess.CalledProcessError as e:
                print(f"Error running run.py for function {func_name} in file {c_filename}")
                continue

    print("Processing completed.")


if __name__ == "__main__":
    main()
