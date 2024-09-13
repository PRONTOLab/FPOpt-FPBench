import os
import sys
import re
import random
import numpy as np

num_samples_per_func = 100
default_regex = "ex\\d+"


def parse_bound(bound):
    if "/" in bound:
        numerator, denominator = map(float, bound.split("/"))
        return numerator / denominator
    return float(bound)


def parse_c_file(filepath, func_regex):
    with open(filepath, "r") as file:
        content = file.read()

    pattern = re.compile(rf"(?s)(// ## PRE(?:.*?\n)+?)\s*([\w\s\*]+?)\s+({func_regex})\s*\(([^)]*)\)")

    matches = pattern.findall(content)

    if not matches:
        exit(f"No functions found with the regex: {func_regex}")

    functions = []

    for comments, return_type, func_name, params in matches:
        param_comments = re.findall(r"// ## PRE (\w+):\s*([-+.\d/]+),\s*([-+.\d/]+)", comments)
        bounds = {
            name: {
                "min": parse_bound(min_val),
                "max": parse_bound(max_val),
            }
            for name, min_val, max_val in param_comments
        }
        params = [param.strip() for param in params.split(",") if param.strip()]
        functions.append((func_name, bounds, params, return_type.strip()))

    return functions


def create_driver_function(functions):
    driver_code = ["int main() {"]
    driver_code.append("    volatile double res;")

    for func_name, bounds, params, return_type in functions:
        print(f"Generating driver code for {func_name}")
        for _ in range(num_samples_per_func):
            call_params = []
            for param in params:
                param_tokens = param.strip().split()
                if len(param_tokens) >= 2:
                    param_name = param_tokens[-1]
                else:
                    exit(f"Cannot parse parameter: {param}")
                try:
                    min_val = bounds[param_name]["min"]
                    max_val = bounds[param_name]["max"]
                except KeyError:
                    exit(
                        f"WARNING: Bounds not found for {param_name} in function {func_name}, manually specify the bounds."
                    )
                random_value = np.random.uniform(min_val, max_val)
                call_params.append(str(random_value))
            driver_code.append(f"    res = {func_name}({', '.join(call_params)});")

    driver_code.append("    return 0;")
    driver_code.append("}")
    return "\n".join(driver_code)


def main():
    if len(sys.argv) < 2:
        exit("Usage: script.py <filepath> [function_regex]")

    filepath = sys.argv[1]
    func_regex = sys.argv[2] if len(sys.argv) > 2 else default_regex

    if len(sys.argv) <= 2:
        print(f"WARNING: No regex provided for target function names. Using default regex: {default_regex}")

    functions = parse_c_file(filepath, func_regex)
    driver_code = create_driver_function(functions)
    new_filepath = os.path.splitext(filepath)[0] + ".cpp"

    with open(filepath, "r") as original_file:
        original_content = original_file.read()

    with open(new_filepath, "w") as new_file:
        new_file.write(original_content)
        new_file.write("\n\n" + driver_code)

    print(f"Driver function appended to the new file: {new_filepath}")


if __name__ == "__main__":
    main()
