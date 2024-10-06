import os
import sys
import re
import numpy as np

DEFAULT_NUM_SAMPLES = 10000
default_regex = "ex\\d+"

np.random.seed(42)


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


def create_baseline_functions(functions):
    baseline_code = []
    for func_name, bounds, params, return_type in functions:
        param_list = ", ".join(params)
        baseline_func_name = f"baseline_{func_name}"
        baseline_code.append(f"__attribute__((noinline))\n{return_type} {baseline_func_name}({param_list}) {{")
        baseline_code.append("    return 42.0;")
        baseline_code.append("}")
        baseline_code.append("")
    return "\n".join(baseline_code)


def create_baseline_driver_function(functions, num_samples_per_func):
    driver_code = [
        "#include <iostream>",
        "#include <random>",
        "",
        "int main() {",
        "    std::mt19937 gen(42);",
        "",
    ]

    for func_name, bounds, params, return_type in functions:
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
            dist_name = f"{func_name}_{param_name}_dist"
            driver_code.append(f"    std::uniform_real_distribution<double> {dist_name}({min_val}, {max_val});")
    driver_code.append("")

    driver_code.append("    double res = 0.;")
    driver_code.append("")

    for func_name, bounds, params, return_type in functions:
        baseline_func_name = f"baseline_{func_name}"
        driver_code.append(f"    for (int i = 0; i < {num_samples_per_func}; ++i) {{")

        call_params = []
        for param in params:
            param_tokens = param.strip().split()
            if len(param_tokens) >= 2:
                param_name = param_tokens[-1]
            else:
                exit(f"Cannot parse parameter: {param}")
            dist_name = f"{func_name}_{param_name}_dist"
            param_value = f"{dist_name}(gen)"
            call_params.append(param_value)

        driver_code.append(f"        res += {baseline_func_name}({', '.join(call_params)});")
        driver_code.append("    }")
        driver_code.append("")

    driver_code.append('    std::cout << "Sum: " << res << std::endl;')
    driver_code.append("    return 0;")
    driver_code.append("}")
    return "\n".join(driver_code)


def main():
    if len(sys.argv) < 2:
        exit("Usage: script.py <filepath> [function_regex] [num_samples_per_func (default: 10000)]")

    filepath = sys.argv[1]
    func_regex = sys.argv[2] if len(sys.argv) > 2 else default_regex
    num_samples_per_func = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_NUM_SAMPLES

    if len(sys.argv) <= 2:
        print(f"WARNING: No regex provided for target function names. Using default regex: {default_regex}")

    functions = parse_c_file(filepath, func_regex)
    baseline_functions_code = create_baseline_functions(functions)
    driver_code = create_baseline_driver_function(functions, num_samples_per_func)
    new_filepath = os.path.splitext(filepath)[0] + "-baseline.cpp"

    with open(new_filepath, "w") as new_file:
        new_file.write("#include <cmath>\n")
        new_file.write("\n\n")
        new_file.write(baseline_functions_code)
        new_file.write("\n\n")
        new_file.write(driver_code)

    print(f"Baseline code written to the new file: {new_filepath}")


if __name__ == "__main__":
    main()
