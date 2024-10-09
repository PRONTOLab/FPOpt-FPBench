import os
import sys
import re
import random
import numpy as np

DEFAULT_NUM_SAMPLES = 100000
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


def create_driver_function(functions, num_samples_per_func):
    driver_code = [
        "#include <iostream>",
        "#include <random>",
        "",
        "int main() {",
        "    std::mt19937 gen(42);",
        "",
    ]
    driver_code.append("    initializeLogger();")

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

        driver_code.append(
            f"    res += __enzyme_autodiff<{return_type}>((void *) {func_name}, {', '.join(call_params)});"
        )
        driver_code.append("    }")
        driver_code.append("")

    driver_code.append('    std::cout << "Sum: " << res << std::endl;')
    driver_code.append("    printLogger();")
    driver_code.append("    destroyLogger();")
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
    driver_code = create_driver_function(functions, num_samples_per_func)
    new_filepath = os.path.splitext(filepath)[0] + "-logged.cpp"

    with open(filepath, "r") as original_file:
        original_content = original_file.read()

    code_to_insert = """#include "fp-logger.hpp"

void thisIsNeverCalledAndJustForTheLinker() {
  enzymeLogError("", 0.0);
  enzymeLogGrad("", 0.0);
  enzymeLogValue("", 0.0, 2, nullptr);
}

int enzyme_dup;
int enzyme_dupnoneed;
int enzyme_out;
int enzyme_const;

template <typename return_type, typename... T>
return_type __enzyme_autodiff(void *, T...);"""

    with open(new_filepath, "w") as new_file:
        new_file.write(original_content)
        new_file.write("\n\n" + code_to_insert + "\n\n" + driver_code)

    print(f"Driver function appended to the new file: {new_filepath}")


if __name__ == "__main__":
    main()
