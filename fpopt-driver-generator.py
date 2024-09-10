import sys
import re
import random
import numpy as np

num_samples_per_func = 100


def parse_bound(bound):
    if "/" in bound:
        numerator, denominator = map(float, bound.split("/"))
        return numerator / denominator
    return float(bound)


def parse_c_file(filepath):
    with open(filepath, "r") as file:
        content = file.read()

    pattern = re.compile(r"(?s)(// ## PRE [\s\S]+?)double (ex\d+)\(([^)]+)\)")
    matches = pattern.findall(content)

    functions = []

    for comments, func_name, params in matches:
        # print(f"Found function: {func_name}")
        # print(f"Comments: {comments}")
        # print(f"Parameters: {params}")
        param_comments = re.findall(r"// ## PRE (\w+): ([-\d./]+), ([-\d./]+), (#t|#f), (#t|#f)", comments)
        # print(param_comments)
        bounds = {
            name: {
                "min": parse_bound(min_val),
                "max": parse_bound(max_val),
                "min_inc": True if min_inc == "#t" else False,
                "max_inc": True if max_inc == "#t" else False,
            }
            for name, min_val, max_val, min_inc, max_inc in param_comments
        }
        params = [param.strip() for param in params.split(",")]
        # print(param_details)
        functions.append((func_name, bounds, params))

    # print(functions)
    return functions


def create_driver_function(functions):
    driver_code = ["void fpopt_logger_driver() {"]

    for func_name, bounds, params in functions:
        print(f"Generating driver code for {func_name}")
        for _ in range(num_samples_per_func):
            call_params = []
            for param in params:
                # print(param)
                _, param_name = param.split()
                # print(param_name)
                try:
                    min_val = bounds[param_name]["min"]
                    max_val = bounds[param_name]["max"]
                except KeyError:
                    exit(
                        f"WARNING: Bounds not found for {param_name} in function {func_name}, manually specify the bounds."
                    )
                random_value = np.random.uniform(min_val, max_val)
                call_params.append(str(random_value))
            driver_code.append(f"    {func_name}({', '.join(call_params)});")

        # Append the function call to the driver function

    driver_code.append("}")
    return "\n".join(driver_code)


def main():
    filepath = sys.argv[1] if len(sys.argv) > 1 else exit("Need a path!")

    functions = parse_c_file(filepath)
    driver_code = create_driver_function(functions)
    with open(filepath, "a") as file:
        file.write("\n\n" + driver_code)
    print("Driver function appended to the file.")


if __name__ == "__main__":
    main()
