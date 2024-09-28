import subprocess
import time
import csv
import os
import struct
import numpy as np
import random

instructions = ["fneg", "fadd", "fsub", "fmul", "fdiv", "fcmp", "fptrunc", "fpext"]
functions = ["sin", "cos", "tan", "exp", "log", "sqrt", "expm1", "log1p", "cbrt", "pow", "fma", "fabs", "hypot"]

precisions = ["double", "float", "half"]
iterations = 100000000

precision_to_llvm_type = {
    "double": "double",
    "float": "float",
    "half": "half",
}

precision_to_intrinsic_suffix = {
    "double": "f64",
    "float": "f32",
    "half": "f16",
}

functions_with_intrinsics = {"sin", "cos", "exp", "log", "sqrt", "pow", "fabs", "fma"}


def float_to_llvm_hex(f, precision):
    if precision == "double":
        f_cast = np.float64(f)
        packed = struct.pack(">d", f_cast)
        [i] = struct.unpack(">Q", packed)
        hex_str = f"0x{i:016X}"
        return hex_str
    elif precision == "float":
        f_cast = np.float32(f)
        packed = struct.pack(">d", f_cast)
        [i] = struct.unpack(">Q", packed)
        hex_str = f"0x{i:016X}"
        return hex_str
    elif precision == "half":
        f_cast = np.float16(f)
        packed = struct.pack(">d", f_cast)
        [i] = struct.unpack(">Q", packed)
        hex_str = f"0x{i:016X}"
        return hex_str
    else:
        return str(f)


def generate_random_fp(precision):
    if precision == "double":
        f = random.uniform(-1e10, 1e10)
        dtype = np.float64
    elif precision == "float":
        f = random.uniform(-1e5, 1e5)
        dtype = np.float32
    elif precision == "half":
        f = random.uniform(-1e3, 1e3)
        dtype = np.float16
    else:
        f = random.uniform(-1e3, 1e3)
        dtype = np.float64

    f = dtype(f).item()
    return f


def generate_baseline_code(iterations):
    code = f"""
define i32 @main() {{
entry:
  %i = alloca i32
  store i32 0, i32* %i
  br label %loop

loop:
  %i_val = load i32, i32* %i
  %cond = icmp slt i32 %i_val, {iterations}
  br i1 %cond, label %body, label %exit

body:
  ; No operation inside the loop
  %i_next = add i32 %i_val, 1
  store i32 %i_next, i32* %i
  br label %loop

exit:
  ret i32 0
}}
"""
    return code


def generate_llvm_code(instruction, precision, iterations):
    llvm_type = precision_to_llvm_type[precision]
    code = ""
    if instruction in ["fadd", "fsub", "fmul", "fdiv"]:
        op_map = {"fadd": "fadd", "fsub": "fsub", "fmul": "fmul", "fdiv": "fdiv"}
        op = op_map[instruction]
        a = generate_random_fp(precision)
        b = generate_random_fp(precision)
        hex_a = float_to_llvm_hex(a, precision)
        hex_b = float_to_llvm_hex(b, precision)
        code = f"""
define i32 @main() {{
entry:
  %i = alloca i32
  %acc = alloca {llvm_type}
  store i32 0, i32* %i
  store {llvm_type} 0.0, {llvm_type}* %acc
  br label %loop

loop:
  %i_val = load i32, i32* %i
  %cond = icmp slt i32 %i_val, {iterations}
  br i1 %cond, label %body, label %exit

body:
  %result = {op} fast {llvm_type} {hex_a}, {hex_b}
  %acc_val = load {llvm_type}, {llvm_type}* %acc
  %new_acc = fadd fast {llvm_type} %acc_val, %result
  store {llvm_type} %new_acc, {llvm_type}* %acc
  %i_next = add i32 %i_val, 1
  store i32 %i_next, i32* %i
  br label %loop

exit:
  %final_acc = load {llvm_type}, {llvm_type}* %acc
  call void @use({llvm_type} %final_acc)
  ret i32 0
}}

define void @use({llvm_type} %val) {{
  ret void
}}
"""
    elif instruction == "fneg":
        a = generate_random_fp(precision)
        hex_a = float_to_llvm_hex(a, precision)
        code = f"""
define i32 @main() {{
entry:
  %i = alloca i32
  %acc = alloca {llvm_type}
  store i32 0, i32* %i
  store {llvm_type} 0.0, {llvm_type}* %acc
  br label %loop

loop:
  %i_val = load i32, i32* %i
  %cond = icmp slt i32 %i_val, {iterations}
  br i1 %cond, label %body, label %exit

body:
  %result = fneg fast {llvm_type} {hex_a}
  %acc_val = load {llvm_type}, {llvm_type}* %acc
  %new_acc = fadd fast {llvm_type} %acc_val, %result
  store {llvm_type} %new_acc, {llvm_type}* %acc
  %i_next = add i32 %i_val, 1
  store i32 %i_next, i32* %i
  br label %loop

exit:
  %final_acc = load {llvm_type}, {llvm_type}* %acc
  call void @use({llvm_type} %final_acc)
  ret i32 0
}}

define void @use({llvm_type} %val) {{
  ret void
}}
"""
    elif instruction == "fcmp":
        a = generate_random_fp(precision)
        b = generate_random_fp(precision)
        hex_a = float_to_llvm_hex(a, precision)
        hex_b = float_to_llvm_hex(b, precision)
        code = f"""
define i32 @main() {{
entry:
  %i = alloca i32
  %acc = alloca i1
  store i32 0, i32* %i
  store i1 false, i1* %acc
  br label %loop

loop:
  %i_val = load i32, i32* %i
  %cond = icmp slt i32 %i_val, {iterations}
  br i1 %cond, label %body, label %exit

body:
  %cmp = fcmp fast olt {llvm_type} {hex_a}, {hex_b}
  store i1 %cmp, i1* %acc
  %i_next = add i32 %i_val, 1
  store i32 %i_next, i32* %i
  br label %loop

exit:
  %final_cmp = load i1, i1* %acc
  call void @use(i1 %final_cmp)
  ret i32 0
}}

define void @use(i1 %val) {{
  ret void
}}
"""
    elif instruction == "fptrunc":
        if precision == "half":
            return ""
        src_type = llvm_type
        dst_type = {"double": "float", "float": "half"}[precision]
        a = generate_random_fp(precision)
        hex_a = float_to_llvm_hex(a, precision)
        code = f"""
define i32 @main() {{
entry:
  %i = alloca i32
  %acc = alloca {dst_type}
  store i32 0, i32* %i
  store {dst_type} 0.0, {dst_type}* %acc
  br label %loop

loop:
  %i_val = load i32, i32* %i
  %cond = icmp slt i32 %i_val, {iterations}
  br i1 %cond, label %body, label %exit

body:
  %result = fptrunc {src_type} {hex_a} to {dst_type}
  %acc_val = load {dst_type}, {dst_type}* %acc
  %new_acc = fadd fast {dst_type} %acc_val, %result
  store {dst_type} %new_acc, {dst_type}* %acc
  %i_next = add i32 %i_val, 1
  store i32 %i_next, i32* %i
  br label %loop

exit:
  %final_acc = load {dst_type}, {dst_type}* %acc
  call void @use({dst_type} %final_acc)
  ret i32 0
}}

define void @use({dst_type} %val) {{
  ret void
}}
"""
    elif instruction == "fpext":
        if precision == "double":
            return ""
        src_type = llvm_type
        dst_type = {"float": "double", "half": "float"}[precision]
        a = generate_random_fp(precision)
        hex_a = float_to_llvm_hex(a, precision)
        code = f"""
define i32 @main() {{
entry:
  %i = alloca i32
  %acc = alloca {dst_type}
  store i32 0, i32* %i
  store {dst_type} 0.0, {dst_type}* %acc
  br label %loop

loop:
  %i_val = load i32, i32* %i
  %cond = icmp slt i32 %i_val, {iterations}
  br i1 %cond, label %body, label %exit

body:
  %result = fpext {src_type} {hex_a} to {dst_type}
  %acc_val = load {dst_type}, {dst_type}* %acc
  %new_acc = fadd fast {dst_type} %acc_val, %result
  store {dst_type} %new_acc, {dst_type}* %acc
  %i_next = add i32 %i_val, 1
  store i32 %i_next, i32* %i
  br label %loop

exit:
  %final_acc = load {dst_type}, {dst_type}* %acc
  call void @use({dst_type} %final_acc)
  ret i32 0
}}

define void @use({dst_type} %val) {{
  ret void
}}
"""
    else:
        return ""
    return code


def generate_llvm_function_call(function_name, precision, iterations):
    llvm_type = precision_to_llvm_type[precision]
    intrinsic_suffix = precision_to_intrinsic_suffix[precision]
    code = ""
    a = generate_random_fp(precision)
    hex_a = float_to_llvm_hex(a, precision)
    if function_name in {"pow", "hypot", "fma"}:
        b = generate_random_fp(precision)
        hex_b = float_to_llvm_hex(b, precision)
    if function_name == "pow":
        function_intrinsic = f"llvm.pow.{intrinsic_suffix}"
        code += f"declare {llvm_type} @{function_intrinsic}({llvm_type}, {llvm_type})\n"
        function_call = f"call fast {llvm_type} @{function_intrinsic}({llvm_type} {hex_a}, {llvm_type} {hex_b})"
    elif function_name == "hypot":
        code += f"declare {llvm_type} @{function_name}({llvm_type}, {llvm_type})\n"
        function_call = f"call fast {llvm_type} @{function_name}({llvm_type} {hex_a}, {llvm_type} {hex_b})"
    elif function_name == "fma":
        b = generate_random_fp(precision)
        c = generate_random_fp(precision)
        hex_b = float_to_llvm_hex(b, precision)
        hex_c = float_to_llvm_hex(c, precision)
        function_intrinsic = f"llvm.fma.{intrinsic_suffix}"
        code += f"declare {llvm_type} @{function_intrinsic}({llvm_type}, {llvm_type}, {llvm_type})\n"
        function_call = f"call fast {llvm_type} @{function_intrinsic}({llvm_type} {hex_a}, {llvm_type} {hex_b}, {llvm_type} {hex_c})"
    else:
        if function_name in functions_with_intrinsics:
            function_intrinsic = f"llvm.{function_name}.{intrinsic_suffix}"
            code += f"declare {llvm_type} @{function_intrinsic}({llvm_type})\n"
            function_call = f"call fast {llvm_type} @{function_intrinsic}({llvm_type} {hex_a})"
        else:
            code += f"declare {llvm_type} @{function_name}({llvm_type})\n"
            function_call = f"call fast {llvm_type} @{function_name}({llvm_type} {hex_a})"

    code += f"""
define i32 @main() {{
entry:
  %i = alloca i32
  %acc = alloca {llvm_type}
  store i32 0, i32* %i
  store {llvm_type} 0.0, {llvm_type}* %acc
  br label %loop

loop:
  %i_val = load i32, i32* %i
  %cond = icmp slt i32 %i_val, {iterations}
  br i1 %cond, label %body, label %exit

body:
  %result = {function_call}
  %acc_val = load {llvm_type}, {llvm_type}* %acc
  %new_acc = fadd fast {llvm_type} %acc_val, %result
  store {llvm_type} %new_acc, {llvm_type}* %acc
  %i_next = add i32 %i_val, 1
  store i32 %i_next, i32* %i
  br label %loop

exit:
  %final_acc = load {llvm_type}, {llvm_type}* %acc
  call void @use({llvm_type} %final_acc)
  ret i32 0
}}

define void @use({llvm_type} %val) {{
  ret void
}}
"""
    return code


def compile_and_run(filename, executable):
    compile_cmd = ["clang", "-O0", filename, "-lm", "-o", executable]
    subprocess.run(compile_cmd, check=True)
    start_time = time.perf_counter()
    subprocess.run([f"./{executable}"], check=True)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    return elapsed_time


csv_file = "benchmark_results.csv"
baseline_time = 0.0

with open(csv_file, "w", newline="") as csvfile:
    fieldnames = ["instruction", "precision", "cost"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    print("Benchmarking baseline")
    llvm_code = generate_baseline_code(iterations)
    filename = "baseline.ll"
    with open(filename, "w") as f:
        f.write(llvm_code)

    executable = "baseline_executable"
    try:
        baseline_time = compile_and_run(filename, executable)
        print(f"Baseline time: {baseline_time:.6f} seconds")
    except subprocess.CalledProcessError as e:
        print(f"Error compiling or running baseline: {e}")
        baseline_time = 0.0

    if os.path.exists(executable):
        os.remove(executable)

    for precision in precisions:
        for instr in instructions:
            llvm_code = generate_llvm_code(instr, precision, iterations)
            if not llvm_code.strip():
                continue
            filename = f"{instr}_{precision}.ll"
            with open(filename, "w") as f:
                f.write(llvm_code)
            executable = f"{instr}_{precision}"
            try:
                elapsed_time = compile_and_run(filename, executable)
                adjusted_time = elapsed_time - baseline_time
                adjusted_time = max(adjusted_time, 0.0)
                writer.writerow({"instruction": instr, "precision": precision, "cost": adjusted_time})
                print(f"Benchmarked instruction {instr} with precision {precision}: {adjusted_time:.6f} seconds")
            except subprocess.CalledProcessError as e:
                print(f"Error compiling or running instruction {instr} for {precision}: {e}")
            if os.path.exists(executable):
                os.remove(executable)

        for func in functions:
            llvm_code = generate_llvm_function_call(func, precision, iterations)
            if not llvm_code.strip():
                continue
            filename = f"{func}_{precision}.ll"
            with open(filename, "w") as f:
                f.write(llvm_code)
            executable = f"{func}_{precision}"
            try:
                elapsed_time = compile_and_run(filename, executable)
                adjusted_time = elapsed_time - baseline_time
                adjusted_time = max(adjusted_time, 0.0)
                writer.writerow({"instruction": func, "precision": precision, "cost": adjusted_time})
                print(f"Benchmarked function {func} with precision {precision}: {adjusted_time:.6f} seconds")
            except subprocess.CalledProcessError as e:
                print(f"Error compiling or running function {func} for {precision}: {e}")
            if os.path.exists(executable):
                os.remove(executable)

print(f"Benchmarking completed. Results are saved in '{csv_file}'.")
