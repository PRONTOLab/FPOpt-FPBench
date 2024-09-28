import subprocess
import time
import csv
import os
import struct
import numpy as np
import random

random.seed(42)

instructions = ["fneg", "fadd", "fsub", "fmul", "fdiv", "fcmp", "fptrunc", "fpext"]
functions = ["sin", "cos", "tan", "exp", "log", "sqrt", "expm1", "log1p", "cbrt", "pow", "fma", "fabs", "hypot"]

precisions = ["double", "float", "half"]
iterations = 1000000000
unrolled = 100

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
    random_fps = [generate_random_fp("double") for _ in range(unrolled)]
    hex_fps = [float_to_llvm_hex(f, "double") for f in random_fps]
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
        random_pairs = [(generate_random_fp(precision), generate_random_fp(precision)) for _ in range(unrolled)]
        hex_pairs = [(float_to_llvm_hex(a, precision), float_to_llvm_hex(b, precision)) for a, b in random_pairs]
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
  %acc_val0 = load {llvm_type}, {llvm_type}* %acc
"""

        for idx, (hex_a, hex_b) in enumerate(hex_pairs):
            code += f"  %result{idx} = {op} fast {llvm_type} {hex_a}, {hex_b}\n"
            code += f"  %acc_val{idx+1} = fadd fast {llvm_type} %acc_val{idx}, %result{idx}\n"

        code += f"""
  store {llvm_type} %acc_val{len(hex_pairs)}, {llvm_type}* %acc
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
        random_fps = [generate_random_fp(precision) for _ in range(unrolled)]
        hex_fps = [float_to_llvm_hex(f, precision) for f in random_fps]
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
  %acc_val0 = load {llvm_type}, {llvm_type}* %acc
"""

        for idx, hex_a in enumerate(hex_fps):
            code += f"  %result{idx} = fneg fast {llvm_type} {hex_a}\n"
            code += f"  %acc_val{idx+1} = fadd fast {llvm_type} %acc_val{idx}, %result{idx}\n"

        code += f"""
  store {llvm_type} %acc_val{len(hex_fps)}, {llvm_type}* %acc
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
        code = f"""
define i32 @main() {{
entry:
  %i = alloca i32
  %acc = alloca i32
  store i32 0, i32* %i
  store i32 0, i32* %acc
  br label %loop

loop:
  %i_val = load i32, i32* %i
  %cond = icmp slt i32 %i_val, {iterations}
  br i1 %cond, label %body, label %exit

body:
"""

        for idx in range(unrolled):
            a = generate_random_fp(precision)
            b = generate_random_fp(precision)
            hex_a = float_to_llvm_hex(a, precision)
            hex_b = float_to_llvm_hex(b, precision)
            code += f"  %cmp{idx} = fcmp fast olt {llvm_type} {hex_a}, {hex_b}\n"
            code += f"  %cmp_int{idx} = zext i1 %cmp{idx} to i32\n"

        code += f"  %acc_val0 = load i32, i32* %acc\n"

        for idx in range(unrolled):
            code += f"  %acc_val{idx+1} = add i32 %acc_val{idx}, %cmp_int{idx}\n"

        code += f"""
  store i32 %acc_val{unrolled}, i32* %acc
  %i_next = add i32 %i_val, 1
  store i32 %i_next, i32* %i
  br label %loop

exit:
  %final_acc = load i32, i32* %acc
  call void @use_i32(i32 %final_acc)
  ret i32 0
}}

define void @use_i32(i32 %val) {{
  ret void
}}
"""
    elif instruction == "fptrunc":
        if precision == "half":
            return ""
        src_type = llvm_type
        dst_type = {"double": "float", "float": "half"}[precision]
        random_fps = [generate_random_fp(precision) for _ in range(unrolled)]
        hex_fps = [float_to_llvm_hex(f, precision) for f in random_fps]
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
  %acc_val0 = load {dst_type}, {dst_type}* %acc
"""

        for idx, hex_a in enumerate(hex_fps):
            code += f"  %result{idx} = fptrunc {src_type} {hex_a} to {dst_type}\n"
            code += f"  %acc_val{idx+1} = fadd fast {dst_type} %acc_val{idx}, %result{idx}\n"

        code += f"""
  store {dst_type} %acc_val{len(hex_fps)}, {dst_type}* %acc
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
        random_fps = [generate_random_fp(precision) for _ in range(unrolled)]
        hex_fps = [float_to_llvm_hex(f, precision) for f in random_fps]
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
  %acc_val0 = load {dst_type}, {dst_type}* %acc
"""

        for idx, hex_a in enumerate(hex_fps):
            code += f"  %result{idx} = fpext {src_type} {hex_a} to {dst_type}\n"
            code += f"  %acc_val{idx+1} = fadd fast {dst_type} %acc_val{idx}, %result{idx}\n"

        code += f"""
  store {dst_type} %acc_val{len(hex_fps)}, {dst_type}* %acc
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
    if function_name == "pow":
        function_intrinsic = f"llvm.pow.{intrinsic_suffix}"
        code += f"declare {llvm_type} @{function_intrinsic}({llvm_type}, {llvm_type})\n"
        function_call_template = (
            f"call fast {llvm_type} @{function_intrinsic}({llvm_type} {{arg1}}, {llvm_type} {{arg2}})"
        )
    elif function_name == "fma":
        function_intrinsic = f"llvm.fma.{intrinsic_suffix}"
        code += f"declare {llvm_type} @{function_intrinsic}({llvm_type}, {llvm_type}, {llvm_type})\n"
        function_call_template = f"call fast {llvm_type} @{function_intrinsic}({llvm_type} {{arg1}}, {llvm_type} {{arg2}}, {llvm_type} {{arg3}})"
    elif function_name in functions_with_intrinsics:
        function_intrinsic = f"llvm.{function_name}.{intrinsic_suffix}"
        code += f"declare {llvm_type} @{function_intrinsic}({llvm_type})\n"
        function_call_template = f"call fast {llvm_type} @{function_intrinsic}({llvm_type} {{arg1}})"
    else:
        code += f"declare {llvm_type} @{function_name}({llvm_type})\n"
        function_call_template = f"call fast {llvm_type} @{function_name}({llvm_type} {{arg1}})"
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
  %acc_val0 = load {llvm_type}, {llvm_type}* %acc
"""
    for idx in range(unrolled):
        if function_name == "pow":
            a = generate_random_fp(precision)
            b = generate_random_fp(precision)
            hex_a = float_to_llvm_hex(a, precision)
            hex_b = float_to_llvm_hex(b, precision)
            function_call = function_call_template.format(arg1=hex_a, arg2=hex_b)
        elif function_name == "hypot":
            a = generate_random_fp(precision)
            b = generate_random_fp(precision)
            hex_a = float_to_llvm_hex(a, precision)
            hex_b = float_to_llvm_hex(b, precision)
            function_call = function_call_template.format(arg1=hex_a, arg2=hex_b)
        elif function_name == "fma":
            a = generate_random_fp(precision)
            b = generate_random_fp(precision)
            c = generate_random_fp(precision)
            hex_a = float_to_llvm_hex(a, precision)
            hex_b = float_to_llvm_hex(b, precision)
            hex_c = float_to_llvm_hex(c, precision)
            function_call = function_call_template.format(arg1=hex_a, arg2=hex_b, arg3=hex_c)
        else:
            a = generate_random_fp(precision)
            hex_a = float_to_llvm_hex(a, precision)
            function_call = function_call_template.format(arg1=hex_a)

        code += f"  %result{idx} = {function_call}\n"
        code += f"  %acc_val{idx+1} = fadd fast {llvm_type} %acc_val{idx}, %result{idx}\n"

    code += f"""
  store {llvm_type} %acc_val{unrolled}, {llvm_type}* %acc
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


def compile_and_run(ll_filename, executable):
    asm_filename = os.path.splitext(ll_filename)[0] + ".s"
    try:
        llc_cmd = ["llc", "-O0", ll_filename, "-o", asm_filename]
        print(f"Compiling LLVM IR to assembly: {' '.join(llc_cmd)}")
        subprocess.run(llc_cmd, check=True)
        print(f"Assembly generated: {asm_filename}")

        compile_cmd = ["clang", "-O0", asm_filename, "-lm", "-o", executable]
        print(f"Compiling assembly to executable: {' '.join(compile_cmd)}")
        subprocess.run(compile_cmd, check=True)
        print(f"Executable generated: {executable}")

        print(f"Running executable: ./{executable}")
        start_time = time.perf_counter()
        subprocess.run([f"./{executable}"], check=True)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Execution time: {elapsed_time:.6f} seconds")

        return elapsed_time

    except subprocess.CalledProcessError as e:
        print(f"An error occurred during compilation or execution: {e}")
        raise

    finally:
        if os.path.exists(asm_filename):
            print(f"Cleaned up assembly file: {asm_filename}")


csv_file = "benchmark_results.csv"
# baseline_time = 0.0

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
