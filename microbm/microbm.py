import time
import csv
import os
import struct
import numpy as np
import random
import llvmlite.binding as llvm
import ctypes

random.seed(42)

llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

FAST_MATH_FLAG = "reassoc nsz arcp contract afn"

instructions = ["fneg", "fadd", "fsub", "fmul", "fdiv", "fcmp", "fptrunc", "fpext"]
functions = [
    "fmuladd",
    "sin",
    "cos",
    "tan",
    "exp",
    "log",
    "sqrt",
    "expm1",
    "log1p",
    "cbrt",
    "pow",
    "fabs",
    "hypot",
    "fma",
]
precisions = ["float", "double"]
iterations = 100000000
unrolled = 128

precision_to_llvm_type = {
    "double": "double",
    "float": "float",
    "half": "half",
    "fp80": "x86_fp80",
    "fp128": "fp128",
    "bf16": "bfloat",
}

precision_to_intrinsic_suffix = {
    "double": "f64",
    "float": "f32",
    "half": "f16",
    "fp80": "f80",
    "fp128": "f128",
    "bf16": "bf16",
}

functions_with_intrinsics = {"sin", "cos", "exp", "log", "sqrt", "pow", "fabs", "fmuladd"}

precision_ranks = {"bf16": 0, "half": 1, "float": 2, "double": 3, "fp80": 4, "fp128": 5}
precisions_ordered = ["bf16", "half", "float", "double", "fp80", "fp128"]


def get_zero_literal(precision):
    if precision == "double":
        return "0.0"
    elif precision == "float":
        return "0.0"
    elif precision == "half":
        return "0.0"
    elif precision == "bf16":
        return "0xR0000"
    elif precision == "fp80":
        return "0xK00000000000000000000"
    elif precision == "fp128":
        return "0xL00000000000000000000000000000000"
    else:
        return "0.0"


def float64_to_fp80_bytes(value: np.float64) -> bytes:
    packed = struct.pack(">d", value)
    (bits,) = struct.unpack(">Q", packed)
    sign = (bits >> 63) & 0x1
    exponent = (bits >> 52) & 0x7FF
    mantissa = bits & 0xFFFFFFFFFFFFF

    if exponent == 0:
        if mantissa == 0:
            fp80_exponent = 0
            fp80_mantissa = 0
        else:
            shift = 0
            while (mantissa & (1 << 52)) == 0:
                mantissa <<= 1
                shift += 1
            exponent = 1 - shift
            exponent_bias_64 = 1023
            exponent_bias_80 = 16383
            fp80_exponent = exponent - exponent_bias_64 + exponent_bias_80
            fp80_mantissa = mantissa << (63 - 52)
    elif exponent == 0x7FF:
        fp80_exponent = 0x7FFF
        if mantissa == 0:
            fp80_mantissa = 0x8000000000000000
        else:
            fp80_mantissa = 0xC000000000000000 | (mantissa << (63 - 52))
    else:
        exponent_bias_64 = 1023
        exponent_bias_80 = 16383
        fp80_exponent = exponent - exponent_bias_64 + exponent_bias_80
        fp80_mantissa = (0x8000000000000000) | (mantissa << (63 - 52))

    exponent_sign = (sign << 15) | fp80_exponent
    fp80_bits = (exponent_sign << 64) | fp80_mantissa
    fp80_bytes = fp80_bits.to_bytes(10, byteorder="big")
    return fp80_bytes


def float64_to_fp128_bytes(value: np.float64) -> bytes:
    packed = struct.pack(">d", value)
    (bits,) = struct.unpack(">Q", packed)
    sign = (bits >> 63) & 0x1
    exponent = (bits >> 52) & 0x7FF
    mantissa = bits & 0xFFFFFFFFFFFFF

    if exponent == 0:
        fp128_exponent = 0
    elif exponent == 0x7FF:
        fp128_exponent = 0x7FFF
    else:
        exponent_bias_64 = 1023
        exponent_bias_128 = 16383
        fp128_exponent = exponent - exponent_bias_64 + exponent_bias_128

    fp128_mantissa = mantissa << 60
    fp128_bits = (sign << 127) | (fp128_exponent << 112) | fp128_mantissa
    fp128_bytes = fp128_bits.to_bytes(16, byteorder="big")
    return fp128_bytes


def float_to_llvm_hex(f, precision):
    if precision == "double":
        f_cast = np.float64(f)
        packed = struct.pack(">d", f_cast)
        [i] = struct.unpack(">Q", packed)
        return f"0x{i:016X}"
    elif precision == "float":
        f_cast = np.float32(f)
        packed = struct.pack(">d", f_cast)
        [i] = struct.unpack(">Q", packed)
        return f"0x{i:016X}"
    elif precision == "half":
        f_cast = np.float16(f)
        packed = f_cast.tobytes()
        [i] = struct.unpack(">H", packed)
        return f"0xH{i:04X}"
    elif precision == "bf16":
        f_cast = np.float32(f)
        [bits] = struct.unpack(">I", struct.pack(">f", f_cast))
        bf16_bits = bits >> 16
        return f"0xR{bf16_bits:04X}"
    elif precision == "fp80":
        f_cast = np.float64(f)
        fp80_bytes = float64_to_fp80_bytes(f_cast)
        return f"0xK{fp80_bytes.hex().upper()}"
    elif precision == "fp128":
        f_cast = np.float64(f)
        fp128_bytes = float64_to_fp128_bytes(f_cast)
        swapped = fp128_bytes[8:] + fp128_bytes[:8]
        return f"0xL{swapped.hex().upper()}"
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
    elif precision == "bf16":
        f = random.uniform(-1e3, 1e3)
        dtype = np.float32
    elif precision == "fp80":
        f = random.uniform(-1e10, 1e10)
        dtype = np.float64
    elif precision == "fp128":
        f = random.uniform(-1e10, 1e10)
        dtype = np.float64
    else:
        f = random.uniform(-1e3, 1e3)
        dtype = np.float64
    return dtype(f).item()


def generate_baseline_code(iterations):
    return f"""
define i32 @main() optnone noinline {{
entry:
  %i = alloca i32
  store i32 0, i32* %i
  br label %loop

loop:
  %i_val = load i32, i32* %i
  %cond = icmp slt i32 %i_val, {iterations}
  br i1 %cond, label %body, label %exit

body:
  %i_next = add i32 %i_val, 1
  store i32 %i_next, i32* %i
  br label %loop

exit:
  ret i32 0
}}
"""


def generate_llvm_code(instruction, src_precision, dst_precision, iterations):
    src_type = precision_to_llvm_type.get(src_precision)
    dst_type = precision_to_llvm_type.get(dst_precision)
    if not src_type or not dst_type:
        return ""
    zero_literal_dst = get_zero_literal(dst_precision)
    if instruction == "fptrunc":
        random_fps = [generate_random_fp(src_precision) for _ in range(unrolled)]
        hex_fps = [float_to_llvm_hex(f, src_precision) for f in random_fps]
        code = f"""
define i32 @main() optnone noinline {{
entry:
  %i = alloca i32
  %acc = alloca {dst_type}
  store i32 0, i32* %i
  store {dst_type} {zero_literal_dst}, {dst_type}* %acc
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
            code += f"  %acc_val{idx+1} = fadd {FAST_MATH_FLAG} {dst_type} %acc_val{idx}, %result{idx}\n"
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
        return code

    elif instruction == "fpext":
        random_fps = [generate_random_fp(src_precision) for _ in range(unrolled)]
        hex_fps = [float_to_llvm_hex(f, src_precision) for f in random_fps]
        code = f"""
define i32 @main() optnone noinline {{
entry:
  %i = alloca i32
  %acc = alloca {dst_type}
  store i32 0, i32* %i
  store {dst_type} {zero_literal_dst}, {dst_type}* %acc
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
            code += f"  %acc_val{idx+1} = fadd {FAST_MATH_FLAG} {dst_type} %acc_val{idx}, %result{idx}\n"
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
        return code
    return ""


def generate_llvm_code_other(instruction, precision, iterations):
    llvm_type = precision_to_llvm_type[precision]
    zero_literal = get_zero_literal(precision)
    if instruction in ["fadd", "fsub", "fmul", "fdiv"]:
        op_map = {"fadd": "fadd", "fsub": "fsub", "fmul": "fmul", "fdiv": "fdiv"}
        op = op_map[instruction]
        random_pairs = [(generate_random_fp(precision), generate_random_fp(precision)) for _ in range(unrolled)]
        hex_pairs = [(float_to_llvm_hex(a, precision), float_to_llvm_hex(b, precision)) for a, b in random_pairs]
        code = f"""
define i32 @main() optnone noinline {{
entry:
  %i = alloca i32
  %acc = alloca {llvm_type}
  store i32 0, i32* %i
  store {llvm_type} {zero_literal}, {llvm_type}* %acc
  br label %loop

loop:
  %i_val = load i32, i32* %i
  %cond = icmp slt i32 %i_val, {iterations}
  br i1 %cond, label %body, label %exit

body:
  %acc_val0 = load {llvm_type}, {llvm_type}* %acc
"""
        for idx, (hex_a, hex_b) in enumerate(hex_pairs):
            code += f"  %result{idx} = {op} {FAST_MATH_FLAG} {llvm_type} {hex_a}, {hex_b}\n"
            code += f"  %acc_val{idx+1} = fadd {FAST_MATH_FLAG} {llvm_type} %acc_val{idx}, %result{idx}\n"
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
        return code

    elif instruction == "fneg":
        random_fps = [generate_random_fp(precision) for _ in range(unrolled)]
        hex_fps = [float_to_llvm_hex(f, precision) for f in random_fps]
        code = f"""
define i32 @main() optnone noinline {{
entry:
  %i = alloca i32
  %acc = alloca {llvm_type}
  store i32 0, i32* %i
  store {llvm_type} {zero_literal}, {llvm_type}* %acc
  br label %loop

loop:
  %i_val = load i32, i32* %i
  %cond = icmp slt i32 %i_val, {iterations}
  br i1 %cond, label %body, label %exit

body:
  %acc_val0 = load {llvm_type}, {llvm_type}* %acc
"""
        for idx, hex_a in enumerate(hex_fps):
            code += f"  %result{idx} = fneg {FAST_MATH_FLAG} {llvm_type} {hex_a}\n"
            code += f"  %acc_val{idx+1} = fadd {FAST_MATH_FLAG} {llvm_type} %acc_val{idx}, %result{idx}\n"
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
        return code

    elif instruction == "fcmp":
        code = f"""
define i32 @main() optnone noinline {{
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
            code += f"  %cmp{idx} = fcmp {FAST_MATH_FLAG} olt {llvm_type} {hex_a}, {hex_b}\n"
            code += f"  %cmp_int{idx} = zext i1 %cmp{idx} to i32\n"

        code += "  %acc_val0 = load i32, i32* %acc\n"
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
        return code
    return ""


def generate_llvm_function_call(function_name, precision, iterations):
    llvm_type = precision_to_llvm_type[precision]
    intrinsic_suffix = precision_to_intrinsic_suffix.get(precision)
    if not intrinsic_suffix:
        return ""
    zero_literal = get_zero_literal(precision)
    if function_name == "pow":
        fn = f"llvm.pow.{intrinsic_suffix}"
        decl = f"declare {llvm_type} @{fn}({llvm_type}, {llvm_type})"
        tmpl = f"call {FAST_MATH_FLAG} {llvm_type} @{fn}({llvm_type} {{arg1}}, {llvm_type} {{arg2}})"
    elif function_name == "fmuladd":
        fn = f"llvm.fmuladd.{intrinsic_suffix}"
        decl = f"declare {llvm_type} @{fn}({llvm_type}, {llvm_type}, {llvm_type})"
        tmpl = (
            f"call {FAST_MATH_FLAG} {llvm_type} @{fn}({llvm_type} {{arg1}}, {llvm_type} {{arg2}}, {llvm_type} {{arg3}})"
        )
    elif function_name in functions_with_intrinsics:
        fn = f"llvm.{function_name}.{intrinsic_suffix}"
        decl = f"declare {llvm_type} @{fn}({llvm_type})"
        tmpl = f"call {FAST_MATH_FLAG} {llvm_type} @{fn}({llvm_type} {{arg1}})"
    else:
        fn = function_name
        decl = f"declare {llvm_type} @{fn}({llvm_type})"
        tmpl = f"call {FAST_MATH_FLAG} {llvm_type} @{fn}({llvm_type} {{arg1}})"

    code = (
        decl
        + f"""
define i32 @main() optnone noinline {{
entry:
  %i = alloca i32
  %acc = alloca {llvm_type}
  store i32 0, i32* %i
  store {llvm_type} {zero_literal}, {llvm_type}* %acc
  br label %loop

loop:
  %i_val = load i32, i32* %i
  %cond = icmp slt i32 %i_val, {iterations}
  br i1 %cond, label %body, label %exit

body:
  %acc_val0 = load {llvm_type}, {llvm_type}* %acc
"""
    )
    for idx in range(unrolled):
        if function_name in ["pow", "hypot"]:
            a = generate_random_fp(precision)
            b = generate_random_fp(precision)
            hex_a = float_to_llvm_hex(a, precision)
            hex_b = float_to_llvm_hex(b, precision)
            call_ = tmpl.format(arg1=hex_a, arg2=hex_b)
        elif function_name == "fmuladd":
            a = generate_random_fp(precision)
            b = generate_random_fp(precision)
            c = generate_random_fp(precision)
            hex_a = float_to_llvm_hex(a, precision)
            hex_b = float_to_llvm_hex(b, precision)
            hex_c = float_to_llvm_hex(c, precision)
            call_ = tmpl.format(arg1=hex_a, arg2=hex_b, arg3=hex_c)
        else:
            a = generate_random_fp(precision)
            hex_a = float_to_llvm_hex(a, precision)
            call_ = tmpl.format(arg1=hex_a)
        code += f"  %result{idx} = {call_}\n"
        code += f"  %acc_val{idx+1} = fadd {FAST_MATH_FLAG} {llvm_type} %acc_val{idx}, %result{idx}\n"
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


def create_execution_engine():
    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()
    mod = llvm.parse_assembly("")
    engine = llvm.create_mcjit_compiler(mod, target_machine)
    return engine


def run_llvm_ir_jit(llvm_ir):
    engine = create_execution_engine()
    mod = llvm.parse_assembly(llvm_ir)
    mod.verify()
    engine.add_module(mod)
    engine.finalize_object()
    func_ptr = engine.get_function_address("main")
    cfunc = ctypes.CFUNCTYPE(ctypes.c_int)(func_ptr)
    start = time.perf_counter()
    retval = cfunc()
    end = time.perf_counter()
    print(f"DEBUG: JIT-execution time: {end - start:.6f}s")
    return (end - start), retval


csv_file = "results.csv"

with open(csv_file, "w", newline="") as csvfile:
    fieldnames = ["instruction", "precision", "cost"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    llvm_code = generate_baseline_code(iterations)
    print("DEBUG: Running baseline")
    baseline_time, _ = run_llvm_ir_jit(llvm_code)

    for precision in precisions:
        for instr in instructions:
            if instr in ["fptrunc", "fpext"]:
                src_precision = precision
                src_rank = precision_ranks.get(src_precision)
                if src_rank is None:
                    continue
                if instr == "fptrunc":
                    dst_precisions = [
                        p for p in precisions_ordered if p in precisions and precision_ranks[p] < src_rank
                    ]
                else:
                    dst_precisions = [
                        p for p in precisions_ordered if p in precisions and precision_ranks[p] > src_rank
                    ]
                for dst_precision in dst_precisions:
                    if (src_precision == "half" and dst_precision == "bf16") or (
                        src_precision == "bf16" and dst_precision == "half"
                    ):
                        continue
                    code = generate_llvm_code(instr, src_precision, dst_precision, iterations)
                    if not code.strip():
                        continue
                    print(f"DEBUG: Running '{instr}_{src_precision}_to_{dst_precision}'")
                    elapsed, _ = run_llvm_ir_jit(code)
                    adjusted = elapsed - baseline_time
                    writer.writerow(
                        {
                            "instruction": f"{instr}_{src_precision}_to_{dst_precision}",
                            "precision": src_precision,
                            "cost": int(adjusted),
                        }
                    )
            else:
                code = generate_llvm_code_other(instr, precision, iterations)
                if not code.strip():
                    continue
                print(f"DEBUG: Running '{instr}'")
                elapsed, _ = run_llvm_ir_jit(code)
                adjusted = elapsed - baseline_time
                writer.writerow({"instruction": instr, "precision": precision, "cost": int(adjusted)})

        for func in functions:
            code = generate_llvm_function_call(func, precision, iterations)
            if not code.strip():
                continue
            print(f"DEBUG: Running '{func}'")
            elapsed, _ = run_llvm_ir_jit(code)
            adjusted = elapsed - baseline_time
            writer.writerow({"instruction": func, "precision": precision, "cost": int(adjusted)})

print(f"Results in '{csv_file}'. Baseline: {baseline_time:.6f}s")
