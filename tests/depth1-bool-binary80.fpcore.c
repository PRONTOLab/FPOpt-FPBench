#include <fenv.h>
#include <math.h>
#include <stdint.h>
#define TRUE 1
#define FALSE 0

long double ex0() {
	long double tmp;
	if (1.0l < 1.0l) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex1(long double arg1) {
	long double tmp;
	if (arg1 < 1.0l) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex2(long double arg1) {
	long double tmp;
	if (1.0l < arg1) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex3(long double arg1) {
	long double tmp;
	if (arg1 < arg1) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex4(long double arg1, long double arg2) {
	long double tmp;
	if (arg1 < arg2) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex5(long double arg1, long double arg2) {
	long double tmp;
	if (arg2 < arg1) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex6() {
	long double tmp;
	if (1.0l > 1.0l) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex7(long double arg1) {
	long double tmp;
	if (arg1 > 1.0l) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex8(long double arg1) {
	long double tmp;
	if (1.0l > arg1) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex9(long double arg1) {
	long double tmp;
	if (arg1 > arg1) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex10(long double arg1, long double arg2) {
	long double tmp;
	if (arg1 > arg2) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex11(long double arg1, long double arg2) {
	long double tmp;
	if (arg2 > arg1) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex12() {
	long double tmp;
	if (1.0l <= 1.0l) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex13(long double arg1) {
	long double tmp;
	if (arg1 <= 1.0l) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex14(long double arg1) {
	long double tmp;
	if (1.0l <= arg1) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex15(long double arg1) {
	long double tmp;
	if (arg1 <= arg1) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex16(long double arg1, long double arg2) {
	long double tmp;
	if (arg1 <= arg2) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex17(long double arg1, long double arg2) {
	long double tmp;
	if (arg2 <= arg1) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex18() {
	long double tmp;
	if (1.0l >= 1.0l) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex19(long double arg1) {
	long double tmp;
	if (arg1 >= 1.0l) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex20(long double arg1) {
	long double tmp;
	if (1.0l >= arg1) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex21(long double arg1) {
	long double tmp;
	if (arg1 >= arg1) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex22(long double arg1, long double arg2) {
	long double tmp;
	if (arg1 >= arg2) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex23(long double arg1, long double arg2) {
	long double tmp;
	if (arg2 >= arg1) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex24() {
	long double tmp;
	if (1.0l == 1.0l) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex25(long double arg1) {
	long double tmp;
	if (arg1 == 1.0l) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex26(long double arg1) {
	long double tmp;
	if (1.0l == arg1) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex27(long double arg1) {
	long double tmp;
	if (arg1 == arg1) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex28(long double arg1, long double arg2) {
	long double tmp;
	if (arg1 == arg2) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex29(long double arg1, long double arg2) {
	long double tmp;
	if (arg2 == arg1) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex30() {
	long double tmp;
	if (1.0l != 1.0l) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex31(long double arg1) {
	long double tmp;
	if (arg1 != 1.0l) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex32(long double arg1) {
	long double tmp;
	if (1.0l != arg1) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex33(long double arg1) {
	long double tmp;
	if (arg1 != arg1) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex34(long double arg1, long double arg2) {
	long double tmp;
	if (arg1 != arg2) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex35(long double arg1, long double arg2) {
	long double tmp;
	if (arg2 != arg1) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex36() {
	long double tmp;
	if (isfinite(1.0l)) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex37(long double arg1) {
	long double tmp;
	if (isfinite(arg1)) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex38() {
	long double tmp;
	if (isinf(1.0l)) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex39(long double arg1) {
	long double tmp;
	if (isinf(arg1)) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex40() {
	long double tmp;
	if (isnan(1.0l)) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex41(long double arg1) {
	long double tmp;
	if (isnan(arg1)) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex42() {
	long double tmp;
	if (isnormal(1.0l)) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex43(long double arg1) {
	long double tmp;
	if (isnormal(arg1)) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex44() {
	long double tmp;
	if (signbit(1.0l)) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

long double ex45(long double arg1) {
	long double tmp;
	if (signbit(arg1)) {
		tmp = 1.0l;
	} else {
		tmp = 0.0l;
	}
	return tmp;
}

