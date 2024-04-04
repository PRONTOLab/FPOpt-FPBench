#include <fenv.h>
#include <math.h>
#include <stdint.h>
#define TRUE 1
#define FALSE 0

double ex0() {
	double tmp;
	if (1.0 < 1.0) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex1(double arg1) {
	double tmp;
	if (arg1 < 1.0) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex2(double arg1) {
	double tmp;
	if (1.0 < arg1) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex3(double arg1) {
	double tmp;
	if (arg1 < arg1) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex4(double arg1, double arg2) {
	double tmp;
	if (arg1 < arg2) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex5(double arg1, double arg2) {
	double tmp;
	if (arg2 < arg1) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex6() {
	double tmp;
	if (1.0 > 1.0) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex7(double arg1) {
	double tmp;
	if (arg1 > 1.0) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex8(double arg1) {
	double tmp;
	if (1.0 > arg1) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex9(double arg1) {
	double tmp;
	if (arg1 > arg1) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex10(double arg1, double arg2) {
	double tmp;
	if (arg1 > arg2) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex11(double arg1, double arg2) {
	double tmp;
	if (arg2 > arg1) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex12() {
	double tmp;
	if (1.0 <= 1.0) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex13(double arg1) {
	double tmp;
	if (arg1 <= 1.0) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex14(double arg1) {
	double tmp;
	if (1.0 <= arg1) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex15(double arg1) {
	double tmp;
	if (arg1 <= arg1) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex16(double arg1, double arg2) {
	double tmp;
	if (arg1 <= arg2) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex17(double arg1, double arg2) {
	double tmp;
	if (arg2 <= arg1) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex18() {
	double tmp;
	if (1.0 >= 1.0) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex19(double arg1) {
	double tmp;
	if (arg1 >= 1.0) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex20(double arg1) {
	double tmp;
	if (1.0 >= arg1) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex21(double arg1) {
	double tmp;
	if (arg1 >= arg1) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex22(double arg1, double arg2) {
	double tmp;
	if (arg1 >= arg2) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex23(double arg1, double arg2) {
	double tmp;
	if (arg2 >= arg1) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex24() {
	double tmp;
	if (1.0 == 1.0) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex25(double arg1) {
	double tmp;
	if (arg1 == 1.0) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex26(double arg1) {
	double tmp;
	if (1.0 == arg1) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex27(double arg1) {
	double tmp;
	if (arg1 == arg1) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex28(double arg1, double arg2) {
	double tmp;
	if (arg1 == arg2) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex29(double arg1, double arg2) {
	double tmp;
	if (arg2 == arg1) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex30() {
	double tmp;
	if (1.0 != 1.0) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex31(double arg1) {
	double tmp;
	if (arg1 != 1.0) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex32(double arg1) {
	double tmp;
	if (1.0 != arg1) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex33(double arg1) {
	double tmp;
	if (arg1 != arg1) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex34(double arg1, double arg2) {
	double tmp;
	if (arg1 != arg2) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex35(double arg1, double arg2) {
	double tmp;
	if (arg2 != arg1) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex36() {
	double tmp;
	if (isfinite(1.0)) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex37(double arg1) {
	double tmp;
	if (isfinite(arg1)) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex38() {
	double tmp;
	if (isinf(1.0)) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex39(double arg1) {
	double tmp;
	if (isinf(arg1)) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex40() {
	double tmp;
	if (isnan(1.0)) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex41(double arg1) {
	double tmp;
	if (isnan(arg1)) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex42() {
	double tmp;
	if (isnormal(1.0)) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex43(double arg1) {
	double tmp;
	if (isnormal(arg1)) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex44() {
	double tmp;
	if (signbit(1.0)) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex45(double arg1) {
	double tmp;
	if (signbit(arg1)) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

