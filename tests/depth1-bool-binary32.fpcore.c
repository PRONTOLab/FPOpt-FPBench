#include <fenv.h>
#include <math.h>
#include <stdint.h>
#define TRUE 1
#define FALSE 0

float ex0() {
	float tmp;
	if (1.0f < 1.0f) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex1(float arg1) {
	float tmp;
	if (arg1 < 1.0f) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex2(float arg1) {
	float tmp;
	if (1.0f < arg1) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex3(float arg1) {
	float tmp;
	if (arg1 < arg1) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex4(float arg1, float arg2) {
	float tmp;
	if (arg1 < arg2) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex5(float arg1, float arg2) {
	float tmp;
	if (arg2 < arg1) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex6() {
	float tmp;
	if (1.0f > 1.0f) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex7(float arg1) {
	float tmp;
	if (arg1 > 1.0f) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex8(float arg1) {
	float tmp;
	if (1.0f > arg1) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex9(float arg1) {
	float tmp;
	if (arg1 > arg1) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex10(float arg1, float arg2) {
	float tmp;
	if (arg1 > arg2) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex11(float arg1, float arg2) {
	float tmp;
	if (arg2 > arg1) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex12() {
	float tmp;
	if (1.0f <= 1.0f) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex13(float arg1) {
	float tmp;
	if (arg1 <= 1.0f) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex14(float arg1) {
	float tmp;
	if (1.0f <= arg1) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex15(float arg1) {
	float tmp;
	if (arg1 <= arg1) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex16(float arg1, float arg2) {
	float tmp;
	if (arg1 <= arg2) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex17(float arg1, float arg2) {
	float tmp;
	if (arg2 <= arg1) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex18() {
	float tmp;
	if (1.0f >= 1.0f) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex19(float arg1) {
	float tmp;
	if (arg1 >= 1.0f) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex20(float arg1) {
	float tmp;
	if (1.0f >= arg1) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex21(float arg1) {
	float tmp;
	if (arg1 >= arg1) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex22(float arg1, float arg2) {
	float tmp;
	if (arg1 >= arg2) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex23(float arg1, float arg2) {
	float tmp;
	if (arg2 >= arg1) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex24() {
	float tmp;
	if (1.0f == 1.0f) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex25(float arg1) {
	float tmp;
	if (arg1 == 1.0f) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex26(float arg1) {
	float tmp;
	if (1.0f == arg1) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex27(float arg1) {
	float tmp;
	if (arg1 == arg1) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex28(float arg1, float arg2) {
	float tmp;
	if (arg1 == arg2) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex29(float arg1, float arg2) {
	float tmp;
	if (arg2 == arg1) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex30() {
	float tmp;
	if (1.0f != 1.0f) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex31(float arg1) {
	float tmp;
	if (arg1 != 1.0f) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex32(float arg1) {
	float tmp;
	if (1.0f != arg1) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex33(float arg1) {
	float tmp;
	if (arg1 != arg1) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex34(float arg1, float arg2) {
	float tmp;
	if (arg1 != arg2) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex35(float arg1, float arg2) {
	float tmp;
	if (arg2 != arg1) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex36() {
	float tmp;
	if (isfinite(1.0f)) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex37(float arg1) {
	float tmp;
	if (isfinite(arg1)) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex38() {
	float tmp;
	if (isinf(1.0f)) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex39(float arg1) {
	float tmp;
	if (isinf(arg1)) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex40() {
	float tmp;
	if (isnan(1.0f)) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex41(float arg1) {
	float tmp;
	if (isnan(arg1)) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex42() {
	float tmp;
	if (isnormal(1.0f)) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex43(float arg1) {
	float tmp;
	if (isnormal(arg1)) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex44() {
	float tmp;
	if (signbit(1.0f)) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

float ex45(float arg1) {
	float tmp;
	if (signbit(arg1)) {
		tmp = 1.0f;
	} else {
		tmp = 0.0f;
	}
	return tmp;
}

