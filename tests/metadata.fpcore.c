#include <fenv.h>
#include <math.h>
#include <stdint.h>
#define TRUE 1
#define FALSE 0

double ex0(double x) {
	double tmp;
	if (1.0 < x && x < 1.000001) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex1(double x) {
	double tmp;
	if (x == 1.0) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex2(double x) {
	double tmp;
	if (-1e-320 < x && x < 1e-320) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex3(double x) {
	double tmp;
	if (1.797693134862315e+308 < x && x < ((double) INFINITY)) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex4(double x) {
	double tmp;
	if ((-5.0 < x && x < -1.0) || (1.0 < x && x < 5.0)) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex5(double x) {
	double tmp;
	if ((1.0 < x && x < 1.00001) || (1.7e+308 < x && x < ((double) INFINITY))) {
		tmp = 1.0;
	} else {
		tmp = 0.0;
	}
	return tmp;
}

double ex6(double x) {
	double tmp = 6755399441055744.0;
	double n = tmp;
	double tmp_1 = (x + n) - n;
	return tmp_1;
}

double ex7(double x) {
	float tmp = x - 1.0f;
	float y = tmp;
	return ((double) y) + 1.0;
}

double ex8(double x) {
	fesetround(FE_UPWARD);
	float tmp = x - 1.0f;
	fesetround(FE_TONEAREST);
	float y = tmp;
	return ((double) y) + 1.0;
}

double ex9(double x, double y) {
	fesetround(FE_UPWARD);
	double tmp_3 = x + y;
	fesetround(FE_TONEAREST);
	fesetround(FE_DOWNWARD);
	double tmp_4 = x + y;
	fesetround(FE_TONEAREST);
	double tmp_2 = tmp_3 - tmp_4;
	return tmp_2;
}

double ex10(double x, double y) {
	float tmp_3 = x + y;
	float tmp_4 = x - y;
	double tmp_2 = ((double) tmp_3) - ((double) tmp_4);
	return tmp_2;
}

double ex11(double x, double y) {
	double tmp_3 = x + y;
	double tmp_4 = x - y;
	float tmp_2 = tmp_3 - tmp_4;
	return tmp_2;
}

double ex12(double x) {
	double tmp = (float) x;
	return (double) tmp;
}

double ex13(double x) {
	double tmp = x;
	return tmp;
}

double ex14(double x) {
	double tmp_2 = x;
	double tmp_1 = tmp_2;
	return tmp_1;
}

double ex15(double x) {
	double y_1 = x;
	double tmp = y_1;
	return tmp;
}

float ex16(float x) {
	double tmp = exp(x);
	return tmp;
}

double ex17() {
	double x = 0x1.000002p-1;
	double y = 0x1.ffffffffp-26;
	double tmp_2 = x + y;
	double tmp_1 = (float) tmp_2;
	return tmp_1;
}

float ex18() {
	float x = 0x1p0;
	float y = 0x1p-30;
	double tmp = ((double) x) + ((double) y);
	return tmp;
}

double ex19(float x) {
	return x;
}

double ex20(float x, double y) {
	return ((double) x) + y;
}

double ex21(float x) {
	float y = x;
	return ((double) y) + ((double) x);
}

double ex22(double x) {
	float tmp_2;
	if (x < 0.0f) {
		tmp_2 = ((float) M_E) + x;
	} else {
		tmp_2 = ((float) M_E) - x;
	}
	float tmp_1 = tmp_2;
	return tmp_1;
}

double ex23(double x) {
	float y_6 = x + 1.0f;
	float z_7 = x - 1.0f;
	double y_11 = x + ((double) y_6);
	double z_12 = x + ((double) z_7);
	double tmp_10 = y_11 + z_12;
	double tmp_5 = tmp_10;
	return tmp_5;
}

