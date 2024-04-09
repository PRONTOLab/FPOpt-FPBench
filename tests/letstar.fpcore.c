#include <fenv.h>
#include <math.h>
#include <stdint.h>
#define TRUE 1
#define FALSE 0

double ex0(double a, double b) {
	double a_1 = b;
	double b_2 = a;
	return a_1 - b_2;
}

double ex1(double a, double b) {
	double a_1 = b;
	double b_2 = a_1;
	return a_1 - b_2;
}

double ex2(double a, double b) {
	double a_1 = 1.0;
	double b_2 = a_1;
	return b_2;
}

double ex3(double a) {
	// TODO hangs on mpfr
	// double c = 0.0;
	// double d = 0.0;
	// int tmp = c < a;
	// while (tmp) {
	// 	double c_1 = 1.0 + c;
	// 	double d_2 = d + c;
	// 	c = c_1;
	// 	d = d_2;
	// 	tmp = c < a;
	// }
	// return d;
	return 0;
}

double ex4(double a) {
	// TODO hangs on mpfr
	// double c = 0.0;
	// double d = 0.0;
	// int tmp = c < a;
	// while (tmp) {
	// 	c = 1.0 + c;
	// 	d = d + c;
	// 	tmp = c < a;
	// }
	// return d;
	return 0;
}

