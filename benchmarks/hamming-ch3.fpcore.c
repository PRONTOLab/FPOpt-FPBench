#include <fenv.h>
#include <math.h>
#include <stdint.h>
#define TRUE 1
#define FALSE 0

double ex0(double x) {
	return sqrt((x + 1.0)) - sqrt(x);
}

double ex1(double x, double eps) {
	return sin((x + eps)) - sin(x);
}

double ex2(double x) {
	return (1.0 - cos(x)) / sin(x);
}

double ex3(double N) {
	return atan((N + 1.0)) - atan(N);
}

double ex4(double x) {
	return (1.0 / sqrt(x)) - (1.0 / sqrt((x + 1.0)));
}

double ex5(double x) {
	return (1.0 / (x + 1.0)) - (1.0 / x);
}

double ex6(double x, double eps) {
	return tan((x + eps)) - tan(x);
}

double ex7(double x) {
	return ((1.0 / (x + 1.0)) - (2.0 / x)) + (1.0 / (x - 1.0));
}

double ex8(double x) {
	return pow((x + 1.0), (1.0 / 3.0)) - pow(x, (1.0 / 3.0));
}

double ex9(double x, double eps) {
	return cos((x + eps)) - cos(x);
}

double ex10(double N) {
	return log((N + 1.0)) - log(N);
}

double ex11(double x) {
	return (exp(x) - 2.0) + exp(-x);
}

double ex12(double a, double b, double c) {
	return (-b + sqrt(((b * b) - (4.0 * (a * c))))) / (2.0 * a);
}

double ex13(double a, double b, double c) {
	return (-b - sqrt(((b * b) - (4.0 * (a * c))))) / (2.0 * a);
}

double ex14(double a, double b2, double c) {
	return (-b2 + sqrt(((b2 * b2) - (a * c)))) / a;
}

double ex15(double a, double b2, double c) {
	return (-b2 - sqrt(((b2 * b2) - (a * c)))) / a;
}

double ex16(double x) {
	return exp(x) - 1.0;
}

double ex17(double N) {
	return (((N + 1.0) * log((N + 1.0))) - (N * log(N))) - 1.0;
}

double ex18(double x) {
	return (1.0 / x) - (1.0 / tan(x));
}

double ex19(double x) {
	return log((1.0 - x)) / log((1.0 + x));
}

double ex20(double x) {
	return (1.0 - cos(x)) / (x * x);
}

double ex21(double a, double b, double eps) {
	return (eps * (exp(((a + b) * eps)) - 1.0)) / ((exp((a * eps)) - 1.0) * (exp((b * eps)) - 1.0));
}

double ex22(double eps) {
	return log(((1.0 - eps) / (1.0 + eps)));
}

double ex23(double x) {
	return sqrt(((exp((2.0 * x)) - 1.0) / (exp(x) - 1.0)));
}

double ex24(double x) {
	return (x - sin(x)) / (x - tan(x));
}

double ex25(double x, double n) {
	return pow((x + 1.0), (1.0 / n)) - pow(x, (1.0 / n));
}

double ex26(double a, double x) {
	return exp((a * x)) - 1.0;
}

double ex27(double x) {
	return exp(x) / (exp(x) - 1.0);
}

