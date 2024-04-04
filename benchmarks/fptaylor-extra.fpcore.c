#include <fenv.h>
#include <math.h>
#include <stdint.h>
#define TRUE 1
#define FALSE 0

float ex0(float t) {
	float tmp_2 = t + 1.0f;
	double tmp_1 = ((double) t) / ((double) tmp_2);
	return (float) tmp_1;
}

double ex1(double x1, double x2, double x3, double x4, double x5, double x6) {
	return (((((-x2 * x3) - (x1 * x4)) + (x2 * x5)) + (x3 * x6)) - (x5 * x6)) + (x1 * (((((-x1 + x2) + x3) - x4) + x5) + x6));
}

double ex2(double x1, double x2, double x3, double x4, double x5, double x6) {
	return (((((((x1 * x4) * (((((-x1 + x2) + x3) - x4) + x5) + x6)) + ((x2 * x5) * (((((x1 - x2) + x3) + x4) - x5) + x6))) + ((x3 * x6) * (((((x1 + x2) - x3) + x4) + x5) - x6))) + ((-x2 * x3) * x4)) + ((-x1 * x3) * x5)) + ((-x1 * x2) * x6)) + ((-x4 * x5) * x6);
}

double ex3(double x) {
	return 1.0 / (sqrt((x + 1.0)) + sqrt(x));
}

double ex4(double x) {
	return (exp(x) - 1.0) / x;
}

float ex5(float x) {
	return (expf(x) - 1.0f) / x;
}

double ex6(double x1, double x2) {
	return x1 + x2;
}

double ex7(double x) {
	return (exp(x) - 1.0) / log(exp(x));
}

float ex8(float x, float y) {
	return x / (x + y);
}

double ex9(double x1, double x2) {
	return sqrt(((x1 * x1) + (x2 * x2)));
}

float ex10(float x1, float x2) {
	return sqrtf(((x1 * x1) + (x2 * x2)));
}

double ex11(double x) {
	return log((1.0 + exp(x)));
}

double ex12(double x0, double x1, double x2) {
	double p0 = (x0 + x1) - x2;
	double p1 = (x1 + x2) - x0;
	double p2 = (x2 + x0) - x1;
	return (p0 + p1) + p2;
}

double ex13(double z) {
	return z / (z + 1.0);
}

double ex14(double x, double y) {
	double t = x * y;
	return (t - 1.0) / ((t * t) - 1.0);
}

float ex15(float x, float y) {
	return sqrtf((x + (y * y)));
}

float ex16(float x, float y) {
	return sinf((x * y));
}

double ex17(double x1, double x2) {
	double a = ((x1 * x1) + x2) - 11.0;
	double b = (x1 + (x2 * x2)) - 7.0;
	return (a * a) + (b * b);
}

