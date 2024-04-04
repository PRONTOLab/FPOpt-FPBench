#include <fenv.h>
#include <math.h>
#include <stdint.h>
#define TRUE 1
#define FALSE 0

double ex0(double t) {
	return t / (t + 1.0);
}

double ex1(double x, double y) {
	double t = x * y;
	return (t - 1.0) / ((t * t) - 1.0);
}

float ex2(float x0, float x1, float x2) {
	float p0 = (x0 + x1) - x2;
	float p1 = (x1 + x2) - x0;
	float p2 = (x2 + x0) - x1;
	return (p0 + p1) + p2;
}

double ex3(double x0, double x1, double x2, double x3, double x4, double x5, double x6, double x7) {
	return ((((((x0 + x1) + x2) + x3) + x4) + x5) + x6) + x7;
}

double ex4(double x, double y) {
	return (x + y) / (x - y);
}

double ex5(double m0, double m1, double m2, double w0, double w1, double w2, double a0, double a1, double a2) {
	double v2 = (w2 * (0.0 - m2)) * (-3.0 * ((1.0 * (a2 / w2)) * (a2 / w2)));
	double v1 = (w1 * (0.0 - m1)) * (-3.0 * ((1.0 * (a1 / w1)) * (a1 / w1)));
	double v0 = (w0 * (0.0 - m0)) * (-3.0 * ((1.0 * (a0 / w0)) * (a0 / w0)));
	return 0.0 + ((v0 * 1.0) + ((v1 * 1.0) + ((v2 * 1.0) + 0.0)));
}

double ex6(double x) {
	double r1 = x - 1.0;
	double r2 = x * x;
	return r1 / (r2 - 1.0);
}

double ex7(double x) {
	return 1.0 / (x + 1.0);
}

float ex8(float x0, float x1, float x2, float x3) {
	return ((x0 + x1) + x2) + x3;
}

float ex9(float x0, float x1, float x2, float x3) {
	return (x0 + x1) + (x2 + x3);
}

