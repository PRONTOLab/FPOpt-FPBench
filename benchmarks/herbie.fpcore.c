#include <fenv.h>
#include <math.h>
#include <stdint.h>
#define TRUE 1
#define FALSE 0

double ex0(double re, double im) {
	return 0.5 * sqrt((2.0 * (sqrt(((re * re) + (im * im))) + re)));
}

double ex1(double re, double im) {
	return (0.5 * sin(re)) * (exp(-im) - exp(im));
}

double ex2(double cp, double cn, double t, double s) {
	return (pow((1.0 / (1.0 + exp(-s))), cp) * pow((1.0 - (1.0 / (1.0 + exp(-s)))), cn)) / (pow((1.0 / (1.0 + exp(-t))), cp) * pow((1.0 - (1.0 / (1.0 + exp(-t)))), cn));
}

