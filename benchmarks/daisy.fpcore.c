#include <fenv.h>
#include <math.h>
#include <stdint.h>
#define TRUE 1
#define FALSE 0

double ex0(double x, double y) {
	return sqrt(((x * x) + (y * y)));
}

double ex1(double x, double y) {
	double pi = 3.14159265359;
	double radiant = atan((y / x));
	return radiant * (180.0 / pi);
}

double ex2(double radius, double theta) {
	double pi = 3.14159265359;
	double radiant = theta * (pi / 180.0);
	return radius * cos(radiant);
}

double ex3(double radius, double theta) {
	double pi = 3.14159265359;
	double radiant = theta * (pi / 180.0);
	return radius * sin(radiant);
}

double ex4(double t, double resistance, double frequency, double inductance, double maxVoltage) {
	double pi = 3.14159265359;
	double impedance_re = resistance;
	double impedance_im = ((2.0 * pi) * frequency) * inductance;
	double denom = (impedance_re * impedance_re) + (impedance_im * impedance_im);
	double current_re = (maxVoltage * impedance_re) / denom;
	double current_im = -(maxVoltage * impedance_im) / denom;
	double maxCurrent = sqrt(((current_re * current_re) + (current_im * current_im)));
	double theta = atan((current_im / current_re));
	return maxCurrent * cos(((((2.0 * pi) * frequency) * t) + theta));
}

double ex5(double a, double b, double c, double d, double e, double f, double g, double h, double i) {
	return ((((a * e) * i) + ((b * f) * g)) + ((c * d) * h)) - ((((c * e) * g) + ((b * d) * i)) + ((a * f) * h));
}

double ex6(double a, double b, double c, double d, double e, double f, double g, double h, double i) {
	return ((a * (e * i)) + ((g * (b * f)) + (c * (d * h)))) - ((e * (c * g)) + ((i * (b * d)) + (a * (f * h))));
}

