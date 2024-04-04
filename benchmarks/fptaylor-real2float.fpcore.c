#include <fenv.h>
#include <math.h>
#include <stdint.h>
#define TRUE 1
#define FALSE 0

double ex0(double x) {
	double e = exp(x);
	return log((1.0 + e));
}

double ex1(double x, double r, double lat, double lon) {
	double sinLat = sin(lat);
	double cosLon = cos(lon);
	return x + ((r * sinLat) * cosLon);
}

double ex2(double lat1, double lat2, double lon1, double lon2) {
	double dLon = lon2 - lon1;
	double s_lat1 = sin(lat1);
	double c_lat1 = cos(lat1);
	double s_lat2 = sin(lat2);
	double c_lat2 = cos(lat2);
	double s_dLon = sin(dLon);
	double c_dLon = cos(dLon);
	return atan(((c_lat2 * s_dLon) / ((c_lat1 * s_lat2) - ((s_lat1 * c_lat2) * c_dLon))));
}

double ex3(double x1, double x2, double x3, double x4, double x5, double x6) {
	return (((((-25.0 * ((x1 - 2.0) * (x1 - 2.0))) - ((x2 - 2.0) * (x2 - 2.0))) - ((x3 - 1.0) * (x3 - 1.0))) - ((x4 - 4.0) * (x4 - 4.0))) - ((x5 - 1.0) * (x5 - 1.0))) - ((x6 - 4.0) * (x6 - 4.0));
}

double ex4(double x1, double x2) {
	return -x1 - x2;
}

double ex5(double x1, double x2) {
	return ((-12.0 * x1) - (7.0 * x2)) + (x2 * x2);
}

double ex6(double x1, double x2, double x3) {
	double e1 = ((3.0 * ((x1 - 0.3689) * (x1 - 0.3689))) + (10.0 * ((x2 - 0.117) * (x2 - 0.117)))) + (30.0 * ((x3 - 0.2673) * (x3 - 0.2673)));
	double e2 = ((0.1 * ((x1 - 0.4699) * (x1 - 0.4699))) + (10.0 * ((x2 - 0.4387) * (x2 - 0.4387)))) + (35.0 * ((x3 - 0.747) * (x3 - 0.747)));
	double e3 = ((3.0 * ((x1 - 0.1091) * (x1 - 0.1091))) + (10.0 * ((x2 - 0.8732) * (x2 - 0.8732)))) + (30.0 * ((x3 - 0.5547) * (x3 - 0.5547)));
	double e4 = ((0.1 * ((x1 - 0.03815) * (x1 - 0.03815))) + (10.0 * ((x2 - 0.5743) * (x2 - 0.5743)))) + (35.0 * ((x3 - 0.8828) * (x3 - 0.8828)));
	double exp1 = exp(-e1);
	double exp2 = exp(-e2);
	double exp3 = exp(-e3);
	double exp4 = exp(-e4);
	return -((((1.0 * exp1) + (1.2 * exp2)) + (3.0 * exp3)) + (3.2 * exp4));
}

double ex7(double x1, double x2, double x3, double x4, double x5, double x6) {
	double e1 = (((((10.0 * ((x1 - 0.1312) * (x1 - 0.1312))) + (3.0 * ((x2 - 0.1696) * (x2 - 0.1696)))) + (17.0 * ((x3 - 0.5569) * (x3 - 0.5569)))) + (3.5 * ((x4 - 0.0124) * (x4 - 0.0124)))) + (1.7 * ((x5 - 0.8283) * (x5 - 0.8283)))) + (8.0 * ((x6 - 0.5886) * (x6 - 0.5886)));
	double e2 = (((((0.05 * ((x1 - 0.2329) * (x1 - 0.2329))) + (10.0 * ((x2 - 0.4135) * (x2 - 0.4135)))) + (17.0 * ((x3 - 0.8307) * (x3 - 0.8307)))) + (0.1 * ((x4 - 0.3736) * (x4 - 0.3736)))) + (8.0 * ((x5 - 0.1004) * (x5 - 0.1004)))) + (14.0 * ((x6 - 0.9991) * (x6 - 0.9991)));
	double e3 = (((((3.0 * ((x1 - 0.2348) * (x1 - 0.2348))) + (3.5 * ((x2 - 0.1451) * (x2 - 0.1451)))) + (1.7 * ((x3 - 0.3522) * (x3 - 0.3522)))) + (10.0 * ((x4 - 0.2883) * (x4 - 0.2883)))) + (17.0 * ((x5 - 0.3047) * (x5 - 0.3047)))) + (8.0 * ((x6 - 0.665) * (x6 - 0.665)));
	double e4 = (((((17.0 * ((x1 - 0.4047) * (x1 - 0.4047))) + (8.0 * ((x2 - 0.8828) * (x2 - 0.8828)))) + (0.05 * ((x3 - 0.8732) * (x3 - 0.8732)))) + (10.0 * ((x4 - 0.5743) * (x4 - 0.5743)))) + (0.1 * ((x5 - 0.1091) * (x5 - 0.1091)))) + (14.0 * ((x6 - 0.0381) * (x6 - 0.0381)));
	double exp1 = exp(-e1);
	double exp2 = exp(-e2);
	double exp3 = exp(-e3);
	double exp4 = exp(-e4);
	return -((((1.0 * exp1) + (1.2 * exp2)) + (3.0 * exp3)) + (3.2 * exp4));
}

double ex8(double x1, double x2, double x3, double x4, double x5, double x6) {
	return ((((x2 * x5) + (x3 * x6)) - (x2 * x3)) - (x5 * x6)) + (x1 * (((((-x1 + x2) + x3) - x4) + x5) + x6));
}

double ex9(double x1, double x2, double x3, double x4) {
	return (((((((x1 * x4) * (((-x1 + x2) + x3) - x4)) + (x2 * (((x1 - x2) + x3) + x4))) + (x3 * (((x1 + x2) - x3) + x4))) - ((x2 * x3) * x4)) - (x1 * x3)) - (x1 * x2)) - x4;
}

double ex10(double x1, double x2, double x3, double x4, double x5, double x6) {
	return (((((((x1 * x4) * (((((-x1 + x2) + x3) - x4) + x5) + x6)) + ((x2 * x5) * (((((x1 - x2) + x3) + x4) - x5) + x6))) + ((x3 * x6) * (((((x1 + x2) - x3) + x4) + x5) - x6))) - ((x2 * x3) * x4)) - ((x1 * x3) * x5)) - ((x1 * x2) * x6)) - ((x4 * x5) * x6);
}

