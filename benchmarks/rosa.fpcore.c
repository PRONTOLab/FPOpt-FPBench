#include <fenv.h>
#include <math.h>
#include <stdint.h>
#define TRUE 1
#define FALSE 0

double ex0(double u, double v, double T) {
	double t1 = 331.4 + (0.6 * T);
	return (-t1 * v) / ((t1 + u) * (t1 + u));
}

double ex1(double u, double v, double T) {
	double t1 = 331.4 + (0.6 * T);
	return (-t1 * v) / ((t1 + u) * (t1 + u));
}

double ex2(double u, double v, double T) {
	double t1 = 331.4 + (0.6 * T);
	return (-t1 * v) / ((t1 + u) * (t1 + u));
}

double ex3(double x1, double x2, double x3) {
	return ((-(x1 * x2) - ((2.0 * x2) * x3)) - x1) - x3;
}

double ex4(double x1, double x2, double x3) {
	return ((((((2.0 * x1) * x2) * x3) + ((3.0 * x3) * x3)) - (((x2 * x1) * x2) * x3)) + ((3.0 * x3) * x3)) - x2;
}

double ex5(double x1, double x2) {
	double t = (((3.0 * x1) * x1) + (2.0 * x2)) - x1;
	double t_42_ = (((3.0 * x1) * x1) - (2.0 * x2)) - x1;
	double d = (x1 * x1) + 1.0;
	double s = t / d;
	double s_42_ = t_42_ / d;
	return x1 + (((((((((2.0 * x1) * s) * (s - 3.0)) + ((x1 * x1) * ((4.0 * s) - 6.0))) * d) + (((3.0 * x1) * x1) * s)) + ((x1 * x1) * x1)) + x1) + (3.0 * s_42_));
}

double ex6(double v, double w, double r) {
	return ((3.0 + (2.0 / (r * r))) - (((0.125 * (3.0 - (2.0 * v))) * (((w * w) * r) * r)) / (1.0 - v))) - 4.5;
}

double ex7(double v, double w, double r) {
	return ((6.0 * v) - (((0.5 * v) * (((w * w) * r) * r)) / (1.0 - v))) - 2.5;
}

double ex8(double v, double w, double r) {
	return ((3.0 - (2.0 / (r * r))) - (((0.125 * (1.0 + (2.0 * v))) * (((w * w) * r) * r)) / (1.0 - v))) - 0.5;
}

double ex9(double x) {
	double r = 4.0;
	double K = 1.11;
	return (r * x) / (1.0 + (x / K));
}

double ex10(double x) {
	double r = 4.0;
	double K = 1.11;
	return ((r * x) * x) / (1.0 + ((x / K) * (x / K)));
}

double ex11(double v) {
	double p = 35000000.0;
	double a = 0.401;
	double b = 4.27e-5;
	double t = 300.0;
	double n = 1000.0;
	double k = 1.3806503e-23;
	return ((p + ((a * (n / v)) * (n / v))) * (v - (n * b))) - ((k * n) * t);
}

double ex12(double x) {
	return ((x - (((x * x) * x) / 6.0)) + (((((x * x) * x) * x) * x) / 120.0)) - (((((((x * x) * x) * x) * x) * x) * x) / 5040.0);
}

double ex13(double x) {
	return (((1.0 + (0.5 * x)) - ((0.125 * x) * x)) + (((0.0625 * x) * x) * x)) - ((((0.0390625 * x) * x) * x) * x);
}

double ex14(double x) {
	return (0.954929658551372 * x) - (0.12900613773279798 * ((x * x) * x));
}

double ex15(double c) {
	double a = 3.0;
	double b = 3.5;
	double discr = (b * b) - ((a * c) * 4.0);
	double tmp_1;
	if (((b * b) - (a * c)) > 10.0) {
		double tmp_2;
		if (b > 0.0) {
			tmp_2 = (c * 2.0) / (-b - sqrt(discr));
		} else if (b < 0.0) {
			tmp_2 = (-b + sqrt(discr)) / (a * 2.0);
		} else {
			tmp_2 = (-b + sqrt(discr)) / (a * 2.0);
		}
		tmp_1 = tmp_2;
	} else {
		tmp_1 = (-b + sqrt(discr)) / (a * 2.0);
	}
	return tmp_1;
}

double ex16(double x) {
	double tmp;
	if (((x * x) - x) >= 0.0) {
		tmp = x / 10.0;
	} else {
		tmp = (x * x) + 2.0;
	}
	return tmp;
}

double ex17(double x) {
	double tmp;
	if (x < 1e-5) {
		tmp = 1.0 + (0.5 * x);
	} else {
		tmp = sqrt((1.0 + x));
	}
	return tmp;
}

double ex18(double x) {
	double tmp;
	if (x < 0.0001) {
		tmp = 1.0 + (0.5 * x);
	} else {
		tmp = sqrt((1.0 + x));
	}
	return tmp;
}

double ex19(double a, double b, double c) {
	double s = ((a + b) + c) / 2.0;
	return sqrt((((s * (s - a)) * (s - b)) * (s - c)));
}

double ex20(double a, double b, double c) {
	double s = ((a + b) + c) / 2.0;
	return sqrt((((s * (s - a)) * (s - b)) * (s - c)));
}

double ex21(double a, double b, double c) {
	double s = ((a + b) + c) / 2.0;
	return sqrt((((s * (s - a)) * (s - b)) * (s - c)));
}

double ex22(double a, double b, double c) {
	double s = ((a + b) + c) / 2.0;
	return sqrt((((s * (s - a)) * (s - b)) * (s - c)));
}

double ex23(double a, double b, double c) {
	double s = ((a + b) + c) / 2.0;
	return sqrt((((s * (s - a)) * (s - b)) * (s - c)));
}

double ex24(double a, double b, double c) {
	double s = ((a + b) + c) / 2.0;
	return sqrt((((s * (s - a)) * (s - b)) * (s - c)));
}

double ex25(double a, double b, double c) {
	double s = ((a + b) + c) / 2.0;
	return sqrt((((s * (s - a)) * (s - b)) * (s - c)));
}

double ex26(double a, double b, double c) {
	double s = ((a + b) + c) / 2.0;
	return sqrt((((s * (s - a)) * (s - b)) * (s - c)));
}

double ex27(double a, double b, double c) {
	double s = ((a + b) + c) / 2.0;
	return sqrt((((s * (s - a)) * (s - b)) * (s - c)));
}

double ex28(double a, double b, double c) {
	double s = ((a + b) + c) / 2.0;
	return sqrt((((s * (s - a)) * (s - b)) * (s - c)));
}

double ex29(double a, double b, double c) {
	double s = ((a + b) + c) / 2.0;
	return sqrt((((s * (s - a)) * (s - b)) * (s - c)));
}

double ex30(double a, double b, double c) {
	double s = ((a + b) + c) / 2.0;
	return sqrt((((s * (s - a)) * (s - b)) * (s - c)));
}

double ex31(double a, double b, double c) {
	double s = ((a + b) + c) / 2.0;
	return sqrt((((s * (s - a)) * (s - b)) * (s - c)));
}

double ex32(double u) {
	return -((u * u) * u) / 6.0;
}

double ex33(double a, double b, double c) {
	double tmp;
	if (a < b) {
		tmp = sqrt(((((c + (b + a)) * (a - (c - b))) * (a + (c - b))) * (c + (b - a)))) / 4.0;
	} else {
		tmp = sqrt(((((c + (a + b)) * (b - (c - a))) * (b + (c - a))) * (c + (a - b)))) / 4.0;
	}
	return tmp;
}

double ex34(double x0, double y0, double z0, double vx0, double vy0, double vz0) {
	double dt = 0.1;
	double solarMass = 39.47841760435743;
	double x = x0;
	double y = y0;
	double z = z0;
	double vx = vx0;
	double vy = vy0;
	double vz = vz0;
	double i = 0.0;
	int tmp = i < 100.0;
	while (tmp) {
		double distance = sqrt((((x * x) + (y * y)) + (z * z)));
		double mag = dt / ((distance * distance) * distance);
		double vxNew = vx - ((x * solarMass) * mag);
		double x_1 = x + (dt * vxNew);
		double distance_2 = sqrt((((x * x) + (y * y)) + (z * z)));
		double mag_3 = dt / ((distance_2 * distance_2) * distance_2);
		double vyNew = vy - ((y * solarMass) * mag_3);
		double y_4 = y + (dt * vyNew);
		double distance_5 = sqrt((((x * x) + (y * y)) + (z * z)));
		double mag_6 = dt / ((distance_5 * distance_5) * distance_5);
		double vzNew = vz - ((z * solarMass) * mag_6);
		double z_7 = z + (dt * vzNew);
		double distance_8 = sqrt((((x * x) + (y * y)) + (z * z)));
		double mag_9 = dt / ((distance_8 * distance_8) * distance_8);
		double vx_10 = vx - ((x * solarMass) * mag_9);
		double distance_11 = sqrt((((x * x) + (y * y)) + (z * z)));
		double mag_12 = dt / ((distance_11 * distance_11) * distance_11);
		double vy_13 = vy - ((y * solarMass) * mag_12);
		double distance_14 = sqrt((((x * x) + (y * y)) + (z * z)));
		double mag_15 = dt / ((distance_14 * distance_14) * distance_14);
		double vz_16 = vz - ((z * solarMass) * mag_15);
		double i_17 = i + 1.0;
		x = x_1;
		y = y_4;
		z = z_7;
		vx = vx_10;
		vy = vy_13;
		vz = vz_16;
		i = i_17;
		tmp = i < 100.0;
	}
	return x;
}

double ex35(double t0, double w0, double N) {
	double h = 0.01;
	double L = 2.0;
	double m = 1.5;
	double g = 9.80665;
	double t = t0;
	double w = w0;
	double n = 0.0;
	int tmp = n < N;
	while (tmp) {
		double k1w = (-g / L) * sin(t);
		double k2t = w + ((h / 2.0) * k1w);
		double t_1 = t + (h * k2t);
		double k2w = (-g / L) * sin((t + ((h / 2.0) * w)));
		double w_2 = w + (h * k2w);
		double n_3 = n + 1.0;
		t = t_1;
		w = w_2;
		n = n_3;
		tmp = n < N;
	}
	return t;
}
