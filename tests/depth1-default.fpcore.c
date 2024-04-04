#include <fenv.h>
#include <math.h>
#include <stdint.h>
#define TRUE 1
#define FALSE 0

double ex0() {
	return -1.0;
}

double ex1(double arg1) {
	return -arg1;
}

double ex2() {
	return 1.0 + 1.0;
}

double ex3(double arg1) {
	return arg1 + 1.0;
}

double ex4(double arg1) {
	return 1.0 + arg1;
}

double ex5(double arg1) {
	return arg1 + arg1;
}

double ex6(double arg1, double arg2) {
	return arg1 + arg2;
}

double ex7(double arg1, double arg2) {
	return arg2 + arg1;
}

double ex8() {
	return 1.0 - 1.0;
}

double ex9(double arg1) {
	return arg1 - 1.0;
}

double ex10(double arg1) {
	return 1.0 - arg1;
}

double ex11(double arg1) {
	return arg1 - arg1;
}

double ex12(double arg1, double arg2) {
	return arg1 - arg2;
}

double ex13(double arg1, double arg2) {
	return arg2 - arg1;
}

double ex14() {
	return 1.0 * 1.0;
}

double ex15(double arg1) {
	return arg1 * 1.0;
}

double ex16(double arg1) {
	return 1.0 * arg1;
}

double ex17(double arg1) {
	return arg1 * arg1;
}

double ex18(double arg1, double arg2) {
	return arg1 * arg2;
}

double ex19(double arg1, double arg2) {
	return arg2 * arg1;
}

double ex20() {
	return 1.0 / 1.0;
}

double ex21(double arg1) {
	return arg1 / 1.0;
}

double ex22(double arg1) {
	return 1.0 / arg1;
}

double ex23(double arg1) {
	return arg1 / arg1;
}

double ex24(double arg1, double arg2) {
	return arg1 / arg2;
}

double ex25(double arg1, double arg2) {
	return arg2 / arg1;
}

double ex26() {
	return fabs(1.0);
}

double ex27(double arg1) {
	return fabs(arg1);
}

double ex28() {
	return fma(1.0, 1.0, 1.0);
}

double ex29(double arg1) {
	return fma(arg1, 1.0, 1.0);
}

double ex30(double arg1) {
	return fma(1.0, arg1, 1.0);
}

double ex31(double arg1) {
	return fma(1.0, 1.0, arg1);
}

double ex32(double arg1) {
	return fma(arg1, arg1, 1.0);
}

double ex33(double arg1) {
	return fma(arg1, 1.0, arg1);
}

double ex34(double arg1) {
	return fma(1.0, arg1, arg1);
}

double ex35(double arg1, double arg2) {
	return fma(arg1, arg2, 1.0);
}

double ex36(double arg1, double arg2) {
	return fma(arg2, arg1, 1.0);
}

double ex37(double arg1, double arg2) {
	return fma(arg1, 1.0, arg2);
}

double ex38(double arg1, double arg2) {
	return fma(1.0, arg1, arg2);
}

double ex39(double arg1, double arg2) {
	return fma(arg2, 1.0, arg1);
}

double ex40(double arg1, double arg2) {
	return fma(1.0, arg2, arg1);
}

double ex41(double arg1) {
	return fma(arg1, arg1, arg1);
}

double ex42(double arg1, double arg2) {
	return fma(arg1, arg1, arg2);
}

double ex43(double arg1, double arg2) {
	return fma(arg1, arg2, arg1);
}

double ex44(double arg1, double arg2) {
	return fma(arg2, arg1, arg1);
}

double ex45(double arg1, double arg2) {
	return fma(arg1, arg2, arg2);
}

double ex46(double arg1, double arg2) {
	return fma(arg2, arg1, arg2);
}

double ex47(double arg1, double arg2) {
	return fma(arg2, arg2, arg1);
}

double ex48(double arg1, double arg2, double arg3) {
	return fma(arg1, arg2, arg3);
}

double ex49(double arg1, double arg2, double arg3) {
	return fma(arg2, arg1, arg3);
}

double ex50(double arg1, double arg2, double arg3) {
	return fma(arg1, arg3, arg2);
}

double ex51(double arg1, double arg2, double arg3) {
	return fma(arg3, arg1, arg2);
}

double ex52(double arg1, double arg2, double arg3) {
	return fma(arg2, arg3, arg1);
}

double ex53(double arg1, double arg2, double arg3) {
	return fma(arg3, arg2, arg1);
}

double ex54() {
	return exp(1.0);
}

double ex55(double arg1) {
	return exp(arg1);
}

double ex56() {
	return exp2(1.0);
}

double ex57(double arg1) {
	return exp2(arg1);
}

double ex58() {
	return expm1(1.0);
}

double ex59(double arg1) {
	return expm1(arg1);
}

double ex60() {
	return log(1.0);
}

double ex61(double arg1) {
	return log(arg1);
}

double ex62() {
	return log10(1.0);
}

double ex63(double arg1) {
	return log10(arg1);
}

double ex64() {
	return log2(1.0);
}

double ex65(double arg1) {
	return log2(arg1);
}

double ex66() {
	return log1p(1.0);
}

double ex67(double arg1) {
	return log1p(arg1);
}

double ex68() {
	return pow(1.0, 1.0);
}

double ex69(double arg1) {
	return pow(arg1, 1.0);
}

double ex70(double arg1) {
	return pow(1.0, arg1);
}

double ex71(double arg1) {
	return pow(arg1, arg1);
}

double ex72(double arg1, double arg2) {
	return pow(arg1, arg2);
}

double ex73(double arg1, double arg2) {
	return pow(arg2, arg1);
}

double ex74() {
	return sqrt(1.0);
}

double ex75(double arg1) {
	return sqrt(arg1);
}

double ex76() {
	return cbrt(1.0);
}

double ex77(double arg1) {
	return cbrt(arg1);
}

double ex78() {
	return hypot(1.0, 1.0);
}

double ex79(double arg1) {
	return hypot(arg1, 1.0);
}

double ex80(double arg1) {
	return hypot(1.0, arg1);
}

double ex81(double arg1) {
	return hypot(arg1, arg1);
}

double ex82(double arg1, double arg2) {
	return hypot(arg1, arg2);
}

double ex83(double arg1, double arg2) {
	return hypot(arg2, arg1);
}

double ex84() {
	return sin(1.0);
}

double ex85(double arg1) {
	return sin(arg1);
}

double ex86() {
	return cos(1.0);
}

double ex87(double arg1) {
	return cos(arg1);
}

double ex88() {
	return tan(1.0);
}

double ex89(double arg1) {
	return tan(arg1);
}

double ex90() {
	return asin(1.0);
}

double ex91(double arg1) {
	return asin(arg1);
}

double ex92() {
	return acos(1.0);
}

double ex93(double arg1) {
	return acos(arg1);
}

double ex94() {
	return atan(1.0);
}

double ex95(double arg1) {
	return atan(arg1);
}

double ex96() {
	return atan2(1.0, 1.0);
}

double ex97(double arg1) {
	return atan2(arg1, 1.0);
}

double ex98(double arg1) {
	return atan2(1.0, arg1);
}

double ex99(double arg1) {
	return atan2(arg1, arg1);
}

double ex100(double arg1, double arg2) {
	return atan2(arg1, arg2);
}

double ex101(double arg1, double arg2) {
	return atan2(arg2, arg1);
}

double ex102() {
	return sinh(1.0);
}

double ex103(double arg1) {
	return sinh(arg1);
}

double ex104() {
	return cosh(1.0);
}

double ex105(double arg1) {
	return cosh(arg1);
}

double ex106() {
	return tanh(1.0);
}

double ex107(double arg1) {
	return tanh(arg1);
}

double ex108() {
	return asinh(1.0);
}

double ex109(double arg1) {
	return asinh(arg1);
}

double ex110() {
	return acosh(1.0);
}

double ex111(double arg1) {
	return acosh(arg1);
}

double ex112() {
	return atanh(1.0);
}

double ex113(double arg1) {
	return atanh(arg1);
}

double ex114() {
	return erf(1.0);
}

double ex115(double arg1) {
	return erf(arg1);
}

double ex116() {
	return erfc(1.0);
}

double ex117(double arg1) {
	return erfc(arg1);
}

double ex118() {
	return tgamma(1.0);
}

double ex119(double arg1) {
	return tgamma(arg1);
}

double ex120() {
	return lgamma(1.0);
}

double ex121(double arg1) {
	return lgamma(arg1);
}

double ex122() {
	return ceil(1.0);
}

double ex123(double arg1) {
	return ceil(arg1);
}

double ex124() {
	return floor(1.0);
}

double ex125(double arg1) {
	return floor(arg1);
}

double ex126() {
	return fmod(1.0, 1.0);
}

double ex127(double arg1) {
	return fmod(arg1, 1.0);
}

double ex128(double arg1) {
	return fmod(1.0, arg1);
}

double ex129(double arg1) {
	return fmod(arg1, arg1);
}

double ex130(double arg1, double arg2) {
	return fmod(arg1, arg2);
}

double ex131(double arg1, double arg2) {
	return fmod(arg2, arg1);
}

double ex132() {
	return remainder(1.0, 1.0);
}

double ex133(double arg1) {
	return remainder(arg1, 1.0);
}

double ex134(double arg1) {
	return remainder(1.0, arg1);
}

double ex135(double arg1) {
	return remainder(arg1, arg1);
}

double ex136(double arg1, double arg2) {
	return remainder(arg1, arg2);
}

double ex137(double arg1, double arg2) {
	return remainder(arg2, arg1);
}

double ex138() {
	return fmax(1.0, 1.0);
}

double ex139(double arg1) {
	return fmax(arg1, 1.0);
}

double ex140(double arg1) {
	return fmax(1.0, arg1);
}

double ex141(double arg1) {
	return fmax(arg1, arg1);
}

double ex142(double arg1, double arg2) {
	return fmax(arg1, arg2);
}

double ex143(double arg1, double arg2) {
	return fmax(arg2, arg1);
}

double ex144() {
	return fmin(1.0, 1.0);
}

double ex145(double arg1) {
	return fmin(arg1, 1.0);
}

double ex146(double arg1) {
	return fmin(1.0, arg1);
}

double ex147(double arg1) {
	return fmin(arg1, arg1);
}

double ex148(double arg1, double arg2) {
	return fmin(arg1, arg2);
}

double ex149(double arg1, double arg2) {
	return fmin(arg2, arg1);
}

double ex150() {
	return fdim(1.0, 1.0);
}

double ex151(double arg1) {
	return fdim(arg1, 1.0);
}

double ex152(double arg1) {
	return fdim(1.0, arg1);
}

double ex153(double arg1) {
	return fdim(arg1, arg1);
}

double ex154(double arg1, double arg2) {
	return fdim(arg1, arg2);
}

double ex155(double arg1, double arg2) {
	return fdim(arg2, arg1);
}

double ex156() {
	return copysign(1.0, 1.0);
}

double ex157(double arg1) {
	return copysign(arg1, 1.0);
}

double ex158(double arg1) {
	return copysign(1.0, arg1);
}

double ex159(double arg1) {
	return copysign(arg1, arg1);
}

double ex160(double arg1, double arg2) {
	return copysign(arg1, arg2);
}

double ex161(double arg1, double arg2) {
	return copysign(arg2, arg1);
}

double ex162() {
	return trunc(1.0);
}

double ex163(double arg1) {
	return trunc(arg1);
}

double ex164() {
	return round(1.0);
}

double ex165(double arg1) {
	return round(arg1);
}

double ex166() {
	return nearbyint(1.0);
}

double ex167(double arg1) {
	return nearbyint(arg1);
}

double ex168() {
	return (double) 1.0;
}

double ex169(double arg1) {
	return (double) arg1;
}

