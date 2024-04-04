#include <fenv.h>
#include <math.h>
#include <stdint.h>
#define TRUE 1
#define FALSE 0

float ex0() {
	return -1.0f;
}

float ex1(float arg1) {
	return -arg1;
}

float ex2() {
	return 1.0f + 1.0f;
}

float ex3(float arg1) {
	return arg1 + 1.0f;
}

float ex4(float arg1) {
	return 1.0f + arg1;
}

float ex5(float arg1) {
	return arg1 + arg1;
}

float ex6(float arg1, float arg2) {
	return arg1 + arg2;
}

float ex7(float arg1, float arg2) {
	return arg2 + arg1;
}

float ex8() {
	return 1.0f - 1.0f;
}

float ex9(float arg1) {
	return arg1 - 1.0f;
}

float ex10(float arg1) {
	return 1.0f - arg1;
}

float ex11(float arg1) {
	return arg1 - arg1;
}

float ex12(float arg1, float arg2) {
	return arg1 - arg2;
}

float ex13(float arg1, float arg2) {
	return arg2 - arg1;
}

float ex14() {
	return 1.0f * 1.0f;
}

float ex15(float arg1) {
	return arg1 * 1.0f;
}

float ex16(float arg1) {
	return 1.0f * arg1;
}

float ex17(float arg1) {
	return arg1 * arg1;
}

float ex18(float arg1, float arg2) {
	return arg1 * arg2;
}

float ex19(float arg1, float arg2) {
	return arg2 * arg1;
}

float ex20() {
	return 1.0f / 1.0f;
}

float ex21(float arg1) {
	return arg1 / 1.0f;
}

float ex22(float arg1) {
	return 1.0f / arg1;
}

float ex23(float arg1) {
	return arg1 / arg1;
}

float ex24(float arg1, float arg2) {
	return arg1 / arg2;
}

float ex25(float arg1, float arg2) {
	return arg2 / arg1;
}

float ex26() {
	return fabsf(1.0f);
}

float ex27(float arg1) {
	return fabsf(arg1);
}

float ex28() {
	return fmaf(1.0f, 1.0f, 1.0f);
}

float ex29(float arg1) {
	return fmaf(arg1, 1.0f, 1.0f);
}

float ex30(float arg1) {
	return fmaf(1.0f, arg1, 1.0f);
}

float ex31(float arg1) {
	return fmaf(1.0f, 1.0f, arg1);
}

float ex32(float arg1) {
	return fmaf(arg1, arg1, 1.0f);
}

float ex33(float arg1) {
	return fmaf(arg1, 1.0f, arg1);
}

float ex34(float arg1) {
	return fmaf(1.0f, arg1, arg1);
}

float ex35(float arg1, float arg2) {
	return fmaf(arg1, arg2, 1.0f);
}

float ex36(float arg1, float arg2) {
	return fmaf(arg2, arg1, 1.0f);
}

float ex37(float arg1, float arg2) {
	return fmaf(arg1, 1.0f, arg2);
}

float ex38(float arg1, float arg2) {
	return fmaf(1.0f, arg1, arg2);
}

float ex39(float arg1, float arg2) {
	return fmaf(arg2, 1.0f, arg1);
}

float ex40(float arg1, float arg2) {
	return fmaf(1.0f, arg2, arg1);
}

float ex41(float arg1) {
	return fmaf(arg1, arg1, arg1);
}

float ex42(float arg1, float arg2) {
	return fmaf(arg1, arg1, arg2);
}

float ex43(float arg1, float arg2) {
	return fmaf(arg1, arg2, arg1);
}

float ex44(float arg1, float arg2) {
	return fmaf(arg2, arg1, arg1);
}

float ex45(float arg1, float arg2) {
	return fmaf(arg1, arg2, arg2);
}

float ex46(float arg1, float arg2) {
	return fmaf(arg2, arg1, arg2);
}

float ex47(float arg1, float arg2) {
	return fmaf(arg2, arg2, arg1);
}

float ex48(float arg1, float arg2, float arg3) {
	return fmaf(arg1, arg2, arg3);
}

float ex49(float arg1, float arg2, float arg3) {
	return fmaf(arg2, arg1, arg3);
}

float ex50(float arg1, float arg2, float arg3) {
	return fmaf(arg1, arg3, arg2);
}

float ex51(float arg1, float arg2, float arg3) {
	return fmaf(arg3, arg1, arg2);
}

float ex52(float arg1, float arg2, float arg3) {
	return fmaf(arg2, arg3, arg1);
}

float ex53(float arg1, float arg2, float arg3) {
	return fmaf(arg3, arg2, arg1);
}

float ex54() {
	return expf(1.0f);
}

float ex55(float arg1) {
	return expf(arg1);
}

float ex56() {
	return exp2f(1.0f);
}

float ex57(float arg1) {
	return exp2f(arg1);
}

float ex58() {
	return expm1f(1.0f);
}

float ex59(float arg1) {
	return expm1f(arg1);
}

float ex60() {
	return logf(1.0f);
}

float ex61(float arg1) {
	return logf(arg1);
}

float ex62() {
	return log10f(1.0f);
}

float ex63(float arg1) {
	return log10f(arg1);
}

float ex64() {
	return log2f(1.0f);
}

float ex65(float arg1) {
	return log2f(arg1);
}

float ex66() {
	return log1pf(1.0f);
}

float ex67(float arg1) {
	return log1pf(arg1);
}

float ex68() {
	return powf(1.0f, 1.0f);
}

float ex69(float arg1) {
	return powf(arg1, 1.0f);
}

float ex70(float arg1) {
	return powf(1.0f, arg1);
}

float ex71(float arg1) {
	return powf(arg1, arg1);
}

float ex72(float arg1, float arg2) {
	return powf(arg1, arg2);
}

float ex73(float arg1, float arg2) {
	return powf(arg2, arg1);
}

float ex74() {
	return sqrtf(1.0f);
}

float ex75(float arg1) {
	return sqrtf(arg1);
}

float ex76() {
	return cbrtf(1.0f);
}

float ex77(float arg1) {
	return cbrtf(arg1);
}

float ex78() {
	return hypotf(1.0f, 1.0f);
}

float ex79(float arg1) {
	return hypotf(arg1, 1.0f);
}

float ex80(float arg1) {
	return hypotf(1.0f, arg1);
}

float ex81(float arg1) {
	return hypotf(arg1, arg1);
}

float ex82(float arg1, float arg2) {
	return hypotf(arg1, arg2);
}

float ex83(float arg1, float arg2) {
	return hypotf(arg2, arg1);
}

float ex84() {
	return sinf(1.0f);
}

float ex85(float arg1) {
	return sinf(arg1);
}

float ex86() {
	return cosf(1.0f);
}

float ex87(float arg1) {
	return cosf(arg1);
}

float ex88() {
	return tanf(1.0f);
}

float ex89(float arg1) {
	return tanf(arg1);
}

float ex90() {
	return asinf(1.0f);
}

float ex91(float arg1) {
	return asinf(arg1);
}

float ex92() {
	return acosf(1.0f);
}

float ex93(float arg1) {
	return acosf(arg1);
}

float ex94() {
	return atanf(1.0f);
}

float ex95(float arg1) {
	return atanf(arg1);
}

float ex96() {
	return atan2f(1.0f, 1.0f);
}

float ex97(float arg1) {
	return atan2f(arg1, 1.0f);
}

float ex98(float arg1) {
	return atan2f(1.0f, arg1);
}

float ex99(float arg1) {
	return atan2f(arg1, arg1);
}

float ex100(float arg1, float arg2) {
	return atan2f(arg1, arg2);
}

float ex101(float arg1, float arg2) {
	return atan2f(arg2, arg1);
}

float ex102() {
	return sinhf(1.0f);
}

float ex103(float arg1) {
	return sinhf(arg1);
}

float ex104() {
	return coshf(1.0f);
}

float ex105(float arg1) {
	return coshf(arg1);
}

float ex106() {
	return tanhf(1.0f);
}

float ex107(float arg1) {
	return tanhf(arg1);
}

float ex108() {
	return asinhf(1.0f);
}

float ex109(float arg1) {
	return asinhf(arg1);
}

float ex110() {
	return acoshf(1.0f);
}

float ex111(float arg1) {
	return acoshf(arg1);
}

float ex112() {
	return atanhf(1.0f);
}

float ex113(float arg1) {
	return atanhf(arg1);
}

float ex114() {
	return erff(1.0f);
}

float ex115(float arg1) {
	return erff(arg1);
}

float ex116() {
	return erfcf(1.0f);
}

float ex117(float arg1) {
	return erfcf(arg1);
}

float ex118() {
	return tgammaf(1.0f);
}

float ex119(float arg1) {
	return tgammaf(arg1);
}

float ex120() {
	return lgammaf(1.0f);
}

float ex121(float arg1) {
	return lgammaf(arg1);
}

float ex122() {
	return ceilf(1.0f);
}

float ex123(float arg1) {
	return ceilf(arg1);
}

float ex124() {
	return floorf(1.0f);
}

float ex125(float arg1) {
	return floorf(arg1);
}

float ex126() {
	return fmodf(1.0f, 1.0f);
}

float ex127(float arg1) {
	return fmodf(arg1, 1.0f);
}

float ex128(float arg1) {
	return fmodf(1.0f, arg1);
}

float ex129(float arg1) {
	return fmodf(arg1, arg1);
}

float ex130(float arg1, float arg2) {
	return fmodf(arg1, arg2);
}

float ex131(float arg1, float arg2) {
	return fmodf(arg2, arg1);
}

float ex132() {
	return remainderf(1.0f, 1.0f);
}

float ex133(float arg1) {
	return remainderf(arg1, 1.0f);
}

float ex134(float arg1) {
	return remainderf(1.0f, arg1);
}

float ex135(float arg1) {
	return remainderf(arg1, arg1);
}

float ex136(float arg1, float arg2) {
	return remainderf(arg1, arg2);
}

float ex137(float arg1, float arg2) {
	return remainderf(arg2, arg1);
}

float ex138() {
	return fmaxf(1.0f, 1.0f);
}

float ex139(float arg1) {
	return fmaxf(arg1, 1.0f);
}

float ex140(float arg1) {
	return fmaxf(1.0f, arg1);
}

float ex141(float arg1) {
	return fmaxf(arg1, arg1);
}

float ex142(float arg1, float arg2) {
	return fmaxf(arg1, arg2);
}

float ex143(float arg1, float arg2) {
	return fmaxf(arg2, arg1);
}

float ex144() {
	return fminf(1.0f, 1.0f);
}

float ex145(float arg1) {
	return fminf(arg1, 1.0f);
}

float ex146(float arg1) {
	return fminf(1.0f, arg1);
}

float ex147(float arg1) {
	return fminf(arg1, arg1);
}

float ex148(float arg1, float arg2) {
	return fminf(arg1, arg2);
}

float ex149(float arg1, float arg2) {
	return fminf(arg2, arg1);
}

float ex150() {
	return fdimf(1.0f, 1.0f);
}

float ex151(float arg1) {
	return fdimf(arg1, 1.0f);
}

float ex152(float arg1) {
	return fdimf(1.0f, arg1);
}

float ex153(float arg1) {
	return fdimf(arg1, arg1);
}

float ex154(float arg1, float arg2) {
	return fdimf(arg1, arg2);
}

float ex155(float arg1, float arg2) {
	return fdimf(arg2, arg1);
}

float ex156() {
	return copysignf(1.0f, 1.0f);
}

float ex157(float arg1) {
	return copysignf(arg1, 1.0f);
}

float ex158(float arg1) {
	return copysignf(1.0f, arg1);
}

float ex159(float arg1) {
	return copysignf(arg1, arg1);
}

float ex160(float arg1, float arg2) {
	return copysignf(arg1, arg2);
}

float ex161(float arg1, float arg2) {
	return copysignf(arg2, arg1);
}

float ex162() {
	return truncf(1.0f);
}

float ex163(float arg1) {
	return truncf(arg1);
}

float ex164() {
	return roundf(1.0f);
}

float ex165(float arg1) {
	return roundf(arg1);
}

float ex166() {
	return nearbyintf(1.0f);
}

float ex167(float arg1) {
	return nearbyintf(arg1);
}

float ex168() {
	return (float) 1.0f;
}

float ex169(float arg1) {
	return (float) arg1;
}

