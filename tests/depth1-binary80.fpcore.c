#include <fenv.h>
#include <math.h>
#include <stdint.h>
#define TRUE 1
#define FALSE 0

long double ex0() {
	return 1.0l + 1.0l;
}

long double ex1(long double arg1) {
	return arg1 + 1.0l;
}

long double ex2(long double arg1) {
	return 1.0l + arg1;
}

long double ex3(long double arg1) {
	return arg1 + arg1;
}

long double ex4(long double arg1, long double arg2) {
	return arg1 + arg2;
}

long double ex5(long double arg1, long double arg2) {
	return arg2 + arg1;
}

long double ex6() {
	return 1.0l - 1.0l;
}

long double ex7(long double arg1) {
	return arg1 - 1.0l;
}

long double ex8(long double arg1) {
	return 1.0l - arg1;
}

long double ex9(long double arg1) {
	return arg1 - arg1;
}

long double ex10(long double arg1, long double arg2) {
	return arg1 - arg2;
}

long double ex11(long double arg1, long double arg2) {
	return arg2 - arg1;
}

long double ex12() {
	return 1.0l * 1.0l;
}

long double ex13(long double arg1) {
	return arg1 * 1.0l;
}

long double ex14(long double arg1) {
	return 1.0l * arg1;
}

long double ex15(long double arg1) {
	return arg1 * arg1;
}

long double ex16(long double arg1, long double arg2) {
	return arg1 * arg2;
}

long double ex17(long double arg1, long double arg2) {
	return arg2 * arg1;
}

long double ex18() {
	return 1.0l / 1.0l;
}

long double ex19(long double arg1) {
	return arg1 / 1.0l;
}

long double ex20(long double arg1) {
	return 1.0l / arg1;
}

long double ex21(long double arg1) {
	return arg1 / arg1;
}

long double ex22(long double arg1, long double arg2) {
	return arg1 / arg2;
}

long double ex23(long double arg1, long double arg2) {
	return arg2 / arg1;
}

long double ex24() {
	return fabsl(1.0l);
}

long double ex25(long double arg1) {
	return fabsl(arg1);
}

long double ex26() {
	return fmal(1.0l, 1.0l, 1.0l);
}

long double ex27(long double arg1) {
	return fmal(arg1, 1.0l, 1.0l);
}

long double ex28(long double arg1) {
	return fmal(1.0l, arg1, 1.0l);
}

long double ex29(long double arg1) {
	return fmal(1.0l, 1.0l, arg1);
}

long double ex30(long double arg1) {
	return fmal(arg1, arg1, 1.0l);
}

long double ex31(long double arg1) {
	return fmal(arg1, 1.0l, arg1);
}

long double ex32(long double arg1) {
	return fmal(1.0l, arg1, arg1);
}

long double ex33(long double arg1, long double arg2) {
	return fmal(arg1, arg2, 1.0l);
}

long double ex34(long double arg1, long double arg2) {
	return fmal(arg2, arg1, 1.0l);
}

long double ex35(long double arg1, long double arg2) {
	return fmal(arg1, 1.0l, arg2);
}

long double ex36(long double arg1, long double arg2) {
	return fmal(1.0l, arg1, arg2);
}

long double ex37(long double arg1, long double arg2) {
	return fmal(arg2, 1.0l, arg1);
}

long double ex38(long double arg1, long double arg2) {
	return fmal(1.0l, arg2, arg1);
}

long double ex39(long double arg1) {
	return fmal(arg1, arg1, arg1);
}

long double ex40(long double arg1, long double arg2) {
	return fmal(arg1, arg1, arg2);
}

long double ex41(long double arg1, long double arg2) {
	return fmal(arg1, arg2, arg1);
}

long double ex42(long double arg1, long double arg2) {
	return fmal(arg2, arg1, arg1);
}

long double ex43(long double arg1, long double arg2) {
	return fmal(arg1, arg2, arg2);
}

long double ex44(long double arg1, long double arg2) {
	return fmal(arg2, arg1, arg2);
}

long double ex45(long double arg1, long double arg2) {
	return fmal(arg2, arg2, arg1);
}

long double ex46(long double arg1, long double arg2, long double arg3) {
	return fmal(arg1, arg2, arg3);
}

long double ex47(long double arg1, long double arg2, long double arg3) {
	return fmal(arg2, arg1, arg3);
}

long double ex48(long double arg1, long double arg2, long double arg3) {
	return fmal(arg1, arg3, arg2);
}

long double ex49(long double arg1, long double arg2, long double arg3) {
	return fmal(arg3, arg1, arg2);
}

long double ex50(long double arg1, long double arg2, long double arg3) {
	return fmal(arg2, arg3, arg1);
}

long double ex51(long double arg1, long double arg2, long double arg3) {
	return fmal(arg3, arg2, arg1);
}

long double ex52() {
	return expl(1.0l);
}

long double ex53(long double arg1) {
	return expl(arg1);
}

long double ex54() {
	return exp2l(1.0l);
}

long double ex55(long double arg1) {
	return exp2l(arg1);
}

long double ex56() {
	return expm1l(1.0l);
}

long double ex57(long double arg1) {
	return expm1l(arg1);
}

long double ex58() {
	return logl(1.0l);
}

long double ex59(long double arg1) {
	return logl(arg1);
}

long double ex60() {
	return log10l(1.0l);
}

long double ex61(long double arg1) {
	return log10l(arg1);
}

long double ex62() {
	return log2l(1.0l);
}

long double ex63(long double arg1) {
	return log2l(arg1);
}

long double ex64() {
	return log1pl(1.0l);
}

long double ex65(long double arg1) {
	return log1pl(arg1);
}

long double ex66() {
	return powl(1.0l, 1.0l);
}

long double ex67(long double arg1) {
	return powl(arg1, 1.0l);
}

long double ex68(long double arg1) {
	return powl(1.0l, arg1);
}

long double ex69(long double arg1) {
	return powl(arg1, arg1);
}

long double ex70(long double arg1, long double arg2) {
	return powl(arg1, arg2);
}

long double ex71(long double arg1, long double arg2) {
	return powl(arg2, arg1);
}

long double ex72() {
	return sqrtl(1.0l);
}

long double ex73(long double arg1) {
	return sqrtl(arg1);
}

long double ex74() {
	return cbrtl(1.0l);
}

long double ex75(long double arg1) {
	return cbrtl(arg1);
}

long double ex76() {
	return hypotl(1.0l, 1.0l);
}

long double ex77(long double arg1) {
	return hypotl(arg1, 1.0l);
}

long double ex78(long double arg1) {
	return hypotl(1.0l, arg1);
}

long double ex79(long double arg1) {
	return hypotl(arg1, arg1);
}

long double ex80(long double arg1, long double arg2) {
	return hypotl(arg1, arg2);
}

long double ex81(long double arg1, long double arg2) {
	return hypotl(arg2, arg1);
}

long double ex82() {
	return sinl(1.0l);
}

long double ex83(long double arg1) {
	return sinl(arg1);
}

long double ex84() {
	return cosl(1.0l);
}

long double ex85(long double arg1) {
	return cosl(arg1);
}

long double ex86() {
	return tanl(1.0l);
}

long double ex87(long double arg1) {
	return tanl(arg1);
}

long double ex88() {
	return asinl(1.0l);
}

long double ex89(long double arg1) {
	return asinl(arg1);
}

long double ex90() {
	return acosl(1.0l);
}

long double ex91(long double arg1) {
	return acosl(arg1);
}

long double ex92() {
	return atanl(1.0l);
}

long double ex93(long double arg1) {
	return atanl(arg1);
}

long double ex94() {
	return atan2l(1.0l, 1.0l);
}

long double ex95(long double arg1) {
	return atan2l(arg1, 1.0l);
}

long double ex96(long double arg1) {
	return atan2l(1.0l, arg1);
}

long double ex97(long double arg1) {
	return atan2l(arg1, arg1);
}

long double ex98(long double arg1, long double arg2) {
	return atan2l(arg1, arg2);
}

long double ex99(long double arg1, long double arg2) {
	return atan2l(arg2, arg1);
}

long double ex100() {
	return sinhl(1.0l);
}

long double ex101(long double arg1) {
	return sinhl(arg1);
}

long double ex102() {
	return coshl(1.0l);
}

long double ex103(long double arg1) {
	return coshl(arg1);
}

long double ex104() {
	return tanhl(1.0l);
}

long double ex105(long double arg1) {
	return tanhl(arg1);
}

long double ex106() {
	return asinhl(1.0l);
}

long double ex107(long double arg1) {
	return asinhl(arg1);
}

long double ex108() {
	return acoshl(1.0l);
}

long double ex109(long double arg1) {
	return acoshl(arg1);
}

long double ex110() {
	return atanhl(1.0l);
}

long double ex111(long double arg1) {
	return atanhl(arg1);
}

long double ex112() {
	return erfl(1.0l);
}

long double ex113(long double arg1) {
	return erfl(arg1);
}

long double ex114() {
	return erfcl(1.0l);
}

long double ex115(long double arg1) {
	return erfcl(arg1);
}

long double ex116() {
	return tgammal(1.0l);
}

long double ex117(long double arg1) {
	return tgammal(arg1);
}

long double ex118() {
	return lgammal(1.0l);
}

long double ex119(long double arg1) {
	return lgammal(arg1);
}

long double ex120() {
	return ceill(1.0l);
}

long double ex121(long double arg1) {
	return ceill(arg1);
}

long double ex122() {
	return floorl(1.0l);
}

long double ex123(long double arg1) {
	return floorl(arg1);
}

long double ex124() {
	return fmodl(1.0l, 1.0l);
}

long double ex125(long double arg1) {
	return fmodl(arg1, 1.0l);
}

long double ex126(long double arg1) {
	return fmodl(1.0l, arg1);
}

long double ex127(long double arg1) {
	return fmodl(arg1, arg1);
}

long double ex128(long double arg1, long double arg2) {
	return fmodl(arg1, arg2);
}

long double ex129(long double arg1, long double arg2) {
	return fmodl(arg2, arg1);
}

long double ex130() {
	return remainderl(1.0l, 1.0l);
}

long double ex131(long double arg1) {
	return remainderl(arg1, 1.0l);
}

long double ex132(long double arg1) {
	return remainderl(1.0l, arg1);
}

long double ex133(long double arg1) {
	return remainderl(arg1, arg1);
}

long double ex134(long double arg1, long double arg2) {
	return remainderl(arg1, arg2);
}

long double ex135(long double arg1, long double arg2) {
	return remainderl(arg2, arg1);
}

long double ex136() {
	return fmaxl(1.0l, 1.0l);
}

long double ex137(long double arg1) {
	return fmaxl(arg1, 1.0l);
}

long double ex138(long double arg1) {
	return fmaxl(1.0l, arg1);
}

long double ex139(long double arg1) {
	return fmaxl(arg1, arg1);
}

long double ex140(long double arg1, long double arg2) {
	return fmaxl(arg1, arg2);
}

long double ex141(long double arg1, long double arg2) {
	return fmaxl(arg2, arg1);
}

long double ex142() {
	return fminl(1.0l, 1.0l);
}

long double ex143(long double arg1) {
	return fminl(arg1, 1.0l);
}

long double ex144(long double arg1) {
	return fminl(1.0l, arg1);
}

long double ex145(long double arg1) {
	return fminl(arg1, arg1);
}

long double ex146(long double arg1, long double arg2) {
	return fminl(arg1, arg2);
}

long double ex147(long double arg1, long double arg2) {
	return fminl(arg2, arg1);
}

long double ex148() {
	return fdiml(1.0l, 1.0l);
}

long double ex149(long double arg1) {
	return fdiml(arg1, 1.0l);
}

long double ex150(long double arg1) {
	return fdiml(1.0l, arg1);
}

long double ex151(long double arg1) {
	return fdiml(arg1, arg1);
}

long double ex152(long double arg1, long double arg2) {
	return fdiml(arg1, arg2);
}

long double ex153(long double arg1, long double arg2) {
	return fdiml(arg2, arg1);
}

long double ex154() {
	return copysignl(1.0l, 1.0l);
}

long double ex155(long double arg1) {
	return copysignl(arg1, 1.0l);
}

long double ex156(long double arg1) {
	return copysignl(1.0l, arg1);
}

long double ex157(long double arg1) {
	return copysignl(arg1, arg1);
}

long double ex158(long double arg1, long double arg2) {
	return copysignl(arg1, arg2);
}

long double ex159(long double arg1, long double arg2) {
	return copysignl(arg2, arg1);
}

long double ex160() {
	return truncl(1.0l);
}

long double ex161(long double arg1) {
	return truncl(arg1);
}

long double ex162() {
	return roundl(1.0l);
}

long double ex163(long double arg1) {
	return roundl(arg1);
}

long double ex164() {
	return nearbyintl(1.0l);
}

long double ex165(long double arg1) {
	return nearbyintl(arg1);
}

long double ex166() {
	return (long double) 1.0l;
}

long double ex167(long double arg1) {
	return (long double) arg1;
}

long double ex168() {
	return -1.0l;
}

long double ex169(long double arg1) {
	return -arg1;
}

