#include <math.h>
#include <stdint.h>
#define TRUE 1
#define FALSE 0

// ## PRE v: 20, 20000
// ## PRE T: -30, 50
// ## PRE u: -100, 100
__attribute__((noinline))
double example(double u, double v, double T) {
  double t1 = 331.4 + (0.6 * T);
  return (-t1 * v) / ((t1 + u) * (t1 + u));
}