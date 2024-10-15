#include <math.h>
#include <stdint.h>
#define TRUE 1
#define FALSE 0

// ## PRE c: 1, 9
// ## PRE a: 1, 9
// ## PRE b: 1, 9
__attribute__((noinline))
double example(double a, double b, double c) {
  double tmp;
  if (a < b) {
    tmp = sqrt(((((c + (b + a)) * (a - (c - b))) * (a + (c - b))) *
                (c + (b - a)))) /
          4.0;
  } else {
    tmp = sqrt(((((c + (a + b)) * (b - (c - a))) * (b + (c - a))) *
                (c + (a - b)))) /
          4.0;
  }
  return tmp;
}
