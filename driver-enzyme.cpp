#include <functional>
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>

#include <enzyme/fprt/fprt.h>

#define FROM 64
#define TO 32

constexpr unsigned INPUT_NUM = 1;
constexpr unsigned MAX_ARG_NUM = 50;

template <typename T,
          std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
std::vector<std::vector<T>> &inputs() {
  static auto v = []() {
    std::vector<std::vector<T>> v;
    for (unsigned i = 0; i < MAX_ARG_NUM; i++) {
      std::vector<T> a;
      for (unsigned i = 0; i < INPUT_NUM; i++) {
        a.push_back(((T)(rand())) / 1000);
      }
      v.push_back(a);
    }
    return v;
  }();
  return v;
}

template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
std::vector<std::vector<T>> &inputs() {
  static auto v = []() {
    std::vector<std::vector<T>> v;
    for (unsigned i = 0; i < MAX_ARG_NUM; i++) {
      std::vector<T> a;
      for (unsigned i = 0; i < INPUT_NUM; i++) {
        a.push_back(((T)rand()) % 100 + 1);
      }
      v.push_back(a);
    }
    return v;
  }();
  return v;
}

template <typename t>
void run(std::function<t()> f, const char *name, std::string argstr,
         unsigned i) {
  auto res = f();
  if (std::is_same<double, t>::value)
    res = __enzyme_expand_mem_value_d(res, FROM, TO);
  std::cout << name << "(" << (i == 0 ? "" : argstr.c_str() + 2)
            << ") = " << std::setprecision(15) << res << std::endl;
}

template <typename t, typename arg, typename... args>
void run(std::function<t(arg, args...)> f, const char *name, std::string argstr,
         unsigned argno) {
  for (arg i : inputs<arg>()[argno]) {
    run(std::function<t(args...)>([=](args... a) {
          arg ti = i;
          if (std::is_same<double, arg>::value)
            ti = __enzyme_truncate_mem_value_d(i, FROM, TO);
          return f(ti, a...);
        }),
        name, argstr + ", " + std::to_string(i), argno + 1);
  }
}

template <typename fty> fty *__enzyme_truncate_mem_func(fty *, int, int);

template <typename t, typename... args>
__attribute__((noinline)) void run2(t (*f)(args...), const char *name) {
  run(std::function<t(args...)>(f), name, "", 0);
  printf("\n");
}

void init() { srand(1); }
void __enzyme_fprt_delete_all();
void cleanup() { __enzyme_fprt_delete_all(); }

#define RUN(arg) run2(__enzyme_truncate_mem_func(arg, FROM, TO), #arg);
