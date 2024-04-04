#include <functional>
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>

constexpr unsigned INPUT_NUM = 1;

template <typename T,
          std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
__attribute__((noinline)) std::vector<T> inputs() {
  static auto v = []() {
    std::vector<T> a;
    for (unsigned i = 0; i < INPUT_NUM; i++) {
      a.push_back(((T)rand()) / 1000);
    }
    return a;
  }();
  return v;
}

template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
__attribute__((noinline)) std::vector<T> inputs() {
  static auto v = []() {
    std::vector<T> a;
    for (unsigned i = 0; i < INPUT_NUM; i++) {
      a.push_back(rand() % 100);
    }
    return a;
  }();
  return v;
}

template <typename t>
void run(std::function<t()> f, const char *name, std::string argstr) {
  std::cout << name << "(" << argstr << ") = " << std::setprecision(15) << f()
            << std::endl;
}

template <typename t, typename arg, typename... args>
void run(std::function<t(arg, args...)> f, const char *name,
         std::string argstr) {
  for (arg i : inputs<arg>()) {
    run(std::function<t(args...)>([=](args... a) { return f(i, a...); }), name,
        argstr + ", " + std::to_string(i));
  }
}

template <typename t, typename... args>
void run2(t (*f)(args...), const char *name) {
  run(std::function<t(args...)>(f), name, "");
  printf("\n");
}

#define RUN(arg) run2(arg, #arg);
