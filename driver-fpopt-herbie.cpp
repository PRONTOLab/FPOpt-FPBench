#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

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

  std::cout << name << "(" << (i == 0 ? "" : argstr.c_str() + 2)
            << ") = " << std::setprecision(15) << f() << std::endl;
}

template <typename t, typename arg, typename... args>
void run(std::function<t(arg, args...)> f, const char *name, std::string argstr,
         unsigned argno) {
  for (arg i : inputs<arg>()[argno]) {
    run(std::function<t(args...)>([=](args... a) { return f(i, a...); }), name,
        argstr + ", " + std::to_string(i), argno + 1);
  }
}

template <typename t, typename... args>
__attribute__((noinline)) void run2(t (*f)(args...), const char *name) {
  std::cout << "\n### Results for " << name << " ###\n";
  run(std::function<t(args...)>(f), name, "", 0);
  std::cout << "### End of results for " << name << " ###\n\n";
}

void init() { srand(1); }
void cleanup() {}

#define RUN(arg) run2(arg, #arg)
