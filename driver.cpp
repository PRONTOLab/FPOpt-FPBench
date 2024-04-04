#include <stdio.h>
#include <string>
#include <functional>
#include <vector>

template<typename T, std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
std::vector<T> inputs() {
    return {-5, -3.14, -2.7, -1, -0.3, -.01, 0, .01, 0.3, 1, 2.7, 3.14, 5};
}

template<typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
std::vector<T> inputs() {
    return {-5, -3, -1, 0, 1, 3, 5};
}

template <typename t>
void run(std::function<t()> f, const char*name, std::string argstr) {
    printf("%e=%s(%s)\n", f(), name, argstr.c_str());
}

template <typename t, typename arg, typename... args>
void run(std::function<t(arg, args...)> f, const char*name, std::string argstr) {
    for (arg i : inputs<arg>()) {
        run(std::function<t(args...)>([=](args... a){ return f(i, a...); }), name, argstr+", "+std::to_string(i));
    }
}

template <typename t, typename... args>
void run2(t (*f)(args...), const char*name) {
    run(std::function<t(args...)>(f), name, "");
    printf("\n\n\n");
}

#define RUN(arg) run2(arg, #arg);
int main() {
    RUN(ex0)
    RUN(ex1)
    RUN(ex2)
}
