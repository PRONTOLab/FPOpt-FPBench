#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// https://arxiv.org/pdf/1806.06403
double geomean(const std::vector<double> &dataset, double epsilon = 1e-5) {
  std::vector<double> dataset_nozeros;
  for (double x : dataset) {
    if (x > 0.0)
      dataset_nozeros.push_back(x);
  }

  if (dataset_nozeros.empty()) {
    return 0.0;
  }

  double sum_log = 0.0;
  for (double x : dataset_nozeros) {
    sum_log += std::log(x);
  }
  double geomeanNozeros = std::exp(sum_log / dataset_nozeros.size());

  double min_val =
      *std::min_element(dataset_nozeros.begin(), dataset_nozeros.end());
  double deltamin = 0.0;
  double deltamax = std::max(geomeanNozeros - min_val, 0.0);
  double delta = (deltamin + deltamax) / 2.0;
  double epsilon_threshold = epsilon * geomeanNozeros;

  auto compute_auxExp = [&](double d) -> double {
    double sum = 0.0;
    for (double x : dataset_nozeros) {
      sum += std::log(x + d);
    }
    return std::exp(sum / dataset_nozeros.size()) - d;
  };

  double auxExp = compute_auxExp(delta);

  while ((auxExp - geomeanNozeros) > epsilon_threshold) {
    if (auxExp < geomeanNozeros)
      deltamin = delta;
    else
      deltamax = delta;
    delta = (deltamin + deltamax) / 2.0;
    auxExp = compute_auxExp(delta);
  }

  double sum_log_all = 0.0;
  for (double x : dataset) {
    sum_log_all += std::log(x + delta);
  }
  double gmeanE = std::exp(sum_log_all / dataset.size()) - delta;

  assert(!std::isnan(gmeanE) && !std::isinf(gmeanE));
  return gmeanE;
}

class ValueInfo {
public:
  double minRes = std::numeric_limits<double>::max();
  double maxRes = std::numeric_limits<double>::lowest();
  std::vector<double> minOperands;
  std::vector<double> maxOperands;
  unsigned executions = 0;
  std::vector<double> loggedValues;

  void update(double res, const double *operands, unsigned numOperands) {
    minRes = std::min(minRes, res);
    maxRes = std::max(maxRes, res);
    if (minOperands.empty()) {
      minOperands.resize(numOperands, std::numeric_limits<double>::max());
      maxOperands.resize(numOperands, std::numeric_limits<double>::lowest());
    }
    for (unsigned i = 0; i < numOperands; ++i) {
      minOperands[i] = std::min(minOperands[i], operands[i]);
      maxOperands[i] = std::max(maxOperands[i], operands[i]);
    }
    ++executions;

    if (!std::isnan(res) && !std::isinf(res)) {
      loggedValues.push_back(std::fabs(res));
    }
  }

  double getGeomean() const {
    if (loggedValues.empty()) {
      return 0.0;
    }

    return geomean(loggedValues);
  }
};

class ErrorInfo {
public:
  double minErr = std::numeric_limits<double>::max();
  double maxErr = std::numeric_limits<double>::lowest();

  void update(double err) {
    minErr = std::min(minErr, err);
    maxErr = std::max(maxErr, err);
  }
};

class GradInfo {
public:
  std::vector<double> loggedValues;

  void update(double grad) {
    if (!std::isnan(grad) && !std::isinf(grad)) {
      loggedValues.push_back(std::fabs(grad));
    }
  }

  double getGeomean() const {
    if (loggedValues.empty()) {
      return 0.0;
    }

    return geomean(loggedValues);
  }
};

class Logger {
private:
  std::unordered_map<std::string, ValueInfo> valueInfo;
  std::unordered_map<std::string, ErrorInfo> errorInfo;
  std::unordered_map<std::string, GradInfo> gradInfo;

public:
  void updateValue(const std::string &id, double res, unsigned numOperands,
                   const double *operands) {
    auto &info = valueInfo.emplace(id, ValueInfo()).first->second;
    info.update(res, operands, numOperands);
  }

  void updateError(const std::string &id, double err) {
    auto &info = errorInfo.emplace(id, ErrorInfo()).first->second;
    info.update(err);
  }

  void updateGrad(const std::string &id, double grad) {
    auto &info = gradInfo.emplace(id, GradInfo()).first->second;
    info.update(grad);
  }

  void print() const {
    std::cout << std::scientific
              << std::setprecision(std::numeric_limits<double>::max_digits10);

    for (const auto &pair : valueInfo) {
      const auto &id = pair.first;
      const auto &info = pair.second;
      std::cout << "Value:" << id << "\n";
      std::cout << "\tMinRes = " << info.minRes << "\n";
      std::cout << "\tMaxRes = " << info.maxRes << "\n";
      std::cout << "\tExecutions = " << info.executions << "\n";
      std::cout << "\tGeometric Average = " << info.getGeomean() << "\n";
      for (unsigned i = 0; i < info.minOperands.size(); ++i) {
        std::cout << "\tOperand[" << i << "] = [" << info.minOperands[i] << ", "
                  << info.maxOperands[i] << "]\n";
      }
    }

    for (const auto &pair : gradInfo) {
      const auto &id = pair.first;
      const auto &info = pair.second;
      std::cout << "Grad:" << id << "\n";
      std::cout << "\tGrad = " << info.getGeomean() << "\n";
    }
  }
};

Logger *logger = nullptr;

void initializeLogger() { logger = new Logger(); }

void destroyLogger() {
  delete logger;
  logger = nullptr;
}

void printLogger() { logger->print(); }

void enzymeLogError(const char *id, double err) {
  assert(logger && "Logger is not initialized");
  logger->updateError(id, err);
}

void enzymeLogGrad(const char *id, double grad) {
  assert(logger && "Logger is not initialized");
  logger->updateGrad(id, grad);
}

void enzymeLogValue(const char *id, double res, unsigned numOperands,
                    double *operands) {
  assert(logger && "Logger is not initialized");
  logger->updateValue(id, res, numOperands, operands);
}
