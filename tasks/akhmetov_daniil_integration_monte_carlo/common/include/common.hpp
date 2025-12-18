#pragma once

#include <cmath>
#include <cstdint>
#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace akhmetov_daniil_integration_monte_carlo {

enum class FuncType : std::uint8_t {
  kLinearFunc = 0,     // 3x + 2
  kQuadraticFunc = 1,  // x² + 1
  kSinFunc = 2,        // sin(x)
  kExpFunc = 3,        // e^x
  kConstFunc = 4       // 5 (const)
};

class FunctionPair {
 public:
  static double Function(FuncType func_id, double x) {
    switch (func_id) {
      case FuncType::kLinearFunc:
        return (3.0 * x) + 2.0;
      case FuncType::kQuadraticFunc:
        return (x * x) + 1.0;
      case FuncType::kSinFunc:
        return std::sin(x);
      case FuncType::kExpFunc:
        return std::exp(x);
      case FuncType::kConstFunc:
        return 5.0;
      default:
        return 0.0;
    }
  }

  static double Integral(FuncType func_id, double x) {
    switch (func_id) {
      case FuncType::kLinearFunc:
        return (1.5 * x * x) + (2.0 * x);
      case FuncType::kQuadraticFunc:
        return (x * x * x / 3.0) + x;
      case FuncType::kSinFunc:
        return -std::cos(x);
      case FuncType::kExpFunc:
        return std::exp(x);
      case FuncType::kConstFunc:
        return 5.0 * x;
      default:
        return 0.0;
    }
  }
};

using InType = std::tuple<double, double, int, FuncType>;
using OutType = double;
using TestType = std::tuple<std::tuple<double, double, int, FuncType>, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace akhmetov_daniil_integration_monte_carlo
