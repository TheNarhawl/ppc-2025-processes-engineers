#include "akhmetov_daniil_integration_monte_carlo/seq/include/ops_seq.hpp"

#include <cmath>

#include "akhmetov_daniil_integration_monte_carlo/common/include/common.hpp"

namespace akhmetov_daniil_integration_monte_carlo {

AkhmetovDaniilIntegrationMonteCarloSEQ::AkhmetovDaniilIntegrationMonteCarloSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool AkhmetovDaniilIntegrationMonteCarloSEQ::ValidationImpl() {
  const auto &[a, b, n, func_id] = GetInput();

  return (a < b) && (n > 0) && (func_id >= FuncType::kLinearFunc) && (func_id <= FuncType::kConstFunc);
}

bool AkhmetovDaniilIntegrationMonteCarloSEQ::PreProcessingImpl() {
  const auto &[a, b, n, func_id] = GetInput();
  a_ = a;
  b_ = b;
  point_count_ = n;
  func_id_ = func_id;

  return true;
}

bool AkhmetovDaniilIntegrationMonteCarloSEQ::RunImpl() {
  const double magic_constant = 0.75487766624669276;
  double current = 0.5;

  double sum = 0.0;
  for (int i = 0; i < point_count_; ++i) {
    current += magic_constant;
    if (current >= 1.0) {
      current -= 1.0;
    }

    double x = a_ + ((b_ - a_) * current);

    double fx = FunctionPair::Function(func_id_, x);
    sum += fx;
  }

  double average = sum / static_cast<double>(point_count_);
  double integral = (b_ - a_) * average;
  GetOutput() = integral;

  return true;
}

bool AkhmetovDaniilIntegrationMonteCarloSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace akhmetov_daniil_integration_monte_carlo
