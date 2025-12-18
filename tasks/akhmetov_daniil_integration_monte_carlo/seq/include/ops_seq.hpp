#pragma once

#include "akhmetov_daniil_integration_monte_carlo/common/include/common.hpp"
#include "task/include/task.hpp"

namespace akhmetov_daniil_integration_monte_carlo {

class AkhmetovDaniilIntegrationMonteCarloSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit AkhmetovDaniilIntegrationMonteCarloSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  int point_count_{};
  double a_{};
  double b_{};
  FuncType func_id_{};
};

}  // namespace akhmetov_daniil_integration_monte_carlo
