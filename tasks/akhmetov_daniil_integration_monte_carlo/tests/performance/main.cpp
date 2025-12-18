#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <tuple>

#include "akhmetov_daniil_integration_monte_carlo/common/include/common.hpp"
#include "akhmetov_daniil_integration_monte_carlo/mpi/include/ops_mpi.hpp"
#include "akhmetov_daniil_integration_monte_carlo/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace akhmetov_daniil_integration_monte_carlo {

class AkhmetovDaniilIntegrationMonteCarloRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;

  void SetUp() override {
    input_data_ = std::make_tuple(0.0, 2.0, 10000000, FuncType::kQuadraticFunc);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto &[a, b, n, func_id] = input_data_;

    double exp_integral = 0.0;
    exp_integral = FunctionPair::Integral(func_id, b) - FunctionPair::Integral(func_id, a);

    double sredn = exp_integral / (b - a);
    double std_dev = (b - a) / std::sqrt(n) * std::max(std::abs(sredn), 1.0);
    double epsilon = std::max(10.0 * std_dev, 1e-2);
    return std::abs(output_data - exp_integral) <= epsilon;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(AkhmetovDaniilIntegrationMonteCarloRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, AkhmetovDaniilIntegrationMonteCarloMPI, AkhmetovDaniilIntegrationMonteCarloSEQ>(
        PPC_SETTINGS_akhmetov_daniil_integration_monte_carlo);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = AkhmetovDaniilIntegrationMonteCarloRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, AkhmetovDaniilIntegrationMonteCarloRunPerfTestProcesses, kGtestValues,
                         kPerfTestName);

}  // namespace akhmetov_daniil_integration_monte_carlo
