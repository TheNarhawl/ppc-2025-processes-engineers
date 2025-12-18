#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <string>
#include <tuple>

#include "akhmetov_daniil_integration_monte_carlo/common/include/common.hpp"
#include "akhmetov_daniil_integration_monte_carlo/mpi/include/ops_mpi.hpp"
#include "akhmetov_daniil_integration_monte_carlo/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace akhmetov_daniil_integration_monte_carlo {

constexpr double kMyPi = std::numbers::pi;

class AkhmetovDaniilRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
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

 private:
  InType input_data_;
};

namespace {

TEST_P(AkhmetovDaniilRunFuncTestsProcesses, MonteCarloIntegration) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {
    std::make_tuple(std::make_tuple(0.0, 1.0, 100000, FuncType::kLinearFunc), "kLinearFunc_test1"),
    std::make_tuple(std::make_tuple(-1.0, 2.0, 150000, FuncType::kLinearFunc), "kLinearFunc_test2"),

    std::make_tuple(std::make_tuple(0.0, 2.0, 100000, FuncType::kQuadraticFunc), "kQuadraticFunc_test1"),
    std::make_tuple(std::make_tuple(-1.0, 3.0, 200000, FuncType::kQuadraticFunc), "kQuadraticFunc_test2"),

    std::make_tuple(std::make_tuple(0.0, kMyPi, 200000, FuncType::kSinFunc), "kSinFunc_test1"),
    std::make_tuple(std::make_tuple(0.0, 2.0 * kMyPi, 300000, FuncType::kSinFunc), "kSinFunc_test2"),

    std::make_tuple(std::make_tuple(0.0, 1.0, 100000, FuncType::kExpFunc), "kExpFunc_test1"),
    std::make_tuple(std::make_tuple(-1.0, 2.0, 200000, FuncType::kExpFunc), "kExpFunc_test2"),

    std::make_tuple(std::make_tuple(0.0, 5.0, 50000, FuncType::kConstFunc), "kConstFunc_test1"),
    std::make_tuple(std::make_tuple(-3.0, 3.0, 50000, FuncType::kConstFunc), "kConstFunc_test2")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<AkhmetovDaniilIntegrationMonteCarloMPI, InType>(
                                               kTestParam, PPC_SETTINGS_akhmetov_daniil_integration_monte_carlo),
                                           ppc::util::AddFuncTask<AkhmetovDaniilIntegrationMonteCarloSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_akhmetov_daniil_integration_monte_carlo));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = AkhmetovDaniilRunFuncTestsProcesses::PrintFuncTestName<AkhmetovDaniilRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(MonteCarloTests, AkhmetovDaniilRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace akhmetov_daniil_integration_monte_carlo
