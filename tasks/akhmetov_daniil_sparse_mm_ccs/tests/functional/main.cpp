#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "akhmetov_daniil_sparse_mm_ccs/common/include/common.hpp"
#include "akhmetov_daniil_sparse_mm_ccs/mpi/include/ops_mpi.hpp"
#include "akhmetov_daniil_sparse_mm_ccs/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace akhmetov_daniil_sparse_mm_ccs {

using InType = ::akhmetov_daniil_sparse_mm_ccs::InType;
using OutType = ::akhmetov_daniil_sparse_mm_ccs::OutType;
using TestType = std::tuple<int, std::vector<std::vector<double>>, std::vector<std::vector<double>>,
                            std::vector<std::vector<double>>>;

class AkhmetovDaniilSparseMmCcsFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_sparse_mm_ccs";
  }

 protected:
  void SetUp() override {
    const auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    dense_a_ = std::get<1>(params);
    dense_b_ = std::get<2>(params);
    dense_expected_ = std::get<3>(params);

    if (dense_a_.empty() || dense_a_.front().empty()) {
      col_ptr_a_.assign(1, 0);
      values_a_.clear();
      row_indices_a_.clear();
    } else {
      ConvertDenseToCcs(dense_a_, values_a_, row_indices_a_, col_ptr_a_);
    }

    if (dense_b_.empty() || dense_b_.front().empty()) {
      col_ptr_b_.assign(1, 0);
      values_b_.clear();
      row_indices_b_.clear();
    } else {
      ConvertDenseToCcs(dense_b_, values_b_, row_indices_b_, col_ptr_b_);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    auto &values = output_data.values;
    auto &row_indices = output_data.row_indices;
    auto &col_ptr = output_data.col_ptr;
    const int rows = output_data.rows;
    const int cols = output_data.cols;

    if (col_ptr.empty()) {
      return false;
    }
    if (col_ptr[0] != 0) {
      return false;
    }
    if (values.size() != row_indices.size()) {
      return false;
    }
    for (std::size_t i = 0; i + 1 < col_ptr.size(); ++i) {
      if (col_ptr[i] > col_ptr[i + 1]) {
        return false;
      }
    }
    if (static_cast<std::size_t>(col_ptr.back()) != values.size()) {
      return false;
    }

    for (const int r : row_indices) {
      if (r < 0 || r >= rows) {
        return false;
      }
    }

    std::vector<std::vector<double>> dense_result;

    if (!ConvertCcsToDense(values, row_indices, col_ptr, rows, cols, dense_result)) {
      return false;
    }

    if (dense_expected_.empty()) {
      return dense_result.empty();
    }

    if (dense_result.size() != dense_expected_.size()) {
      return false;
    }
    if (!dense_result.empty() && dense_result.front().size() != dense_expected_.front().size()) {
      return false;
    }

    constexpr double kTolerance = 1e-10;
    for (std::size_t i = 0; i < dense_expected_.size(); ++i) {
      for (std::size_t j = 0; j < dense_expected_[i].size(); ++j) {
        double diff = std::abs(dense_result[i][j] - dense_expected_[i][j]);
        if (diff > kTolerance) {
          return false;
        }
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    SparseMatrixCCS matrix_a;
    matrix_a.rows = static_cast<int>(dense_a_.size());
    matrix_a.cols = dense_a_.empty() ? 0 : static_cast<int>(dense_a_.front().size());
    matrix_a.values = values_a_;
    matrix_a.row_indices = row_indices_a_;
    matrix_a.col_ptr = col_ptr_a_;

    SparseMatrixCCS matrix_b;
    matrix_b.rows = static_cast<int>(dense_b_.size());
    matrix_b.cols = dense_b_.empty() ? 0 : static_cast<int>(dense_b_.front().size());
    matrix_b.values = values_b_;
    matrix_b.row_indices = row_indices_b_;
    matrix_b.col_ptr = col_ptr_b_;

    InType input;
    input.push_back(std::move(matrix_a));
    input.push_back(std::move(matrix_b));
    return input;
  }

 private:
  static void ConvertDenseToCcs(const std::vector<std::vector<double>> &dense, std::vector<double> &values,
                                std::vector<int> &row_indices, std::vector<int> &col_ptr) {
    values.clear();
    row_indices.clear();
    col_ptr.clear();

    if (dense.empty() || dense.front().empty()) {
      col_ptr.push_back(0);
      return;
    }

    const int rows = static_cast<int>(dense.size());
    const int cols = static_cast<int>(dense.front().size());

    col_ptr.reserve(static_cast<std::size_t>(cols) + 1U);
    col_ptr.push_back(0);

    for (int j = 0; j < cols; ++j) {
      for (int i = 0; i < rows; ++i) {
        const double val = dense[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
        if (std::abs(val) > 1e-12) {
          values.push_back(val);
          row_indices.push_back(i);
        }
      }
      col_ptr.push_back(static_cast<int>(values.size()));
    }
  }

  static bool ConvertCcsToDense(const std::vector<double> &values, const std::vector<int> &row_indices,
                                const std::vector<int> &col_ptr, int rows, int cols,
                                std::vector<std::vector<double>> &dense) {
    if (rows < 0 || cols < 0) {
      return false;
    }

    if (cols + 1 != static_cast<int>(col_ptr.size())) {
      return false;
    }

    dense.assign(static_cast<std::size_t>(rows), std::vector<double>(static_cast<std::size_t>(cols), 0.0));

    for (int j = 0; j < cols; ++j) {
      const int start = col_ptr[j];
      const int end = col_ptr[j + 1];
      if (start < 0 || end < start || static_cast<std::size_t>(end) > values.size() ||
          static_cast<std::size_t>(end) > row_indices.size()) {
        return false;
      }

      for (int idx = start; idx < end; ++idx) {
        const int i = row_indices[static_cast<std::size_t>(idx)];
        if (i < 0 || i >= rows) {
          return false;
        }
        dense[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = values[static_cast<std::size_t>(idx)];
      }
    }

    return true;
  }

  std::vector<std::vector<double>> dense_a_;
  std::vector<std::vector<double>> dense_b_;
  std::vector<std::vector<double>> dense_expected_;

  std::vector<double> values_a_;
  std::vector<int> row_indices_a_;
  std::vector<int> col_ptr_a_;

  std::vector<double> values_b_;
  std::vector<int> row_indices_b_;
  std::vector<int> col_ptr_b_;
};

namespace {

TEST_P(AkhmetovDaniilSparseMmCcsFuncTests, FunctionalTests) {
  ExecuteTest(GetParam());
}

TEST_P(AkhmetovDaniilSparseMmCcsFuncTests, CoverageTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kFunctionalTests = {
    // 1. Простое умножение 2x2
    TestType{1, {{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}, {{19.0, 22.0}, {43.0, 50.0}}},

    // 2. Единичная матрица
    TestType{2, {{1.0, 0.0}, {0.0, 1.0}}, {{1.0, 2.0}, {3.0, 4.0}}, {{1.0, 2.0}, {3.0, 4.0}}},

    // 3. Разреженная диагональная матрица
    TestType{3,
             {{1.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 3.0}},
             {{4.0, 0.0, 0.0}, {0.0, 5.0, 0.0}, {0.0, 0.0, 6.0}},
             {{4.0, 0.0, 0.0}, {0.0, 10.0, 0.0}, {0.0, 0.0, 18.0}}},

    // 4. Нулевая матрица
    TestType{4, {{0.0, 0.0}, {0.0, 0.0}}, {{1.0, 2.0}, {3.0, 4.0}}, {{0.0, 0.0}, {0.0, 0.0}}},

    // 5. 2x3 на 3x2
    TestType{
        5, {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}, {{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}}, {{58.0, 64.0}, {139.0, 154.0}}},

    // 6. Разреженная с несколькими ненулевыми в столбце
    TestType{6,
             {{1.0, 0.0, 2.0}, {0.0, 3.0, 0.0}, {4.0, 0.0, 5.0}},
             {{6.0, 0.0}, {0.0, 7.0}, {8.0, 0.0}},
             {{22.0, 0.0}, {0.0, 21.0}, {64.0, 0.0}}},

    // 7. 1x1
    TestType{7, {{2.0}}, {{3.0}}, {{6.0}}},

    // 8. Вектор-строка на вектор-столбец
    TestType{8, {{1.0, 2.0, 3.0}}, {{4.0}, {5.0}, {6.0}}, {{32.0}}},

    // 9. Отрицательные числа
    TestType{9, {{-1.0, -2.0}, {-3.0, -4.0}}, {{1.0, 2.0}, {3.0, 4.0}}, {{-7.0, -10.0}, {-15.0, -22.0}}},

    // 10. Дробные числа
    TestType{10, {{0.5, 0.5}, {0.5, 0.5}}, {{2.0, 4.0}, {6.0, 8.0}}, {{4.0, 6.0}, {4.0, 6.0}}},
};

const std::array<TestType, 5> kCoverageTests = {
    // 11. Пустые матрицы
    TestType{11, {}, {}, {}},

    // 12. Один ненулевой элемент (произведение даст нули)
    TestType{12, {{0.0, 0.0}, {0.0, 1.0}}, {{2.0, 0.0}, {0.0, 0.0}}, {{0.0, 0.0}, {0.0, 0.0}}},

    // 13. Большие разреженные матрицы
    TestType{13,
             {{1.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 2.0, 0.0}, {0.0, 3.0, 0.0, 0.0}},
             {{0.0, 1.0}, {0.0, 0.0}, {0.0, 0.0}, {4.0, 0.0}},
             {{0.0, 1.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}},

    // 14. Умножение на единичную (симметричное)
    TestType{14, {{1.0, 2.0}, {3.0, 4.0}}, {{1.0, 0.0}, {0.0, 1.0}}, {{1.0, 2.0}, {3.0, 4.0}}},

    // 15. Маленькие дробные числа
    TestType{15, {{0.001, 0.002}, {0.003, 0.004}}, {{1000.0, 2000.0}, {3000.0, 4000.0}}, {{7.0, 10.0}, {15.0, 22.0}}},
};

const auto kFunctionalTasksList = std::tuple_cat(ppc::util::AddFuncTask<SparseMatrixMultiplicationCCSMPI, InType>(
                                                     kFunctionalTests, PPC_SETTINGS_akhmetov_daniil_sparse_mm_ccs),
                                                 ppc::util::AddFuncTask<SparseMatrixMultiplicationCCSSeq, InType>(
                                                     kFunctionalTests, PPC_SETTINGS_akhmetov_daniil_sparse_mm_ccs));

const auto kCoverageTasksList = std::tuple_cat(ppc::util::AddFuncTask<SparseMatrixMultiplicationCCSMPI, InType>(
                                                   kCoverageTests, PPC_SETTINGS_akhmetov_daniil_sparse_mm_ccs),
                                               ppc::util::AddFuncTask<SparseMatrixMultiplicationCCSSeq, InType>(
                                                   kCoverageTests, PPC_SETTINGS_akhmetov_daniil_sparse_mm_ccs));

inline const auto kFunctionalGtestValues = ppc::util::ExpandToValues(kFunctionalTasksList);
inline const auto kCoverageGtestValues = ppc::util::ExpandToValues(kCoverageTasksList);

inline const auto kPerfTestName =
    AkhmetovDaniilSparseMmCcsFuncTests::PrintFuncTestName<AkhmetovDaniilSparseMmCcsFuncTests>;

INSTANTIATE_TEST_SUITE_P(Functional, AkhmetovDaniilSparseMmCcsFuncTests, kFunctionalGtestValues, kPerfTestName);

INSTANTIATE_TEST_SUITE_P(Coverage, AkhmetovDaniilSparseMmCcsFuncTests, kCoverageGtestValues, kPerfTestName);

}  // namespace
}  // namespace akhmetov_daniil_sparse_mm_ccs
