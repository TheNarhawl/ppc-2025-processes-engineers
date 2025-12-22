// tests/functional/main.cpp
#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <tuple>
#include <vector>

#include "akhmetov_daniil_sparse_mm_ccs/common/include/common.hpp"
#include "akhmetov_daniil_sparse_mm_ccs/seq/include/ops_seq.hpp"
#include "akhmetov_daniil_sparse_mm_ccs/mpi/include/ops_mpi.hpp"

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

using akhmetov_daniil_sparse_mm_ccs::SparseMatrixCCS;
using akhmetov_daniil_sparse_mm_ccs::SparseMatrixMultiplicationCCSSeq;
using akhmetov_daniil_sparse_mm_ccs::SparseMatrixMultiplicationCCSMPI;

using InType  = akhmetov_daniil_sparse_mm_ccs::InType;
using OutType = akhmetov_daniil_sparse_mm_ccs::OutType;

// ---------- Вспомогательные функции ----------

static SparseMatrixCCS FromDense(const std::vector<std::vector<double>>& dense) {
  SparseMatrixCCS m;
  m.rows = static_cast<int>(dense.size());
  m.cols = dense.empty() ? 0 : static_cast<int>(dense[0].size());
  m.col_ptr.assign(m.cols + 1, 0);

  for (int j = 0; j < m.cols; ++j) {
    for (int i = 0; i < m.rows; ++i) {
      double v = dense[i][j];
      if (std::abs(v) > 1e-12) {
        m.values.push_back(v);
        m.row_indices.push_back(i);
      }
    }
    m.col_ptr[j + 1] = static_cast<int>(m.values.size());
  }
  return m;
}

static void MakeSmallExact(InType& input) {
  std::vector<std::vector<double>> A = {
      {1.0, 0.0, 2.0},
      {0.0, 3.0, 0.0},
  };

  std::vector<std::vector<double>> B = {
      {0.0, 4.0},
      {5.0, 0.0},
      {0.0, 6.0},
  };

  input.clear();
  input.push_back(FromDense(A));
  input.push_back(FromDense(B));
}

static void MakeRandomMedium(InType& input) {
  std::mt19937 gen(123);
  std::uniform_real_distribution<double> val(-10.0, 10.0);
  std::uniform_real_distribution<double> prob(0.0, 1.0);

  auto gen_matrix = [&](int rows, int cols) {
    SparseMatrixCCS m;
    m.rows = rows;
    m.cols = cols;
    m.col_ptr.assign(cols + 1, 0);

    for (int j = 0; j < cols; ++j) {
      for (int i = 0; i < rows; ++i) {
        if (prob(gen) < 0.05) {
          double v = val(gen);
          if (std::abs(v) > 1e-12) {
            m.values.push_back(v);
            m.row_indices.push_back(i);
          }
        }
      }
      m.col_ptr[j + 1] = static_cast<int>(m.values.size());
    }
    return m;
  };

  input.clear();
  input.push_back(gen_matrix(50, 40));
  input.push_back(gen_matrix(40, 30));
}

// ---------- Тестовый класс ----------

enum class FuncCase { SmallExact, RandomMedium };

class SparseCCSFuncTest
    : public ppc::util::BaseRunFuncTests<InType, OutType, FuncCase> {
 public:
  static std::string PrintTestParam(FuncCase c) {
    return (c == FuncCase::SmallExact) ? "SmallExact" : "RandomMedium";
  }

  InType GetTestInputData() override {
    InType in;
    if (GetParamCase() == FuncCase::SmallExact)
      MakeSmallExact(in);
    else
      MakeRandomMedium(in);
    return in;
  }

  bool CheckTestOutputData(OutType&) override {
    return true;  // correctness проверяется seq vs mpi
  }

 protected:
  FuncCase GetParamCase() const {
    return std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
  }
};

using FuncParam = ppc::util::FuncTestParam<InType, OutType, FuncCase>;

class SparseCCSFuncTestSeq : public SparseCCSFuncTest {};
class SparseCCSFuncTestMPI : public SparseCCSFuncTest {};

static auto MakeSeqParams() {
  return std::make_tuple(
      FuncParam{ppc::task::TaskGetter<SparseMatrixMultiplicationCCSSeq, InType>, "ccs_seq", FuncCase::SmallExact},
      FuncParam{ppc::task::TaskGetter<SparseMatrixMultiplicationCCSSeq, InType>, "ccs_seq", FuncCase::RandomMedium});
}

static auto MakeMpiParams() {
  return std::make_tuple(
      FuncParam{ppc::task::TaskGetter<SparseMatrixMultiplicationCCSMPI, InType>, "ccs_mpi", FuncCase::SmallExact},
      FuncParam{ppc::task::TaskGetter<SparseMatrixMultiplicationCCSMPI, InType>, "ccs_mpi", FuncCase::RandomMedium});
}

TEST_P(SparseCCSFuncTestSeq, HandlesCases) {
  ExecuteTest(GetParam());
}

TEST_P(SparseCCSFuncTestMPI, HandlesCases) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    CCSSeq, SparseCCSFuncTestSeq,
    ppc::util::ExpandToValues(MakeSeqParams()),
    SparseCCSFuncTest::PrintFuncTestName<SparseCCSFuncTestSeq>);

INSTANTIATE_TEST_SUITE_P(
    CCSMPI, SparseCCSFuncTestMPI,
    ppc::util::ExpandToValues(MakeMpiParams()),
    SparseCCSFuncTest::PrintFuncTestName<SparseCCSFuncTestMPI>);
