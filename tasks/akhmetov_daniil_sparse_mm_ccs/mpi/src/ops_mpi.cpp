#include "akhmetov_daniil_sparse_mm_ccs/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace akhmetov_daniil_sparse_mm_ccs {

bool SparseMatrixMultiplicationCCSMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    if (GetInput().size() != 2) {
      return false;
    }
    if (GetInput()[0].cols != GetInput()[1].rows) {
      return false;
    }
  }
  return true;
}

bool SparseMatrixMultiplicationCCSMPI::PreProcessingImpl() {
  return true;
}

void SparseMatrixMultiplicationCCSMPI::BroadcastInputMatrices(int &rows_a, int &cols_a, int &cols_b,
                                                              std::vector<int> &col_ptr_a,
                                                              std::vector<double> &values_a,
                                                              std::vector<int> &rows_ind_a, std::vector<int> &col_ptr_b,
                                                              std::vector<double> &values_b,
                                                              std::vector<int> &rows_ind_b) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    rows_a = GetInput()[0].rows;
    cols_a = GetInput()[0].cols;
    cols_b = GetInput()[1].cols;
    col_ptr_a = GetInput()[0].col_ptr;
    values_a = GetInput()[0].values;
    rows_ind_a = GetInput()[0].row_indices;
    col_ptr_b = GetInput()[1].col_ptr;
    values_b = GetInput()[1].values;
    rows_ind_b = GetInput()[1].row_indices;
  }

  MPI_Bcast(&rows_a, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols_a, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols_b, 1, MPI_INT, 0, MPI_COMM_WORLD);

  col_ptr_a.resize(cols_a + 1);
  MPI_Bcast(col_ptr_a.data(), cols_a + 1, MPI_INT, 0, MPI_COMM_WORLD);

  int nnz_a = col_ptr_a[cols_a];
  values_a.resize(nnz_a);
  rows_ind_a.resize(nnz_a);
  MPI_Bcast(values_a.data(), nnz_a, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(rows_ind_a.data(), nnz_a, MPI_INT, 0, MPI_COMM_WORLD);

  col_ptr_b.resize(cols_b + 1);
  MPI_Bcast(col_ptr_b.data(), cols_b + 1, MPI_INT, 0, MPI_COMM_WORLD);

  int nnz_b = col_ptr_b[cols_b];
  values_b.resize(nnz_b);
  rows_ind_b.resize(nnz_b);
  MPI_Bcast(values_b.data(), nnz_b, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(rows_ind_b.data(), nnz_b, MPI_INT, 0, MPI_COMM_WORLD);
}

void SparseMatrixMultiplicationCCSMPI::ComputeLocalProduct(
    int rank, int size, int rows_a, int cols_b, const std::vector<int> &col_ptr_a, const std::vector<double> &values_a,
    const std::vector<int> &rows_ind_a, const std::vector<int> &col_ptr_b, const std::vector<double> &values_b,
    const std::vector<int> &rows_ind_b, std::vector<double> &local_values, std::vector<int> &local_rows,
    std::vector<int> &local_col_ptr) {
  int chunk = cols_b / size;
  int remainder = cols_b % size;
  int start_col = (rank * chunk) + std::min(rank, remainder);
  int local_cols = chunk + (rank < remainder ? 1 : 0);

  local_col_ptr.assign(local_cols + 1, 0);
  std::vector<double> dense_col(rows_a, 0.0);

  for (int j = 0; j < local_cols; ++j) {
    int global_j = start_col + j;
    std::fill(dense_col.begin(), dense_col.end(), 0.0);  // NOLINT

    for (int k_ptr = col_ptr_b[global_j]; k_ptr < col_ptr_b[global_j + 1]; ++k_ptr) {
      int k = rows_ind_b[k_ptr];
      double val_b = values_b[k_ptr];
      for (int i_ptr = col_ptr_a[k]; i_ptr < col_ptr_a[k + 1]; ++i_ptr) {
        dense_col[rows_ind_a[i_ptr]] += values_a[i_ptr] * val_b;
      }
    }

    for (int i = 0; i < rows_a; ++i) {
      if (std::abs(dense_col[i]) > 1e-15) {
        local_values.push_back(dense_col[i]);
        local_rows.push_back(i);
      }
    }
    local_col_ptr[j + 1] = static_cast<int>(local_values.size());
  }
}

void SparseMatrixMultiplicationCCSMPI::GatherResult(int rank, int size, int rows_a, int cols_b,
                                                    const std::vector<double> &local_values,
                                                    const std::vector<int> &local_rows,
                                                    const std::vector<int> &local_col_ptr) {
  if (rank == 0) {
    res_matrix_.rows = rows_a;
    res_matrix_.cols = cols_b;
    res_matrix_.col_ptr.resize(cols_b + 1, 0);

    int chunk = cols_b / size;
    int remainder = cols_b % size;
    int start_col = 0;

    res_matrix_.values = local_values;
    res_matrix_.row_indices = local_rows;

    for (int j = 0; j < static_cast<int>(local_col_ptr.size()) - 1; ++j) {
      int global_j = start_col + j;
      res_matrix_.col_ptr[global_j + 1] = local_col_ptr[j + 1];
    }

    int total_nnz_so_far = static_cast<int>(local_values.size());

    for (int proc = 1; proc < size; ++proc) {
      int proc_cols = chunk + (proc < remainder ? 1 : 0);
      int proc_start = (proc * chunk) + std::min(proc, remainder);
      int proc_nnz = 0;

      MPI_Recv(&proc_nnz, 1, MPI_INT, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      std::vector<double> vals(proc_nnz);
      std::vector<int> rows(proc_nnz);
      std::vector<int> ptr(proc_cols + 1);

      MPI_Recv(vals.data(), proc_nnz, MPI_DOUBLE, proc, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(rows.data(), proc_nnz, MPI_INT, proc, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(ptr.data(), proc_cols + 1, MPI_INT, proc, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      res_matrix_.values.insert(res_matrix_.values.end(), vals.begin(), vals.end());
      res_matrix_.row_indices.insert(res_matrix_.row_indices.end(), rows.begin(), rows.end());

      for (int j = 0; j < proc_cols; ++j) {
        int global_col = proc_start + j;
        res_matrix_.col_ptr[global_col + 1] = ptr[j + 1] + total_nnz_so_far;
      }

      total_nnz_so_far += proc_nnz;
    }

  } else {
    int nnz = static_cast<int>(local_values.size());
    MPI_Send(&nnz, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    MPI_Send(local_values.data(), nnz, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    MPI_Send(local_rows.data(), nnz, MPI_INT, 0, 2, MPI_COMM_WORLD);
    MPI_Send(local_col_ptr.data(), static_cast<int>(local_col_ptr.size()), MPI_INT, 0, 3, MPI_COMM_WORLD);
  }
}

bool SparseMatrixMultiplicationCCSMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rows_a = 0;
  int cols_a = 0;
  int cols_b = 0;

  std::vector<int> col_ptr_a;
  std::vector<double> values_a;
  std::vector<int> rows_ind_a;
  std::vector<int> col_ptr_b;
  std::vector<double> values_b;
  std::vector<int> rows_ind_b;

  BroadcastInputMatrices(rows_a, cols_a, cols_b, col_ptr_a, values_a, rows_ind_a, col_ptr_b, values_b, rows_ind_b);

  std::vector<double> local_values;
  std::vector<int> local_rows;
  std::vector<int> local_col_ptr;

  ComputeLocalProduct(rank, size, rows_a, cols_b, col_ptr_a, values_a, rows_ind_a, col_ptr_b, values_b, rows_ind_b,
                      local_values, local_rows, local_col_ptr);

  GatherResult(rank, size, rows_a, cols_b, local_values, local_rows, local_col_ptr);

  return true;
}

bool SparseMatrixMultiplicationCCSMPI::PostProcessingImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    GetOutput() = std::move(res_matrix_);

    auto &output = GetOutput();

    int rows = output.rows;
    int cols = output.cols;
    int col_ptr_size = static_cast<int>(output.col_ptr.size());
    int values_size = static_cast<int>(output.values.size());

    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&col_ptr_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&values_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
      output.rows = rows;
      output.cols = cols;
      output.col_ptr.resize(col_ptr_size);
      output.values.resize(values_size);
      output.row_indices.resize(values_size);
    }

    MPI_Bcast(output.col_ptr.data(), col_ptr_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(output.values.data(), values_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(output.row_indices.data(), values_size, MPI_INT, 0, MPI_COMM_WORLD);

  } else {
    int rows = 0;
    int cols = 0;
    int col_ptr_size = 0;
    int values_size = 0;
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&col_ptr_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&values_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    GetOutput().rows = rows;
    GetOutput().cols = cols;
    GetOutput().col_ptr.resize(col_ptr_size);
    GetOutput().values.resize(values_size);
    GetOutput().row_indices.resize(values_size);

    MPI_Bcast(GetOutput().col_ptr.data(), col_ptr_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(GetOutput().values.data(), values_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(GetOutput().row_indices.data(), values_size, MPI_INT, 0, MPI_COMM_WORLD);
  }

  return true;
}

}  // namespace akhmetov_daniil_sparse_mm_ccs
