#include "akhmetov_daniil_sparse_mm_ccs/mpi/include/ops_mpi.hpp"
#include <algorithm>
#include <cmath>

#include <vector>

namespace akhmetov_daniil_sparse_mm_ccs {

bool SparseMatrixMultiplicationCCSMPI::ValidationImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    if (GetInput().size() != 2) return false;
    if (GetInput()[0].cols != GetInput()[1].rows) return false;
  }
  return true;
}

bool SparseMatrixMultiplicationCCSMPI::PreProcessingImpl() {
  return true;
}

bool SparseMatrixMultiplicationCCSMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rowsA, colsA, colsB;
  if (rank == 0) {
    rowsA = GetInput()[0].rows;
    colsA = GetInput()[0].cols;
    colsB = GetInput()[1].cols;
  }
  MPI_Bcast(&rowsA, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&colsA, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&colsB, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Подготовка локальных данных для A (она нужна всем целиком)
  std::vector<int> col_ptrA(colsA + 1);
  if (rank == 0) col_ptrA = GetInput()[0].col_ptr;
  MPI_Bcast(col_ptrA.data(), colsA + 1, MPI_INT, 0, MPI_COMM_WORLD);

  int nnzA = col_ptrA[colsA];
  std::vector<double> valuesA(nnzA);
  std::vector<int> rows_indA(nnzA);
  if (rank == 0) {
    valuesA = GetInput()[0].values;
    rows_indA = GetInput()[0].row_indices;
  }
  MPI_Bcast(valuesA.data(), nnzA, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(rows_indA.data(), nnzA, MPI_INT, 0, MPI_COMM_WORLD);

  // Распределение столбцов B
  int chunk = colsB / size;
  int remainder = colsB % size;
  int start_col = rank * chunk + std::min(rank, remainder);
  int end_col = start_col + chunk + (rank < remainder ? 1 : 0);
  int local_cols = end_col - start_col;

  // Рассылка структуры B (упрощенно - целиком, либо только нужные столбцы)
  // Для простоты и корректности CCS здесь рассылаем структуру col_ptr
  std::vector<int> col_ptrB(colsB + 1);
  if (rank == 0) col_ptrB = GetInput()[1].col_ptr;
  MPI_Bcast(col_ptrB.data(), colsB + 1, MPI_INT, 0, MPI_COMM_WORLD);

  int nnzB = col_ptrB[colsB];
  std::vector<double> valuesB(nnzB);
  std::vector<int> rows_indB(nnzB);
  if (rank == 0) {
    valuesB = GetInput()[1].values;
    rows_indB = GetInput()[1].row_indices;
  }
  MPI_Bcast(valuesB.data(), nnzB, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(rows_indB.data(), nnzB, MPI_INT, 0, MPI_COMM_WORLD);

  // Локальные вычисления
  std::vector<double> local_values;
  std::vector<int> local_rows;
  std::vector<int> local_col_ptr(local_cols + 1, 0);
  std::vector<double> dense_col(rowsA, 0.0);

  for (int j = 0; j < local_cols; ++j) {
    int global_j = start_col + j;
    std::fill(dense_col.begin(), dense_col.end(), 0.0);

    for (int k_ptr = col_ptrB[global_j]; k_ptr < col_ptrB[global_j + 1]; ++k_ptr) {
      int k = rows_indB[k_ptr];
      double valB = valuesB[k_ptr];
      for (int i_ptr = col_ptrA[k]; i_ptr < col_ptrA[k + 1]; ++i_ptr) {
        dense_col[rows_indA[i_ptr]] += valuesA[i_ptr] * valB;
      }
    }

    for (int i = 0; i < rowsA; ++i) {
      if (std::abs(dense_col[i]) > 1e-15) {
        local_values.push_back(dense_col[i]);
        local_rows.push_back(i);
      }
    }
    local_col_ptr[j + 1] = static_cast<int>(local_values.size());
  }

  // Сбор результатов
  if (rank == 0) {
    res_matrix_.rows = rowsA;
    res_matrix_.cols = colsB;
    res_matrix_.col_ptr.resize(colsB + 1, 0);
    
    // Копируем свои данные
    res_matrix_.values = local_values;
    res_matrix_.row_indices = local_rows;
    for (int j = 0; j <= local_cols; ++j) res_matrix_.col_ptr[j] = local_col_ptr[j];

    // Принимаем от других
    for (int p = 1; p < size; ++p) {
      int p_start = p * chunk + std::min(p, remainder);
      int p_cols = chunk + (p < remainder ? 1 : 0);
      int p_nnz;
      MPI_Recv(&p_nnz, 1, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      
      std::vector<double> p_vals(p_nnz);
      std::vector<int> p_rows(p_nnz);
      std::vector<int> p_ptr(p_cols + 1);

      MPI_Recv(p_vals.data(), p_nnz, MPI_DOUBLE, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(p_rows.data(), p_nnz, MPI_INT, p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(p_ptr.data(), p_cols + 1, MPI_INT, p, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      int offset = static_cast<int>(res_matrix_.values.size());
      res_matrix_.values.insert(res_matrix_.values.end(), p_vals.begin(), p_vals.end());
      res_matrix_.row_indices.insert(res_matrix_.row_indices.end(), p_rows.begin(), p_rows.end());

      for (int j = 1; j <= p_cols; ++j) {
        res_matrix_.col_ptr[p_start + j] = p_ptr[j] + offset;
      }
    }
  } else {
    int nnz = static_cast<int>(local_values.size());
    MPI_Send(&nnz, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    MPI_Send(local_values.data(), nnz, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    MPI_Send(local_rows.data(), nnz, MPI_INT, 0, 2, MPI_COMM_WORLD);
    MPI_Send(local_col_ptr.data(), local_cols + 1, MPI_INT, 0, 3, MPI_COMM_WORLD);
  }

  return true;
}

bool SparseMatrixMultiplicationCCSMPI::PostProcessingImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetOutput() = std::move(res_matrix_);
  }
  return true;
}

}  // namespace akhmetov_daniil_sparse_mm_ccs