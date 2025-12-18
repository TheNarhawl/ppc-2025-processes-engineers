#include "akhmetov_daniil_integration_monte_carlo/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <vector>

#include "akhmetov_daniil_integration_monte_carlo/common/include/common.hpp"

namespace akhmetov_daniil_integration_monte_carlo {

AkhmetovDaniilIntegrationMonteCarloMPI::AkhmetovDaniilIntegrationMonteCarloMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool AkhmetovDaniilIntegrationMonteCarloMPI::ValidationImpl() {
  const auto &[a, b, n, func_id] = GetInput();

  return (a < b) && (n > 0) && (func_id >= FuncType::kLinearFunc) && (func_id <= FuncType::kConstFunc);
}

bool AkhmetovDaniilIntegrationMonteCarloMPI::PreProcessingImpl() {
  const auto &[a, b, n, func_id] = GetInput();
  a_ = a;
  b_ = b;
  point_count_ = n;
  func_id_ = func_id;

  return true;
}

bool AkhmetovDaniilIntegrationMonteCarloMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<int> points_per_process(size);
  std::vector<int> displacements(size);

  int base_point_count = point_count_ / size;
  int extra_points = point_count_ % size;

  int curr_start_index = 0;
  for (int i = 0; i < size; ++i) {
    if (i < extra_points) {
      points_per_process[i] = base_point_count + 1;
    } else {
      points_per_process[i] = base_point_count;
    }

    displacements[i] = curr_start_index;
    curr_start_index += points_per_process[i];
  }

  int local_points_to_process = points_per_process[rank];

  std::vector<double> all_seeds;
  if (rank == 0) {
    all_seeds.resize(point_count_);
    for (int i = 0; i < point_count_; ++i) {
      all_seeds[i] = static_cast<double>(i);
    }
  }

  std::vector<double> local_seeds(local_points_to_process);
  MPI_Scatterv(all_seeds.data(), points_per_process.data(), displacements.data(), MPI_DOUBLE, local_seeds.data(),
               local_points_to_process, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double local_sum = 0.0;
  const double magic_constant = 0.75487766624669276;

  for (int i = 0; i < local_points_to_process; ++i) {
    double t = std::fmod(local_seeds[i] * magic_constant, 1.0);
    double x = a_ + ((b_ - a_) * t);

    double fx = FunctionPair::Function(func_id_, x);
    local_sum += fx;
  }

  double total_sum = 0.0;
  MPI_Reduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  double integral = 0.0;
  if (rank == 0) {
    double average = total_sum / static_cast<double>(point_count_);
    integral = (b_ - a_) * average;
  }

  MPI_Bcast(&integral, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GetOutput() = integral;
  return true;
}

bool AkhmetovDaniilIntegrationMonteCarloMPI::PostProcessingImpl() {
  return true;
}

}  // namespace akhmetov_daniil_integration_monte_carlo
