import numpy as np
import torch
import botorch
torch.set_default_dtype(torch.double)

import gpytorch

from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models import MultiTaskGP
from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement


from botorch.optim import optimize_acqf_discrete
from botorch.optim import optimize_acqf

from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.utils.multi_objective.hypervolume import Hypervolume

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import combinations
import sys
index=sys.argv[1]
file_out=sys.argv[0][:-3]+sys.argv[1]
# file_out="test"
# index="1"






import warnings
warnings.simplefilter("ignore")
from sklearn.preprocessing import MinMaxScaler
import numpy as np

device='cuda' if torch.cuda.is_available() else 'cpu'

input_space = 'discrete' #'continuous' # or 'discrete'

# Define input space for dicrete case
num_input = 4
n_candidates = 1000 # Resolution of grid
  # Uniform sampling
# Continuous bounds for optimization
bounds = torch.stack([torch.ones(num_input)*0.1, torch.ones(num_input)]).to(device)
num_obj = 2
throughput = 'batch' # or 'batch'
batch_size = 5 # q
iterations = 20
initial_samples = 15
num_restart = 20
raw_samples = 100

if input_space == 'discrete':
    xs = torch.rand(n_candidates, num_input).to(device)
else:
    bounds = torch.stack([torch.ones(num_input)*0.1, torch.ones(num_input)]).to(device)
    xs = None  # not used


# def synthetic_objective(x):
#     x1 = x[:, 0]
#     x2 = x[:, 1]
#     x3 = x[:, 2]
#     x4 = x[:, 3]

#     # Branin with x3 modulation
#     a = 1.0
#     b = 5.1 / (4 * np.pi**2)
#     c = 5 / np.pi
#     r = 6
#     s = 10
#     t = 1 / (8 * np.pi)
#     f1 = a * (x2 - b * x1**2 + c * (x1+x4) - r)**2 + s * (1 - t * (1 + x3+x4)) * torch.cos(x1 * (1 + x3)) + s

#     # Currin with x3 modulation
#     term1 = 1 - torch.exp(-1 / (2 * x2*x4 * (1 + x3)))
#     numerator = 2300 * x1**3 + 1900 * x1**2 + 2092 * x1 + 60
#     denominator = 100 * x1**3 + 500 * x1**2 + 4 * x1 + 20 - x4**3
#     f2 = term1 * numerator / denominator * (1 + 0.1 * x3)
#     return torch.stack([f1, f2], dim=-1)

import torch

def synthetic_objective(x):
    # x: tensor of shape [N,4] with columns [x1,x2,x3,x4]
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    x4 = x[:, 3]

    # the six base functions
    f1 = x1**2 + torch.exp(- x2 / x3)
    f2 = x1 + x3
    f3 = x2 / (1 + x3)
    f4 = torch.log(x4 + 1) * x1
    f5 = x2 * torch.sin(x4) + torch.exp(x1)
    f6 = torch.sin(x3) + torch.cos(x4)

    # the two combined outputs
    y      = (f1 * f2 + f2 / f3 + f5 * f4 + f6) / 10
    y_prime = f3 * f2**2 + f4 / f1 + f5 * f6

    # stack into [N,8]
    return torch.stack([ y, y_prime,f1, f2, f3, f4, f5, f6], dim=-1)


def plot_objective_outputs(objective_function, input_ranges, resolution=30):
    """
    Plots pairwise scatter plots of outputs from a multi-output objective function.

    Args:
        objective_function (callable): Function mapping (N, n_inputs) tensor to (N, n_outputs).
        input_ranges (list of tuples): List of (min, max) pairs for each input dimension.
        resolution (int): Number of points per input dimension (default: 100 for 2D).
    """

    n_inputs = len(input_ranges)
    if n_inputs != 4:
        raise ValueError("Currently supports only 2 input dimensions for grid-based plotting.")

    # Create grid for 2D input
    x1 = torch.linspace(*input_ranges[0], resolution)
    x2 = torch.linspace(*input_ranges[1], resolution)
    x3 = torch.linspace(*input_ranges[2], resolution)
    x4 = torch.linspace(*input_ranges[3], resolution)
    X1, X2, X3, X4 = torch.meshgrid(x1, x2, x3, x4, indexing='ij')
    grid = torch.stack([X1.flatten(), X2.flatten(), X3.flatten(), X4.flatten()], dim=1)

    # Evaluate the objective function
    objectives = objective_function(grid)  # shape: (N, n_outputs)
    n_outputs = num_obj

    # Plot all pairwise combinations of outputs
    fig, axs = plt.subplots(n_outputs, n_outputs, figsize=(4 * n_outputs, 4 * n_outputs))
    if n_outputs == 1:
        axs = np.array([[axs]])  # make 2D

    for i in range(n_outputs):
        for j in range(n_outputs):
            ax = axs[i, j]
            if i == j:
                ax.hist(objectives[:, i].numpy(), bins=50, color='gray')
                ax.set_title(f'Histogram of f{i+1}')
            else:
                ax.scatter(objectives[:, j].numpy(), objectives[:, i].numpy(), s=1, alpha=0.5)
                ax.set_xlabel(f'f{j+1}')
                ax.set_ylabel(f'f{i+1}')

    plt.tight_layout()
    plt.show()
    ref_point=torch.zeros(num_obj)-5
    train_y=objectives[:,:num_obj]
    pareto_y = train_y[is_non_dominated(train_y)]
    hv = Hypervolume(ref_point).compute(pareto_y)
    print("____Hypervolume_________",hv)

plot_objective_outputs(synthetic_objective, input_ranges=[(0.01, 1), (0.01, 1), (0.01, 1), (0.01, 1)], resolution=30)






train_x = torch.rand(20, 4)
train_y = synthetic_objective(train_x)



def get_acquisition(model, train_y, ref_point):
    train_y=train_y[:,:num_obj]
    nd_mask = is_non_dominated(train_y)
    pareto_y = train_y[nd_mask]
    partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=pareto_y)

    return qExpectedHypervolumeImprovement(model=model, ref_point=ref_point.tolist(), partitioning=partitioning)

def get_candidate(acq_function, train_x):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    bounds = torch.stack([torch.ones(num_input)*0.1, torch.ones(num_input)]).to(device)
    if input_space == 'discrete':
        xs = torch.rand(n_candidates, num_input)
    else:
        bounds = torch.stack([torch.zeros(num_input), torch.ones(num_input)])
        xs = None  # not used
    xs=xs.to(device)
    if input_space == 'discrete':
        mask = ~torch.any((xs.unsqueeze(1) == train_x.unsqueeze(0)).all(-1), dim=1)
        choices = xs[mask]
        candidate, _ = optimize_acqf(acq_function, q=batch_size, bounds=bounds,options={
            "batch_limit": 10,
            "maxiter": 200,
            "nonnegative": True,
            "method": "SLSQP",
            "candidates": xs,

        }, num_restarts=num_restart, raw_samples=raw_samples)
    else:
        candidate, _ = optimize_acqf(acq_function, q=batch_size, bounds=bounds,options={
            "batch_limit": 10,
            "maxiter": 20,
            "nonnegative": True,
            "method": "SLSQP",
            # "candidates": xs,
            "collapse_bounds": True,
        }, num_restarts=num_restart, raw_samples=raw_samples)

    print("____________Optimization SLSQP________________")
    return candidate


def run_optimization(num_queries, init_points):
    # Initialize training inputs
    if input_space == 'discrete':
        indices = torch.linspace(0, len(xs) - 1, steps=init_points).round().long()
        train_x = xs[indices]
    else:
        train_x = torch.rand(init_points, num_input)  # now supports any num_input dimension

    # Get initial observations
    train_y = synthetic_objective(train_x)

    # Set reference point slightly below minimum for HV calculation
    ref_point = torch.zeros(train_y.shape[1])
    hypervolumes = []

    # BO loop
    for _ in range(num_queries):
        model = fit_gp(train_x, train_y)
        acq_function = get_acquisition(model, train_y, ref_point)
        candidate = get_candidate(acq_function, train_x)
        next_y = synthetic_objective(candidate)

        # Update training data
        train_x = torch.cat([train_x, candidate], dim=0)
        train_y = torch.cat([train_y, next_y], dim=0)

        # Pareto and hypervolume update
        pareto_y = train_y[is_non_dominated(train_y)]
        hv = Hypervolume(ref_point).compute(pareto_y)
        hypervolumes.append(hv)

    return torch.tensor(hypervolumes), train_x, train_y

# hypervolumes, final_train_x, final_train_y = run_optimization(iterations, initial_samples)

# # Plot Hypervolume
# plt.figure(figsize=(8, 5))
# plt.plot(range(1, len(hypervolumes)+1), hypervolumes, marker='o')
# plt.xlabel("Number of Queries")
# plt.ylabel("Hypervolume")
# plt.title("Hypervolume Progress")
# plt.grid(True)
# plt.show()

# # Plot Pareto front
# nd_mask = is_non_dominated(final_train_y)
# pareto_y = final_train_y[nd_mask]
# init_train_y = final_train_y[:initial_samples]
# bo_train_y = final_train_y[initial_samples:]


# # Generate a grid of inputs and evaluate the synthetic objective on it
# # Generate a 3D grid of inputs and evaluate the synthetic objective on it
# x1 = torch.linspace(0, 1, 30)
# x2 = torch.linspace(0, 1, 30)
# x3 = torch.linspace(0, 1, 30)
# x4 = torch.linspace(0, 1, 30)

# X1, X2, X3, X4 = torch.meshgrid(x1, x2, x3, x4, indexing='ij')
# grid = torch.stack([X1.flatten(), X2.flatten(), X3.flatten(), X4.flatten()], dim=1)

# # Evaluate objective values over full 3D space
# objective_values = synthetic_objective(grid)
# f1 = objective_values[:, 0]
# f2 = objective_values[:, 1]


# # Final Pareto plot with background scatter
# # Final Pareto plot with background scatter
# plt.figure(figsize=(8, 5))

# # Plot full objective space scatter in the background
# plt.scatter(f1.numpy(), f2.numpy(), s=1, alpha=0.3, label="Objective Space")

# # Initial samples
# plt.scatter(init_train_y[:, 0], init_train_y[:, 1], marker='s', color='blue', label="Initial Points")

# # BO iterations
# plt.scatter(bo_train_y[:, 0], bo_train_y[:, 1], facecolors='none', edgecolors='gray', label="BO Queries", s=100)
# for i, (x, y) in enumerate(bo_train_y, 1):
#     plt.text(x, y, str(i), fontsize=8, ha='center', va='center')

# # Pareto front
# plt.scatter(pareto_y[:, 0], pareto_y[:, 1], color='red', label="Pareto Front")

# plt.xlabel("Objective 1")
# plt.ylabel("Objective 2")
# plt.title("BO Queries over Objective Space")
# plt.legend()
# plt.grid(True)
# plt.show()



from gpytorch.distributions import MultivariateNormal
from botorch.posteriors import GPyTorchPosterior
from linear_operator.operators import to_linear_operator

import torch
from gpytorch.distributions import MultitaskMultivariateNormal
from botorch.posteriors import GPyTorchPosterior

import torch
from gpytorch.distributions import MultitaskMultivariateNormal
from botorch.posteriors import GPyTorchPosterior



import torch
from gpytorch.distributions import MultitaskMultivariateNormal
from botorch.posteriors import GPyTorchPosterior

class PartialModelListGP(ModelListGP):
    def __init__(self, *models, active_tasks=None):
        super().__init__(*models)
        self.active_tasks = active_tasks if active_tasks is not None else list(range(len(models)))

    def posterior(self, X, output_indices=None, observation_noise=False, **kwargs):
        """Returns posterior only for active tasks."""
        if output_indices is None:
            output_indices = self.active_tasks
        return super().posterior(X, output_indices=output_indices, observation_noise=observation_noise, **kwargs)

class HybridPosterior(GPyTorchPosterior):
    """Custom posterior combining DGP mean with GP covariance and exposing mvn for sampling."""

    def __init__(self, dgp_posterior: GPyTorchPosterior, gp_posterior: GPyTorchPosterior):
        # Store the underlying posteriors
        self.dgp_posterior = dgp_posterior
        self.gp_posterior = gp_posterior

        # Extract DGP mean
        dgp_dist = dgp_posterior.distribution
        loc = dgp_dist.mean
        # Ensure at least 2D for multi-output
        if loc.dim() < 2:
            loc = loc.unsqueeze(-1)

        # Extract GP covariance operator
        gp_dist = gp_posterior.distribution
        covar = gp_dist.lazy_covariance_matrix

        # Determine if outputs are interleaved
        interleaved = getattr(gp_dist, "_interleaved", True)

        # Construct the hybrid multitask distribution
        # Args: mean (batch x q x t), covariance_matrix (batch x (q*t) x (q*t)), validate_args=False, interleaved
        hybrid_dist = MultitaskMultivariateNormal(loc, covar, False, interleaved)
        super().__init__(hybrid_dist)

    @property
    def mvn(self) -> MultitaskMultivariateNormal:
        """Expose the hybrid distribution for samplers."""
        return self.distribution

    @property
    def mean(self) -> torch.Tensor:
        """Use the DGP posterior mean."""
        return self.dgp_posterior.mean

    @property
    def variance(self) -> torch.Tensor:
        """Use the GP posterior variance, broadcasted to the DGP shape if needed."""
        gp_var = self.gp_posterior.variance
        dgp_mean = self.mean
        if gp_var.shape != dgp_mean.shape:
            gp_var = gp_var.expand_as(dgp_mean)
        return gp_var

    def rsample(self, sample_shape: torch.Size | None = None) -> torch.Tensor:
        """Draw samples from the hybrid distribution, defaulting to a single sample if no shape is provided."""
        if sample_shape is None:
            sample_shape = torch.Size([1])
        return self.distribution.rsample(sample_shape=sample_shape)

    def rsample_from_base_samples(self, sample_shape: torch.Size, base_samples: torch.Tensor) -> torch.Tensor:
        """Delegate base-sample drawing to the parent class for correctness."""
        return super().rsample_from_base_samples(sample_shape, base_samples)


import pandas as pd
import numpy as np
import torch
import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error



class DGPLastLayer3(gpytorch.models.ApproximateGP):
    def __init__(self, input_dims, output_dims, num_inducing=128, linear_mean=True):
        num_latents = 10
        inducing_points = torch.randn(10, num_inducing, input_dims)

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=torch.Size([num_latents]))

        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=output_dims,
            num_latents=10,
            latent_dim=-1
        )

        super().__init__(variational_strategy)
        self.mean_module = ConstantMean(batch_shape=torch.Size([10])) if linear_mean else LinearMean(input_dims) #gpytorch.means.ConstantMean(batch_shape=torch.Size([10]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([10])),
            batch_shape=torch.Size([10])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

class DGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=32, linear_mean=True):
        inducing_points = torch.randn(output_dims, num_inducing, input_dims)
        batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super().__init__(variational_strategy, input_dims, output_dims)
        self.mean_module = ConstantMean(batch_shape=batch_shape) if linear_mean else LinearMean(input_dims)
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


import gpytorch
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.models.gpytorch import GPyTorchModel, MultiTaskGPyTorchModel
from botorch.models.model import FantasizeMixin
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.models.utils.gpytorch_modules import (
    get_matern_kernel_with_gamma_prior,
    MIN_INFERRED_NOISE_LEVEL,
)
from botorch.posteriors.multitask import MultitaskGPPosterior
from botorch.utils.datasets import MultiTaskDataset, SupervisedDataset
from gpytorch.constraints import GreaterThan
from gpytorch.distributions.multitask_multivariate_normal import (
    MultitaskMultivariateNormal,
)
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels.index_kernel import IndexKernel
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.multitask_kernel import MultitaskKernel
from gpytorch.likelihoods.gaussian_likelihood import (
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
)
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.likelihoods.multitask_gaussian_likelihood import (
    MultitaskGaussianLikelihood,
)
from gpytorch.means import MultitaskMean
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.models.exact_gp import ExactGP
from gpytorch.module import Module
from gpytorch.priors.lkj_prior import LKJCovariancePrior
from gpytorch.priors.prior import Prior
from gpytorch.priors.smoothed_box_prior import SmoothedBoxPrior
from gpytorch.priors.torch_priors import GammaPrior
from gpytorch.settings import detach_test_caches
from gpytorch.utils.errors import CachingError
from gpytorch.utils.memoize import cached, pop_from_cache
from linear_operator.operators import (
    BatchRepeatLinearOperator,
    CatLinearOperator,
    DiagLinearOperator,
    KroneckerProductDiagLinearOperator,
    KroneckerProductLinearOperator,
    RootLinearOperator,
    to_linear_operator,
)
from torch import Tensor
def get_task_value_remapping(
    task_values: Tensor, dtype: torch.dtype
) -> Optional[Tensor]:
    """Construct an mapping of discrete task values to contiguous int-valued floats.

    Args:
        task_values: A sorted long-valued tensor of task values.
        dtype: The dtype of the model inputs (e.g. `X`), which the new
            task values should have mapped to (e.g. float, double).

    Returns:
        A tensor of shape `task_values.max() + 1` that maps task values
        to new task values. The indexing operation `mapper[task_value]`
        will produce a tensor of new task values, of the same shape as
        the original. The elements of the `mapper` tensor that do not
        appear in the original `task_values` are mapped to `nan`. The
        return value will be `None`, when the task values are contiguous
        integers starting from zero.
    """
    task_range = torch.arange(
        len(task_values), dtype=task_values.dtype, device=task_values.device
    )
    mapper = None
    if not torch.equal(task_values, task_range):
        # Create a tensor that maps task values to new task values.
        # The number of tasks should be small, so this should be quite efficient.
        mapper = torch.full(
            (task_values.max().item() + 1,),
            float("nan"),
            dtype=dtype,
            device=task_values.device,
        )
        mapper[task_values] = task_range.to(dtype=dtype)
    return mapper


class MultiTaskDeepGP(DeepGP, MultiTaskGPyTorchModel):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        reduction: Optional[int] = 8,
        train_Yvar: Optional[Tensor] = None,
        mean_module: Optional[Module] = None,
        covar_module: Optional[Module] = None,
        likelihood: Optional[Likelihood] = None,
        task_covar_prior: Optional[Prior] = None,
        output_tasks: Optional[List[int]] = None,
        rank: Optional[int] = None,
        all_tasks: Optional[List[int]] = None,
        input_transform: Optional[InputTransform] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
    ) -> None:
        super().__init__()
        self.reduction=reduction
        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )
        self._validate_tensor_args(X=transformed_X, Y=train_Y, Yvar=train_Yvar)
        (
            all_tasks_inferred,
            task_feature,
            self.num_non_task_features,
        ) = self.get_all_tasks(transformed_X, task_feature, output_tasks)
        if all_tasks is not None and not set(all_tasks_inferred).issubset(all_tasks):
            raise UnsupportedError(
                f"The provided {all_tasks=} does not contain all the task features "
                f"inferred from the training data {all_tasks_inferred=}. "
                "This is not allowed as it will lead to errors during model training."
            )
        all_tasks = all_tasks or all_tasks_inferred
        self.num_tasks = len(all_tasks)
        if outcome_transform is not None:
            train_Y, train_Yvar = outcome_transform(Y=train_Y, Yvar=train_Yvar)

        # squeeze output dim
        train_Y = train_Y.squeeze(-1)
        if output_tasks is None:
            output_tasks = all_tasks
        else:
            if set(output_tasks) - set(all_tasks):
                raise RuntimeError("All output tasks must be present in input data.")
        self._output_tasks = output_tasks
        self._num_outputs = len(output_tasks)

        if likelihood is None:
            if train_Yvar is None:
                likelihood = GaussianLikelihood(noise_prior=GammaPrior(1.1, 0.05))
            else:
                likelihood = FixedNoiseGaussianLikelihood(noise=train_Yvar.squeeze(-1))

        self._task_feature = task_feature
        self._base_idxr = torch.arange(self.num_non_task_features)
        self._base_idxr[task_feature:] += 1

        # Single hidden layer with moderate dimension
        # Single hidden layer with fixed dimension
        hidden_layer = DGPHiddenLayer(
            input_dims=train_X.shape[-1]-1,  # Exclude task feature
            output_dims=self.num_tasks-self.reduction,  # Moderate expansion of dimension
            linear_mean=True
        )

        last_layer = DGPLastLayer3(
            input_dims=self.num_tasks-self.reduction,
            output_dims=self.num_tasks,
            linear_mean=True
        )

        self.hidden_layer = hidden_layer
        self.last_layer = last_layer



        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(num_tasks=self.num_tasks)#,noise_constraint=gpytorch.constraints.Interval(0.0001, 0.001))

        self._rank = rank if rank is not None else self.num_tasks
        task_mapper = get_task_value_remapping(
            task_values=torch.tensor(
                all_tasks, dtype=torch.long, device=train_X.device
            ),
            dtype=train_X.dtype,
        )
        self.register_buffer("_task_mapper", task_mapper)
        self._expected_task_values = set(all_tasks)
        if input_transform is not None:
            self.input_transform = input_transform
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        self.to(train_X)
        self.gp_model = None
    def _split_inputs(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Extracts base features and task indices from input data.

        Args:
            x: The full input tensor with trailing dimension of size `d + 1`.
                Should be of float/double data type.

        Returns:
            2-element tuple containing

            - A `q x d` or `b x q x d` (batch mode) tensor with trailing
            dimension made up of the `d` non-task-index columns of `x`, arranged
            in the order as specified by the indexer generated during model
            instantiation.
            - A `q` or `b x q` (batch mode) tensor of long data type containing
            the task indices.
        """
        batch_shape, d = x.shape[:-2], x.shape[-1]
        x_basic = x[..., self._base_idxr].view(batch_shape + torch.Size([-1, d - 1]))
        task_idcs = (
            x[..., self._task_feature]
            .view(batch_shape + torch.Size([-1, 1]))
            .to(dtype=torch.long)
        )
        task_idcs = self._map_tasks(task_values=task_idcs)
        return x_basic, task_idcs

    def forward(self, x: Tensor) -> MultivariateNormal:
        if self.training:
            x = self.transform_inputs(x)

        x_basic, task_idcs = self._split_inputs(x)
        hidden=self.hidden_layer(x_basic)
#         hidden = self.hidden_layer2(hidden_rep1)

        if len(torch.distributions.Normal(loc=hidden.mean, scale=hidden.variance.sqrt()).rsample().shape) == 3:
            task_id = torch.broadcast_to(
                task_idcs.squeeze(-1),
                (torch.distributions.Normal(loc=hidden.mean, scale=hidden.variance.sqrt()).rsample().shape[-3],
                 task_idcs.squeeze(-1).shape[-1])
            )
        else:
            task_id = torch.broadcast_to(
                task_idcs.squeeze(-1),
                (torch.distributions.Normal(loc=hidden.mean, scale=hidden.variance.sqrt()).rsample().shape[-4],
                 torch.distributions.Normal(loc=hidden.mean, scale=hidden.variance.sqrt()).rsample().shape[-3],
                 task_idcs.squeeze(-1).shape[-1])
            )

        output = self.last_layer(
            torch.distributions.Normal(
                loc=hidden.mean,
                scale=hidden.variance.sqrt()
            ).rsample(),
            task_indices=task_id
        )
        return output
    def posterior(self, X: Tensor, output_indices=None, observation_noise=False, **kwargs):
        """
        Returns a posterior with DGP means and GP variances.
        """
        # First, get the DGP posterior for means
        self.eval()  # Make sure we're in eval mode

        # Check if input contains task feature
        has_task_feature = X.shape[-1] == self.num_non_task_features + 1

        # If X includes task feature, output_indices must be None for GPyTorch compatibility
        if output_indices is None:
            dgp_output_indices = None if has_task_feature else self._output_tasks

            # If output_indices is None and we're not using task features, use all tasks
            if dgp_output_indices is None and not has_task_feature:
                dgp_output_indices = list(range(self._output_tasks ))
        else:
            dgp_output_indices = output_indices

        # Get the DGP posterior
        dgp_posterior = super().posterior(X, output_indices=dgp_output_indices, observation_noise=observation_noise, **kwargs)

        # If GP model is not set, just return the DGP posterior
        if self.gp_model is None:
            return dgp_posterior

        # Get the GP posterior for variances - handle task features correctly

        # For GP model, use the same approach with output_indices
        if output_indices is None:
            gp_output_indices = None if has_task_feature else  self._output_tasks
            if gp_output_indices is None and not has_task_feature:
                gp_output_indices = list(range(self._output_tasks ))
        else:
            gp_output_indices = output_indices
        gp_posterior = self.gp_model.posterior(X, output_indices=gp_output_indices, observation_noise=observation_noise, **kwargs)




        # Create hybrid posterior
        hybrid_posterior = HybridPosterior(
            dgp_posterior=dgp_posterior,
            gp_posterior=gp_posterior
        )

        return hybrid_posterior
    @classmethod
    def get_all_tasks(
        cls,
        train_X: Tensor,
        task_feature: int,
        output_tasks: Optional[List[int]] = None,
    ) -> Tuple[List[int], int, int]:
        if train_X.ndim != 2:
            # Currently, batch mode MTGPs are blocked upstream in GPyTorch
            raise ValueError(f"Unsupported shape {train_X.shape} for train_X.")

        d = train_X.shape[-1] - 1
        if not (-d <= task_feature <= d):
            raise ValueError(f"Must have that -{d} <= task_feature <= {d}")
        task_feature = task_feature % (d + 1)
        all_tasks = (
            train_X[..., task_feature].unique(sorted=True).to(dtype=torch.long).tolist()
        )
        return all_tasks, task_feature, d

    @classmethod
    def construct_inputs(
        cls,
        training_data: Union[SupervisedDataset, MultiTaskDataset],
        task_feature: int,
        output_tasks: Optional[List[int]] = None,
        task_covar_prior: Optional[Prior] = None,
        prior_config: Optional[dict] = None,
        rank: Optional[int] = None,
    ) -> Dict[str, Any]:
        r"""Construct `Model` keyword arguments from a dataset and other args.

        Args:
            training_data: A `SupervisedDataset` or a `MultiTaskDataset`.
            task_feature: Column index of embedded task indicator features.
            output_tasks: A list of task indices for which to compute model
                outputs for. If omitted, return outputs for all task indices.
            task_covar_prior: A GPyTorch `Prior` object to use as prior on
                the cross-task covariance matrix,
            prior_config: Configuration for inter-task covariance prior.
                Should only be used if `task_covar_prior` is not passed directly. Must
                contain `use_LKJ_prior` indicator and should contain float value `eta`.
            rank: The rank of the cross-task covariance matrix.
        """
        if task_covar_prior is not None and prior_config is not None:
            raise ValueError(
                "Only one of `task_covar_prior` and `prior_config` arguments expected."
            )

        if prior_config is not None:
            if not prior_config.get("use_LKJ_prior"):
                raise ValueError("Currently only config for LKJ prior is supported.")

            num_tasks = training_data.X[task_feature].unique().numel()
            sd_prior = GammaPrior(1.0, 0.15)
            sd_prior._event_shape = torch.Size([num_tasks])
            eta = prior_config.get("eta", 0.5)
            if not isinstance(eta, float) and not isinstance(eta, int):
                raise ValueError(f"eta must be a real number, your eta was {eta}.")
            task_covar_prior = LKJCovariancePrior(num_tasks, eta, sd_prior)

        # Call Model.construct_inputs to parse training data
        base_inputs = super().construct_inputs(training_data=training_data)
        if (
            isinstance(training_data, MultiTaskDataset)
            # If task features are included in the data, all tasks will have
            # some observations and they may have different task features.
            and training_data.task_feature_index is None
        ):
            all_tasks = list(range(len(training_data.datasets)))
            base_inputs["all_tasks"] = all_tasks
        if task_covar_prior is not None:
            base_inputs["task_covar_prior"] = task_covar_prior
        if rank is not None:
            base_inputs["rank"] = rank
        base_inputs["task_feature"] = task_feature
        base_inputs["output_tasks"] = output_tasks
        return base_inputs







import numpy as np
from sklearn.preprocessing import StandardScaler

def prepare_and_standardize_data(df, input_vars, output_vars, yvar_cols=None, verbose=True):
    """Added yvar_cols parameter and Yvar scaling"""

    """
    Prepare and standardize data while properly handling partial task observations.
    Applies column-wise standardization for both inputs and outputs.

    Args:
        df: pandas DataFrame containing the data
        input_vars: list of input variable names
        output_vars: list of output variable names
        verbose: whether to print information about observations

    Returns:
        tuple containing:
        - scaled input array
        - scaled output array
        - dictionary of input scalers (one per column)
        - dictionary of output scalers (one per column)
    """
    # Initialize scaled input array and input scalers dictionary
    input_scaled = np.full_like(df[input_vars].values, np.nan, dtype=np.float64)
    input_scalers = {}

    # Standardize inputs separately for each column
    for j, col in enumerate(input_vars):
        # Get non-NaN mask for this input
        mask = ~np.isnan(df[input_vars].values[:, j])

        if mask.any():
            scaler = StandardScaler()
            input_scaled[mask, j] = scaler.fit_transform(
                df[input_vars].values[mask, j].reshape(-1, 1)
            ).ravel()
            input_scalers[col] = scaler

            if verbose:
                print(f"Input {col}: {np.sum(mask)}/{len(mask)} valid observations")

    # Initialize scaled output array and output scalers dictionary
    output_scaled = np.full_like(df[output_vars].values, np.nan, dtype=np.float64)
    output_scalers = {}

    # Standardize outputs separately for each task
    for j, col in enumerate(output_vars):
        # Get non-NaN mask for this task
        mask = ~np.isnan(df[output_vars].values[:, j])

        if mask.any():
            scaler = StandardScaler()
            output_scaled[mask, j] = scaler.fit_transform(
                df[output_vars].values[mask, j].reshape(-1, 1)
            ).ravel()
            output_scalers[col] = scaler

            if verbose:
                print(f"Output {col}: {np.sum(mask)}/{len(mask)} valid observations")
    # Original input/output scaling code remains the same

    # Add Yvar handling
    yvar_scaled = None
    if yvar_cols is not None:
        yvar_scaled = np.full_like(df[output_vars].values, 1e-6, dtype=np.float64)  # Default to small noise

        for j, (output_col, yvar_col) in enumerate(zip(output_vars, yvar_cols)):
            if yvar_col is not None and yvar_col in df.columns:
                # Get non-NaN mask for this task's Yvar
                mask = ~np.isnan(df[yvar_col].values)

                if mask.any():
                    # Scale Yvar using the output scaler's variance
                    output_scaler = output_scalers[output_col]
                    yvar_scaled[mask, j] = df[yvar_col].values[mask] / output_scaler.var_

                    if verbose:
                        print(f"Yvar for {output_col}: {np.sum(mask)}/{len(mask)} valid observations")

    return input_scaled, output_scaled, yvar_scaled, input_scalers, output_scalers


def transform_with_scalers(df, input_vars, output_vars, input_scalers, output_scalers):
    """
    Transform new data using previously fitted scalers.

    Args:
        df: pandas DataFrame containing the data
        input_vars: list of input variable names
        output_vars: list of output variable names
        input_scalers: dictionary of fitted input scalers
        output_scalers: dictionary of fitted output scalers

    Returns:
        tuple containing:
        - scaled input array
        - scaled output array
    """
    # Initialize scaled arrays
    input_scaled = np.full_like(df[input_vars].values, np.nan, dtype=np.float64)
    output_scaled = np.full_like(df[output_vars].values, np.nan, dtype=np.float64)

    # Transform inputs
    for j, col in enumerate(input_vars):
        if col in input_scalers:
            mask = ~np.isnan(df[input_vars].values[:, j])
            if mask.any():
                input_scaled[mask, j] = input_scalers[col].transform(
                    df[input_vars].values[mask, j].reshape(-1, 1)
                ).ravel()

    # Transform outputs
    for j, col in enumerate(output_vars):
        if col in output_scalers:
            mask = ~np.isnan(df[output_vars].values[:, j])
            if mask.any():
                output_scaled[mask, j] = output_scalers[col].transform(
                    df[output_vars].values[mask, j].reshape(-1, 1)
                ).ravel()

    return input_scaled, output_scaled

def inverse_transform_data(scaled_inputs, scaled_outputs, input_vars, output_vars,
                         input_scalers, output_scalers):
    """
    Convert scaled data back to original scale.

    Args:
        scaled_inputs: array of scaled input values
        scaled_outputs: array of scaled output values
        input_vars: list of input variable names
        output_vars: list of output variable names
        input_scalers: dictionary of fitted input scalers
        output_scalers: dictionary of fitted output scalers

    Returns:
        tuple containing:
        - original-scale input array
        - original-scale output array
    """
    # Initialize arrays for original-scale data
    original_inputs = np.full_like(scaled_inputs, np.nan, dtype=np.float64)
    original_outputs = np.full_like(scaled_outputs, np.nan, dtype=np.float64)

    # Inverse transform inputs
    for j, col in enumerate(input_vars):
        if col in input_scalers:
            mask = ~np.isnan(scaled_inputs[:, j])
            if mask.any():
                original_inputs[mask, j] = input_scalers[col].inverse_transform(
                    scaled_inputs[mask, j].reshape(-1, 1)
                ).ravel()

    # Inverse transform outputs
    for j, col in enumerate(output_vars):
        if col in output_scalers:
            mask = ~np.isnan(scaled_outputs[:, j])
            if mask.any():
                original_outputs[mask, j] = output_scalers[col].inverse_transform(
                    scaled_outputs[mask, j].reshape(-1, 1)
                ).ravel()

    return original_inputs, original_outputs










def prepare_training_pairs(input_scaled, output_scaled, yvar_scaled=None):
    """Added yvar_scaled parameter"""
    n_samples, n_tasks = output_scaled.shape
    n_features = input_scaled.shape[1]

    total_pairs = np.sum(~np.isnan(output_scaled))
    train_x = np.zeros((total_pairs, n_features + 1))
    train_y = np.zeros(total_pairs)
    train_yvar = np.zeros(total_pairs) if yvar_scaled is not None else None

    idx = 0
    for i in range(n_samples):
        for task in range(n_tasks):
            if not np.isnan(output_scaled[i, task]):
                train_x[idx, :-1] = input_scaled[i]
                train_x[idx, -1] = task
                train_y[idx] = output_scaled[i, task]
                if yvar_scaled is not None:
                    train_yvar[idx] = yvar_scaled[i, task]
                idx += 1

    return train_x, train_y, train_yvar

def train_model(train_x, train_y, num_tasks, train_yvar=None, num_epochs=500, reduction=8):
    """Train the Deep GP model."""
    """Added train_yvar parameter"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x = torch.tensor(train_x, dtype=torch.float64).to(device)
    train_y = torch.tensor(train_y, dtype=torch.float64).to(device)
    if train_yvar is not None:
        train_yvar = torch.tensor(train_yvar, dtype=torch.float64).to(device)

    model = MultiTaskDeepGP(
        train_X=train_x,
        train_Y=train_y.unsqueeze(-1),
        task_feature=-1,
        train_Yvar=train_yvar.unsqueeze(-1) if train_yvar is not None else None,
        reduction=reduction
    ).to(device)





    model = model.double()

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
    ], lr=0.01)

    mll = DeepApproximateMLL(
        VariationalELBO(
            model.likelihood,
            model,
            num_data=train_x.shape[0],
            beta=0.5
        )
    )

    model.train()
    model.likelihood.train()

    losses = []
    best_loss = float('inf')
    patience = 50
    patience_counter = 0

    for i in range(num_epochs):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        losses.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {i+1}")
            break

        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f'Epoch {i+1}/{num_epochs} - Loss: {loss.item():.3f}')

    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()

    return model



def evaluate_model(model, train_x, train_y, task_names, output_scalers,df_p=None):
    """Evaluate model performance with uncertainty visualization."""
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        posterior = model.posterior(torch.tensor(train_x).to(device))
        mean = posterior.mean.squeeze().cpu().numpy().T
        # Extract standard deviation from the posterior
        std = posterior.variance.sqrt().squeeze().cpu().numpy().T
        print("Mean shape:", mean.shape)
        print("Std shape:", std.shape)

    predictions = mean
    uncertainties = std

    metrics = []
    for task_idx, task_name in enumerate(task_names):
        task_mask = (train_x[:, -1] == task_idx)

        if np.sum(task_mask) > 0:
            task_preds = predictions[task_mask]
            task_std = uncertainties[task_mask]
            task_true = train_y[task_mask]

            mae = mean_absolute_error(task_true, task_preds)
            rmse = np.sqrt(mean_squared_error(task_true, task_preds))
            kendall = kendalltau(task_true, task_preds)[0]
            spearman = spearmanr(task_true, task_preds)[0]

            metrics.append({
                'task': task_name,
                'mae': mae,
                'rmse': rmse,
                'kendall': kendall,
                'spearman': spearman,
                'n_samples': np.sum(task_mask)
            })

            plt.figure(figsize=(12, 5))

            plt.subplot(121)
            # Plot prediction with error bars
            plt.errorbar(task_true, task_preds, yerr=2*task_std, fmt='o', alpha=0.5,
                        capsize=3, markersize=4, elinewidth=1, label='Predictions with 2σ')

            # Plot diagonal line
            plt.plot([min(task_true), max(task_true)],
                    [min(task_true), max(task_true)],
                    'r--', lw=2, label='Perfect prediction')
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values')
            plt.title(f'{task_name}\nRMSE: {rmse:.3f}, Spearman: {spearman:.3f}')
            plt.legend()

            plt.subplot(122)
            plt.hist(task_true, bins=20, alpha=0.5, label='True')
            plt.hist(task_preds, bins=20, alpha=0.5, label='Predicted')
            # Add uncertainty band to histogram
#             plt.fill_between(np.linspace(min(task_preds), max(task_preds), 100),
#                            np.histogram(task_preds - 2*task_std, bins=100)[0],
#                            np.histogram(task_preds + 2*task_std, bins=100)[0],
#                            alpha=0.2, color='blue', label='2σ uncertainty')
            plt.xlabel('Values')
            plt.ylabel('Frequency')
            plt.legend()

            plt.tight_layout()
            plt.show()

    return metrics



# if __name__ == "__main__":
#     # Define variables
#     input_vars = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'V']
#     output_vars = [
#         'VarvYS_pred(MPa) at 298K','SFE_calc', 'VEC Avg',
#         'Yield Strength(Mpa)', 'UTS_True(Mpa)', 'UTS/YS',
#         'Elong_T(%)', 'Hardness (GPa)', 'Modulus(Gpa)',
#         'Grain Size(um)','Avg HDYN/HQS'
#     ]

#     # Load data
#     df = pd.read_csv('HTMDEC_MasterTable_Iterations2.csv', encoding='latin1')

#     # Prepare data
#     print("Preparing and standardizing data...")
#     input_scaled, output_scaled, input_scaler, output_scalers = prepare_and_standardize_data(
#         df, input_vars, output_vars
#     )

#     train_x, train_y = prepare_training_pairs(input_scaled, output_scaled)
#     print(f"Total training pairs: {len(train_x)}")

#     # Train model
#     print("\nTraining model...")
#     model = train_model(
#         train_x,
#         train_y,
#         num_tasks=len(output_vars),
#         num_epochs=5000,
#         reduction=7# Increased epochs
#     )

#     # Evaluate model
#     print("\nEvaluating model...")
#     metrics = evaluate_model(model, train_x, train_y, output_vars, output_scalers)

#     # Print metrics
#     for metric in metrics:
#         print(f"\nTask: {metric['task']}")
#         print(f"MAE: {metric['mae']:.4f}")
#         print(f"RMSE: {metric['rmse']:.4f}")
#         print(f"Kendall's Tau: {metric['kendall']:.4f}")
#         print(f"Spearman's R: {metric['spearman']:.4f}")
#         print(f"Number of samples: {metric['n_samples']}")






















from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer  # This needs to be imported first
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# Rest of the code remains exactly the same...
def impute_with_mice(input_scaled, output_scaled, max_iter=10):
    """
    Impute missing values using MICE with Random Forest estimator.
    """
    # Combine input and output for better imputation
    combined_data = np.hstack([input_scaled, output_scaled])

    # Initialize MICE imputer
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=100),
        max_iter=max_iter,
        random_state=42,
        initial_strategy='mean'
    )

    # Perform imputation
    imputed_data = imputer.fit_transform(combined_data)

    # Extract imputed output data
    output_imputed = imputed_data[:, input_scaled.shape[1]:]

    return output_imputed

def cross_validate_model(input_scaled, output_scaled, n_splits=5, reduction=8, num_epochs=500):
    """
    Perform k-fold cross-validation with MICE imputation.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    all_indices = np.arange(len(input_scaled))

    # Print initial missing data statistics
    print("\nMissing data statistics:")
    for j, var in enumerate(output_vars):
        missing = np.sum(np.isnan(output_scaled[:, j]))
        print(f"{var}: {missing}/{len(output_scaled)} missing values ({missing/len(output_scaled)*100:.1f}%)")

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_indices)):
        print(f"\nTraining fold {fold + 1}/{n_splits}")

        # Split data
        train_input = input_scaled[train_idx]
        train_output = output_scaled[train_idx]
        val_input = input_scaled[val_idx]
        val_output = output_scaled[val_idx]

        # Impute missing values separately for training and validation
        print("Imputing training data...")
        train_output_imputed = impute_with_mice(train_input, train_output)
        print("Imputing validation data...")
        val_output_imputed = impute_with_mice(val_input, val_output)

        # Prepare training pairs with imputed data
        train_x_fold, train_y_fold = prepare_training_pairs(train_input, train_output)
        val_x_fold, val_y_fold = prepare_training_pairs(val_input, val_output_imputed)

        # Train model
        model = train_model(
            train_x_fold,
            train_y_fold,
            num_tasks=output_scaled.shape[1],
            num_epochs=num_epochs,
            reduction=reduction
        )

        # Evaluate on validation set
        metrics = evaluate_model(
            model,
            val_x_fold,
            val_y_fold,
            output_vars,
            output_scalers
        )

        # Add imputation information to metrics
        for metric in metrics:
            task_idx = output_vars.index(metric['task'])
            missing_val = np.sum(np.isnan(val_output[:, task_idx]))
            metric['imputed_samples'] = missing_val

        fold_metrics.append(metrics)

    return fold_metrics

def compare_models_cv(input_scaled, output_scaled, reductions=[1, 3, 5, 7, 8, 9], n_splits=5):
    """
    Compare models with different reduction parameters using cross-validation.
    """
    model_results = {}

    for reduction in reductions:
        print(f"\nEvaluating model with reduction={reduction}")
        fold_metrics = cross_validate_model(
            input_scaled,
            output_scaled,
            n_splits=n_splits,
            reduction=reduction
        )
        model_results[reduction] = fold_metrics

    return model_results

def plot_cv_comparison(model_results):
    """
    Plot comparison of models with different reduction parameters, including imputation info.
    """
    reductions = list(model_results.keys())
    metrics = ['rmse', 'spearman']

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    for idx, metric in enumerate(metrics):
        mean_scores = []
        std_scores = []

        for reduction in reductions:
            # Calculate mean and std across folds and tasks
            fold_scores = []
            for fold_metrics in model_results[reduction]:
                task_scores = [task_metric[metric] for task_metric in fold_metrics]
                fold_scores.append(np.mean(task_scores))

            mean_scores.append(np.mean(fold_scores))
            std_scores.append(np.std(fold_scores))

        axes[idx].errorbar(reductions, mean_scores, yerr=std_scores, fmt='o-', capsize=5)
        axes[idx].set_xlabel('Reduction Parameter')
        axes[idx].set_ylabel(f'Mean {metric.upper()}')
        axes[idx].set_title(f'Cross-validation {metric.upper()} vs Reduction Parameter')
        axes[idx].grid(True)

    plt.tight_layout()
    plt.show()



from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import kendalltau, spearmanr
import numpy as np
import matplotlib.pyplot as plt









def plot_split_comparison(model_results):
    """
    Plot comparison of models with different reduction parameters for train-test split results.
    """
    reductions = list(model_results.keys())
    metrics = ['test_rmse', 'test_spearman']  # Updated metric names

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    for idx, metric in enumerate(metrics):
        mean_scores = []
        std_scores = []

        for reduction in reductions:
            # Calculate mean and std across tasks
            task_scores = [task_metric[metric] for task_metric in model_results[reduction]]
            mean_scores.append(np.mean(task_scores))
            std_scores.append(np.std(task_scores))

        axes[idx].errorbar(reductions, mean_scores, yerr=std_scores, fmt='o-', capsize=5)
        axes[idx].set_xlabel('Reduction Parameter')
        axes[idx].set_ylabel(f'Mean {metric.upper()}')
        axes[idx].set_title(f'Test Set {metric.upper()} vs Reduction Parameter')
        axes[idx].grid(True)

    plt.tight_layout()
    plt.show()

def print_model_comparison(model_results):
    """
    Print detailed comparison of models with different reduction parameters.
    """
    print("\nDetailed Model Comparison:")
    print("=" * 50)

    for reduction, metrics in model_results.items():
        print(f"\nReduction Parameter: {reduction}")
        print("-" * 30)

        # Calculate average metrics across tasks
        avg_test_rmse = np.mean([m['test_rmse'] for m in metrics])
        avg_test_spearman = np.mean([m['test_spearman'] for m in metrics])

        print(f"Average Test RMSE: {avg_test_rmse:.4f}")
        print(f"Average Test Spearman: {avg_test_spearman:.4f}")

        # Print per-task metrics
        print("\nPer-task metrics:")
        for metric in metrics:
            print(f"\n{metric['task']}:")
            print(f"  Test RMSE: {metric['test_rmse']:.4f}")
            print(f"  Test Spearman: {metric['test_spearman']:.4f}")
            print(f"  Train samples: {metric['n_train_samples']}")
            print(f"  Test samples: {metric['n_test_samples']}")

def fit_gps_gp(train_x, train_y):
    """Fit independent GPs for each task and combine into PartialModelListGP.

    Args:
        train_x: Tensor with shape (n, d+1) where last column contains task indices
        train_y: Tensor with shape (n,) containing target values
        n_objectives: Number of tasks/objectives
        active_tasks: List of task indices to use for posterior predictions
    """
    print("___________________training__________________________________")
    from botorch.models import SingleTaskGP, ModelListGP, MultiTaskGP
    from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
    from botorch.fit import fit_gpytorch_mll
    models = []
    gp = MultiTaskGP(
            train_X=train_x,
            train_Y=train_y.unsqueeze(-1),
            task_feature=-1,

        )

    # Fit the GP
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    # Loop through each task

    fit_gpytorch_mll(mll)
    return gp


def train_model_with_validation(train_x, train_y, val_x, val_y, num_tasks, train_yvar=None, test_yvar=None, num_epochs=3000, reduction=8):
    """Train the Deep GP model with validation monitoring and Yvar support."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to tensors
    train_x = torch.tensor(train_x, dtype=torch.float64).to(device)
    train_y = torch.tensor(train_y, dtype=torch.float64).to(device)
    val_x = torch.tensor(val_x, dtype=torch.float64).to(device)
    val_y = torch.tensor(val_y, dtype=torch.float64).to(device)

    # Convert Yvar to tensors if provided
    if train_yvar is not None:
        train_yvar = torch.tensor(train_yvar, dtype=torch.float64).to(device)
    if test_yvar is not None:
        test_yvar = torch.tensor(test_yvar, dtype=torch.float64).to(device)

    model = MultiTaskDeepGP(
        train_X=train_x,
        train_Y=train_y.unsqueeze(-1),
        task_feature=-1,
        output_tasks=list(range(num_obj)),
        train_Yvar=train_yvar.unsqueeze(-1) if train_yvar is not None else None,
        reduction=reduction
    ).to(device)

    model = model.double()

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
    ], lr=0.01)

    mll = DeepApproximateMLL(
        VariationalELBO(
            model.likelihood,
            model,
            num_data=train_x.shape[0],
            beta=0.5
        )
    )

    train_losses = []
    val_losses = []
    best_loss = float('inf')
    patience = 50
    patience_counter = 0

    for i in range(num_epochs):
        # Training step
        model.train()
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        train_losses.append(loss.item())

        loss.backward()
        optimizer.step()

        # Validation step
        if val_y.shape[0] != 0:
            model.eval()
            with torch.no_grad():
                val_output = model(val_x)
                val_loss = -mll(val_output, val_y)
                val_losses.append(val_loss.item())

            if val_loss.item() < best_loss:
                best_loss = val_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {i+1}")
                break

            if i % 50 == 0:
                print(f'Epoch {i+1}/{num_epochs} - Train Loss: {loss.item():.3f} - Val Loss: {val_loss.item():.3f}')
        else:

            model.eval()


            if  loss.item() < best_loss and loss.item() > 0:
                best_loss = loss.item()
                patience_counter = 0
            elif loss.item() < 0:
                break
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {i+1}")
                break

            if i % 50 == 0:
                print(f'Epoch {i+1}/{num_epochs} - Train Loss: {loss.item():.3f}')


    standard_gp_model = fit_gps_gp(
        train_x,
        train_y,
        # n_objectives=n_objectives,
        # active_tasks=list(range(main_task))
    )



    model.gp_model = standard_gp_model.to(device)

    return model
def evaluate_model_train_test(model, train_x, train_y, test_x, test_y, task_names, output_scalers,
                            df_prior=None, train_orig_idx=None, test_orig_idx=None):
    """Evaluate model performance with prior values added back."""
    model.eval()
    device = next(model.parameters()).device

    # Get predictions
    with torch.no_grad():
        train_posterior = model.posterior(torch.tensor(train_x).to(device))
        train_mean = train_posterior.mean.squeeze().cpu().numpy()
        train_std = train_posterior.variance.sqrt().squeeze().cpu().numpy()
        try:
            test_posterior = model.posterior(torch.tensor(test_x).to(device))
            test_mean = test_posterior.mean.squeeze().cpu().numpy()
            test_std = test_posterior.variance.sqrt().squeeze().cpu().numpy()
        except:
            print("______________No test__________________")

    metrics = []
    for task_idx, task_name in enumerate(task_names):
        # Get masks for current task
        train_task_mask = (train_x[:, -1] == task_idx)
        if test_y.shape[0] != 0:
            test_task_mask = (test_x[:, -1] == task_idx)


        if np.sum(train_task_mask) > 0:
            # Get predictions and true values for this task
            train_preds_scaled = train_mean[train_task_mask]
            train_uncertainties_scaled = train_std[train_task_mask]
            train_true_scaled = train_y[train_task_mask]
            if test_y.shape[0] != 0:
                test_preds_scaled = test_mean[test_task_mask]
                test_uncertainties_scaled = test_std[test_task_mask]
                test_true_scaled = test_y[test_task_mask]


            # Descale values
            scaler = output_scalers[task_name]
            train_preds = scaler.inverse_transform(train_preds_scaled.reshape(-1, 1)).ravel()
            train_true = scaler.inverse_transform(train_true_scaled.reshape(-1, 1)).ravel()
            train_uncertainties = train_uncertainties_scaled * np.sqrt(scaler.var_)[0]
            if test_y.shape[0] != 0:
                test_preds = scaler.inverse_transform(test_preds_scaled.reshape(-1, 1)).ravel()
                test_true = scaler.inverse_transform(test_true_scaled.reshape(-1, 1)).ravel()
                test_uncertainties = test_uncertainties_scaled * np.sqrt(scaler.var_)[0]


            # Add prior back if available
            if df_prior is not None and task_name in df_prior.columns:
                # Get indices for current task
                train_task_indices = np.where(train_task_mask)[0]


                # Get corresponding original indices
                train_data_indices = train_orig_idx[train_task_indices]
                # Add prior values
                train_prior = df_prior[task_name].iloc[train_data_indices].values

                if test_y.shape[0] != 0:
                    test_task_indices = np.where(test_task_mask)[0]
                    test_data_indices = test_orig_idx[test_task_indices]
                    test_prior = df_prior[task_name].iloc[test_data_indices].values




                print(f"\nTask: {task_name}")
                print(f"Train prior range: {train_prior.min():.2f} to {train_prior.max():.2f}")
                if test_y.shape[0] != 0:
                    print(f"Test prior range: {test_prior.min():.2f} to {test_prior.max():.2f}")


                train_preds += train_prior
                train_true += train_prior

                if test_y.shape[0] != 0:
                    test_preds += test_prior
                    test_true += test_prior


            # Calculate metrics

            if test_y.shape[0] != 0:
                test_mae = mean_absolute_error(test_true, test_preds)
                test_rmse = np.sqrt(mean_squared_error(test_true, test_preds))
                test_r2 = r2_score(test_true, test_preds)
                test_kendall = kendalltau(test_true, test_preds)[0]
                test_spearman = spearmanr(test_true, test_preds)[0]

                # Additional metrics
                test_log_safe = np.where(test_true <= 0, 1e-10, test_true)
                preds_log_safe = np.where(test_preds <= 0, 1e-10, test_preds)
                gmae = np.mean(np.abs(np.log(test_log_safe) - np.log(preds_log_safe)))

                smape = np.mean(2 * np.abs(test_preds - test_true) /
                              (np.abs(test_preds) + np.abs(test_true))) * 100

                scaling_factor = np.mean(np.abs(np.diff(test_true)))
                mase = np.nan if scaling_factor == 0 else mean_absolute_error(test_true, test_preds) / scaling_factor

                rmspe = np.sqrt(np.mean(np.square((test_true - test_preds) / test_true)))
                metrics.append({
                    'task': task_name,
                    'test_mae': test_mae,
                    'test_rmse': test_rmse,
                    'test_r2': test_r2,
                    'test_kendall': test_kendall,
                    'test_spearman': test_spearman,
                    'gmae': gmae,
                    'smape': smape,
                    'mase': mase,
                    'rmspe': rmspe,
                    'n_train_samples': np.sum(train_task_mask),
                    'n_test_samples': np.sum(test_task_mask)
                })
            else:
                test_preds=train_preds
                test_true=train_true
                test_task_mask = train_task_mask
                test_mae = mean_absolute_error(test_true, test_preds)
                test_rmse = np.sqrt(mean_squared_error(test_true, test_preds))
                test_r2 = r2_score(test_true, test_preds)
                test_kendall = kendalltau(test_true, test_preds)[0]
                test_spearman = spearmanr(test_true, test_preds)[0]

                # Additional metrics
                test_log_safe = np.where(test_true <= 0, 1e-10, test_true)
                preds_log_safe = np.where(test_preds <= 0, 1e-10, test_preds)
                gmae = np.mean(np.abs(np.log(test_log_safe) - np.log(preds_log_safe)))

                smape = np.mean(2 * np.abs(test_preds - test_true) /
                              (np.abs(test_preds) + np.abs(test_true))) * 100

                scaling_factor = np.mean(np.abs(np.diff(test_true)))
                mase = np.nan if scaling_factor == 0 else mean_absolute_error(test_true, test_preds) / scaling_factor

                rmspe = np.sqrt(np.mean(np.square((test_true - test_preds) / test_true)))
                metrics.append({
                    'task': task_name,
                    'test_mae': test_mae,
                    'test_rmse': test_rmse,
                    'test_r2': test_r2,
                    'test_kendall': test_kendall,
                    'test_spearman': test_spearman,
                    'gmae': gmae,
                    'smape': smape,
                    'mase': mase,
                    'rmspe': rmspe,
                    'n_train_samples': np.sum(train_task_mask),
                    'n_test_samples': np.sum(test_task_mask)
                })



            # Create plots
            plt.figure(figsize=(15, 5))

            # Parity plot
            plt.subplot(121)
            plt.errorbar(train_true, train_preds, yerr=2*train_uncertainties,
                        fmt='o', alpha=0.3, capsize=3, markersize=4,
                        elinewidth=1, label='Training', color='blue')

            if test_y.shape[0] != 0:
                all_true = np.concatenate([train_true, test_true])
            else:
                all_true = np.concatenate([train_true])
            plt.plot([min(all_true), max(all_true)],
                    [min(all_true), max(all_true)],
                    'k--', lw=2, label='Perfect prediction')

            plt.xlabel(f'True {task_name}')
            plt.ylabel(f'Predicted {task_name}')
            if test_y.shape[0] != 0:
                plt.errorbar(test_true, test_preds, yerr=2*test_uncertainties,
                          fmt='o', alpha=0.3, capsize=3, markersize=4,
                          elinewidth=1, label='Test', color='red')
            metrics_text = (f'R² = {test_r2:.3f}\n'
                          f'RMSE = {test_rmse:.3f}\n'
                          f'GMAE = {gmae:.3f}\n'
                          f'SMAPE = {smape:.1f}%\n'
                          f'MASE = {mase:.3f}\n'
                          f'RMSPE = {rmspe:.3f}')

            plt.text(0.98, 0.02, metrics_text,
                    transform=plt.gca().transAxes,
                    fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                    verticalalignment='bottom',
                    horizontalalignment='right')

            plt.title(f'{task_name}\nTest Spearman: {test_spearman:.3f}')

            plt.legend()

            # Distribution plot
            plt.subplot(122)
            plt.hist(train_true, bins=20, alpha=0.5, label='Train True', color='blue')
            plt.hist(train_preds, bins=20, alpha=0.5, label='Train Predicted', color='lightblue')
            if test_y.shape[0] !=0 :
                plt.hist(test_true, bins=20, alpha=0.5, label='Test True', color='red')
                plt.hist(test_preds, bins=20, alpha=0.5, label='Test Predicted', color='lightcoral')

            plt.xlabel(f'{task_name}')
            plt.ylabel('Frequency')
            plt.legend()

            plt.tight_layout()
            plt.show()

    return metrics

def compare_models_with_split(input_scaled, output_scaled, input_scalers, output_scalers,input_vars,output_vars, yvar_scaled=None, df_prior=None, reductions=[1, 3, 5, 7, 8, 9]):
    """Compare models with different reduction parameters using train-test split."""
    model_results = {}
    models={}
    for reduction in reductions:
        print(f"\nEvaluating model with reduction={reduction}")

        # Prepare training pairs with indices
        train_x, train_y, train_yvar, original_indices, task_indices = prepare_training_pairs_with_indices(
            input_scaled, output_scaled, yvar_scaled)

        # Create train-test split
        n_samples = len(train_x)
        shuffled_indices = np.random.RandomState(42).permutation(n_samples)
        train_size = int(1.00 * n_samples)

        train_idx = shuffled_indices[:train_size]
        test_idx = shuffled_indices[train_size:]

        # Split the data
        train_x_split = train_x[train_idx]
        train_y_split = train_y[train_idx]
        test_x_split = train_x[test_idx]
        test_y_split = train_y[test_idx]

        # Split Yvar if available
        train_yvar_split = train_yvar[train_idx] if train_yvar is not None else None
        test_yvar_split = train_yvar[test_idx] if train_yvar is not None else None

        # Track original indices
        train_original_indices = original_indices[train_idx]
        test_original_indices = original_indices[test_idx]
        print(train_x_split.shape)
        print(train_y_split.shape)
        print(test_x_split.shape)
        print(test_y_split.shape)
        # Train model
        model = train_model_with_validation(
            train_x_split,
            train_y_split,
            test_x_split,
            test_y_split,
            num_tasks=output_scaled.shape[1],
            train_yvar=train_yvar_split,
            test_yvar=test_yvar_split,
            num_epochs=3000,
            reduction=reduction
        )

        # Rest remains the same...
        # Evaluate model

        metrics = evaluate_model_train_test(
            model,
            train_x_split,
            train_y_split,
            test_x_split,
            test_y_split,
            output_vars,
            output_scalers,
            df_prior=df_prior,
            train_orig_idx=train_original_indices,
            test_orig_idx=test_original_indices
        )

        # model_results[reduction] = metrics
        models[reduction]=model

    return models


import numpy as np

import numpy as np

class IdentityScaler:
    """A scaler that does nothing and reports var_ = [1.0]."""
    def __init__(self):
        self.var_ = np.array([1.0])    # make this a 1‐element array
    def fit(self, X):
        return self
    def transform(self, X):
        return X
    def fit_transform(self, X):
        return X
    def inverse_transform(self, X):
        return X

def prepare_and_standardize_data(
    df, input_vars, output_vars, yvar_cols=None, verbose=True
):
    """Same I/O as before, but all scalers are identity (no actual scaling)."""

    # --- Inputs ---
    input_scaled = np.full_like(df[input_vars].values, np.nan, dtype=np.float64)
    input_scalers = {}
    for j, col in enumerate(input_vars):
        mask = ~np.isnan(df[input_vars].values[:, j])
        if mask.any():
            scaler = IdentityScaler()
            input_scaled[mask, j] = scaler.fit_transform(
                df[input_vars].values[mask, j].reshape(-1, 1)
            ).ravel()
            input_scalers[col] = scaler
            if verbose:
                print(f"Input {col}: {mask.sum()}/{len(mask)} obs (no-op)")

    # --- Outputs ---
    output_scaled = np.full_like(df[output_vars].values, np.nan, dtype=np.float64)
    output_scalers = {}
    for j, col in enumerate(output_vars):
        mask = ~np.isnan(df[output_vars].values[:, j])
        if mask.any():
            scaler = IdentityScaler()
            output_scaled[mask, j] = scaler.fit_transform(
                df[output_vars].values[mask, j].reshape(-1, 1)
            ).ravel()
            output_scalers[col] = scaler
            if verbose:
                print(f"Output {col}: {mask.sum()}/{len(mask)} obs (no-op)")

    # --- Yvar handling ---
    yvar_scaled = None
    if yvar_cols is not None:
        yvar_scaled = np.full_like(df[output_vars].values, 1e-6, dtype=np.float64)
        for j, (out_col, yvar_col) in enumerate(zip(output_vars, yvar_cols)):
            if yvar_col and yvar_col in df.columns:
                mask = ~np.isnan(df[yvar_col].values)
                if mask.any():
                    # since var_ == 1.0 this is just the raw yvar
                    yvar_scaled[mask, j] = df[yvar_col].values[mask] / output_scalers[out_col].var_
                    if verbose:
                        print(f"Yvar for {out_col}: {mask.sum()}/{len(mask)} obs (no-op)")

    return input_scaled, output_scaled, yvar_scaled, input_scalers, output_scalers


def prepare_training_pairs_with_indices(input_scaled, output_scaled, yvar_scaled=None):
    """Create training pairs while keeping track of original indices."""
    n_samples, n_tasks = output_scaled.shape
    n_features = input_scaled.shape[1]

    # Count total valid pairs
    total_pairs = np.sum(~np.isnan(output_scaled))

    train_x = np.zeros((total_pairs, n_features + 1))
    train_y = np.zeros(total_pairs)
    train_yvar = np.zeros(total_pairs) if yvar_scaled is not None else None
    original_indices = np.zeros(total_pairs, dtype=int)
    task_indices = np.zeros(total_pairs, dtype=int)

    idx = 0
    for i in range(n_samples):
        for task in range(n_tasks):
            if not np.isnan(output_scaled[i, task]):
                train_x[idx, :-1] = input_scaled[i]
                train_x[idx, -1] = task
                train_y[idx] = output_scaled[i, task]
                if yvar_scaled is not None:
                    train_yvar[idx] = yvar_scaled[i, task]
                original_indices[idx] = i
                task_indices[idx] = task
                idx += 1

    return train_x, train_y, train_yvar, original_indices, task_indices












def get_acquisition_d(model, train_y_scaled, ref_point):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_y_scaled = train_y_scaled.to(device)
    print(train_y_scaled)

    train_y_scaled2 = train_y_scaled[:, :2]

    print(train_y_scaled2)
    nd_mask = is_non_dominated(train_y_scaled2)
    pareto_y = train_y_scaled2[nd_mask]
    partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=pareto_y)

    return qExpectedHypervolumeImprovement(model=model, ref_point=ref_point.tolist(), partitioning=partitioning)

def get_candidate_d(acq_function, train_x_scaled, xs_scaled):
    num_input = 4
    n_candidates = 1000  # Resolution of grid
    num_obj = 2
    batch_size = 4  # q

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bounds = torch.stack([torch.zeros(num_input), torch.ones(num_input)]).to(device)

    xs_scaled = xs_scaled.to(device)
    train_x_scaled = train_x_scaled.to(device)

    print("______________Optimization SLSQP______________________")
    candidate, _ = optimize_acqf(
        acq_function, 
        q=batch_size, 
        bounds=bounds,
        options={
            "batch_limit": 10,
            "maxiter": 100,
            "nonnegative": True,
            "method": "SLSQP",
            "collapse_bounds": True,
        }, 
        num_restarts=num_restart, 
        raw_samples=raw_samples
    )
    
    
    return candidate



def fit_mtgp(train_x, train_y):
    """
    Fit a Deep Gaussian Process model using MultiTaskGP as a placeholder.
    Will be replaced with actual DGP implementation later.
    """
    device='cuda' if torch.cuda.is_available() else 'cpu'
    # For MultiTaskGP, we need to reshape the data with task indicators
    n_samples = train_x.shape[0]
    n_objectives = train_y.shape[1]
    train_x=train_x.to(device)
    train_y=train_y.to(device)
    # Create task indices
    task_indices = torch.arange(n_objectives).repeat(n_samples, 1).T.reshape(-1, 1).to(device)

    # Repeat input data for each task
    expanded_X = train_x.repeat(n_objectives, 1)

    # Stack all outputs into a single column
    stacked_Y = train_y.T.reshape(-1, 1)

    # Combine inputs and task indices
    X_with_task = torch.cat([expanded_X, task_indices], dim=1)

    # Fit MultiTaskGP model
    model = MultiTaskGP(X_with_task, stacked_Y, task_feature=-1,output_tasks=list(range(num_obj))).to(device)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    return model

def fit_gp(train_x, train_y):
    device='cuda' if torch.cuda.is_available() else 'cpu'

    def build_model(train_x, train_y):

        model = SingleTaskGP(train_x, train_y).to(device)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model


    models = [build_model(train_x, train_y[:, i:i+1]) for i in range(train_y.shape[-1])]
    model = PartialModelListGP(*models, active_tasks=list(range(num_obj))).to(device)
    return model
def fit_dgp(train_x, train_y, num_epochs=500, reduction=3):
    """
    Fit a Deep Gaussian Process surrogate using the MultiTaskDeepGP implementation.

    Args:
        train_x (Tensor): Training inputs, shape (n_samples, n_features).
        train_y (Tensor): Training outputs, shape (n_samples, n_tasks).
        num_epochs (int, optional): Number of training epochs for the DGP. Defaults to 500.
        reduction (int, optional): Reduction parameter for MultiTaskDeepGP. Defaults to 8.

    Returns:
        model: A trained MultiTaskDeepGP model obtained via compare_models_with_split.
    """
    import pandas as pd

    # Step 1: Convert tensors to a pandas DataFrame
    df = pd.DataFrame(train_x.cpu().numpy(), columns=[f"x{i+1}" for i in range(train_x.shape[1])])
    for j in range(train_y.shape[1]):
        df[f"f{j+1}"] = train_y[:, j].cpu().numpy()

    # Define input and output variable names
    input_vars = [f"x{i+1}" for i in range(train_x.shape[1])]
    output_vars = [f"f{j+1}" for j in range(train_y.shape[1])]

    # Step 2: Prepare and standardize the data
    input_scaled, output_scaled, yvar_scaled, input_scalers, output_scalers = prepare_and_standardize_data(
        df,
        input_vars,
        output_vars,
        yvar_cols=None,
        verbose=False
    )

    # Step 3: Use compare_models_with_split to train and retrieve the DGP model
    models = compare_models_with_split(
        input_scaled,
        output_scaled,
        input_scalers,
        output_scalers,
        input_vars,
        output_vars,
        yvar_scaled=None,
        df_prior=None,
        reductions=[reduction]
    )
    model = models[reduction]

    return model





def run_benchmark(surrogate_funcs, surrogate_names, num_queries, init_points):
    """
    Run the optimization with multiple surrogate models and compare performance.

    Args:
        surrogate_funcs (list): List of surrogate fitting functions
        surrogate_names (list): Names of surrogate models for labeling
        num_queries (int): Number of BO iterations
        init_points (int): Number of initial data points

    Returns:
        dict: Contains hypervolumes, train_x, train_y for each surrogate
    """
    device='cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    # Set up initial training data (shared across all surrogates for fair comparison)


    initial_train_x = initial_train_x = torch.rand(init_points, 4)


    # Get initial observations
    initial_train_y = synthetic_objective(initial_train_x).to(device)

    # Set reference point slightly below minimum for HV calculation
    ref_point = torch.zeros(initial_train_y.shape[1]) - 5
    ref_point=ref_point[:num_obj]
    ref_point=ref_point.to(device)
    xs = torch.rand(n_candidates, num_input).to(device)
    # Run optimization with each surrogate
    for i, (surrogate_func, name) in enumerate(zip(surrogate_funcs, surrogate_names)):
        # Copy initial data to ensure fair comparison

        train_x = initial_train_x.clone().to(device)
        train_y = initial_train_y.clone().to(device)

        from sklearn.preprocessing import StandardScaler
        ISCALER    = StandardScaler().fit(torch.cat([train_x,xs],dim=0).cpu())


        xs_scaled=ISCALER.transform(xs.cpu())
        xs_scaled=torch.tensor(xs_scaled).to(device)


        hypervolumes = []
        all_candidates = []
        all_candidate_values = []

        # BO loop
        for _ in range(num_queries):
            train_x_scaled      = ISCALER.transform(train_x.cpu())
            train_x_scaled=torch.tensor(train_x_scaled).to(device)

            OSCALER    = StandardScaler().fit(train_y.cpu())

            train_y_scaled      = OSCALER.transform(train_y.cpu())
            train_y_scaled=torch.tensor(train_y_scaled).to(device)
            # Fit surrogate model
            model = surrogate_func(train_x_scaled, train_y_scaled).to(device)

            # Get acquisition function
            acq_function = get_acquisition_d(model, train_y_scaled, ref_point)

            # Get next candidate
            candidate_scaled = get_candidate_d(acq_function, train_x_scaled,xs_scaled)
            candidate=ISCALER.inverse_transform(candidate_scaled.cpu())
            candidate=torch.tensor(candidate).to(device)

            # Evaluate candidate
            next_y = synthetic_objective(candidate)


            # Save candidate for plotting
            all_candidates.append(candidate)
            all_candidate_values.append(next_y)

            # Update training data
            train_x = torch.cat([train_x, candidate], dim=0)
            train_y = torch.cat([train_y, next_y], dim=0)

            # Pareto and hypervolume update



            train_y_main=train_y[:,:num_obj]
            pareto_y = train_y_main[is_non_dominated(train_y_main)]
            hv = Hypervolume(ref_point).compute(pareto_y)
            hypervolumes.append(hv)



        results[name] = {
            'hypervolumes': torch.tensor(hypervolumes),
            'train_x': train_x,
            'train_y': train_y,
            'initial_x': initial_train_x,
            'initial_y': initial_train_y,
            'all_candidates': all_candidates,
            'all_candidate_values': all_candidate_values
        }

    return results


def plot_hypervolume_comparison(benchmark_results):
    """
    Plot hypervolume progress for all surrogates.
    """
    plt.figure(figsize=(10, 6))
    for name, result in benchmark_results.items():
        plt.plot(range(1, len(result['hypervolumes'])+1),
                 result['hypervolumes'],
                 marker='o',
                 label=name)

    plt.xlabel("Number of Queries")
    plt.ylabel("Hypervolume")
    plt.title("Hypervolume Progress Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"hypervolume_comparison_{index}.png")
    plt.show()


def plot_pareto_comparison(benchmark_results):
    """
    Plot Pareto fronts comparison for any number of surrogate models.
    """
    # Generate points for objective space background
    n_grid = 100
    grid = torch.rand(n_grid*n_grid, num_input)
    objective_values = synthetic_objective(grid)

    plt.figure(figsize=(10, 8))

    # Plot objective space with blue points
    plt.scatter(objective_values[:, 0].cpu().numpy(),
                objective_values[:, 1].cpu().numpy(),
                s=3, alpha=0.3, color='blue', label="Objective Space")

    # Create flexible color and marker schemes
    # Using distinct colors from matplotlib's color cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    # Different marker styles
    markers = ['o', 's', '^', 'D', 'p', '*', 'X', '+', 'v', '<', '>', 'h']

    # Plot each surrogate's Pareto front
    for i, (name, result) in enumerate(benchmark_results.items()):
        train_y = result['train_y']

        train_y_main=train_y[:,:num_obj]
        nd_mask = is_non_dominated(train_y_main)
        pareto_y = train_y_main[nd_mask]

        # Get color and marker (cycling if we have more models than colors/markers)
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        # Plot Pareto front
        plt.scatter(pareto_y[:, 0].cpu().numpy(),
                   pareto_y[:, 1].cpu().numpy(),
                   color=color,
                   marker=marker,
                   s=40,
                   label=f"Pareto Front - {name}")

    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.title("Pareto Front Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"pareto_comparison_{index}.png")
    plt.show()


def plot_bo_queries(result, name):
    """
    Plot BO process like Image 2 for a single surrogate.
    """

    # Unpack data
    train_y = result['train_y']
    initial_y = result['initial_y']
    bo_queries_y = torch.cat([v for v in result['all_candidate_values']], dim=0)

    # Generate points for objective space background
    n_grid = 200
    grid = torch.rand(n_grid*n_grid, num_input)
    objective_values = synthetic_objective(grid)

    # Get Pareto front
    train_y_main=train_y[:,:num_obj]
    nd_mask = is_non_dominated(train_y_main)
    pareto_y = train_y_main[nd_mask]

    plt.figure(figsize=(10, 8))

    # Plot objective space with blue points
    plt.scatter(objective_values[:, 0].cpu().numpy(),
                objective_values[:, 1].cpu().numpy(),
                s=1, alpha=0.3, color='blue', label="Objective Space")

    # Plot initial points
    plt.scatter(initial_y[:, 0].cpu().numpy(),
               initial_y[:, 1].cpu().numpy(),
               color='blue', marker='s', s=40, label="Initial Points")

    # Plot BO queries with numbers
    for i, y in enumerate(bo_queries_y):
        plt.scatter(y[0].item(), y[1].item(),
                   color='gray', edgecolors='black', s=80)
        plt.text(y[0].item(), y[1].item(), str(i+1),
                fontsize=8, ha='center', va='center')

    # Plot Pareto front
    plt.scatter(pareto_y[:, 0].cpu().numpy(),
               pareto_y[:, 1].cpu().numpy(),
               color='red', s=40, label="Pareto Front")

    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.title(f"BO Queries over Objective Space - {name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{name}_bo_queries_{index}.png")
    plt.show()





# Define surrogate functions to benchmark
# surrogate_funcs = [fit_gp, fit_mtgp,fit_dgp]
# surrogate_names = ["GP", "MultiTaskGP","DeepGP"]  # Will rename to DGP later
surrogate_funcs = [fit_gp]
surrogate_names = ["GP"]  # Will rename to DGP later
# Run the benchmark
num_queries=90
init_samples=10
#import pickle
#with open("init_samples.pl","rb") as f:
#    init_samples=pickle.load(f)
# Run the benchmark
benchmark_results = run_benchmark(surrogate_funcs, surrogate_names, num_queries, init_samples)

plot_hypervolume_comparison(benchmark_results)
plot_pareto_comparison(benchmark_results)

# Plot individual BO processes
for name, result in benchmark_results.items():
    plot_bo_queries(result, name)

import pickle
with open(f'benchmark_results-{index}.pkl', 'wb') as f:
    pickle.dump(benchmark_results, f)