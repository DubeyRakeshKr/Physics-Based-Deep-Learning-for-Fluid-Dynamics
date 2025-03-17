# Physics-Based Deep Learning for Fluid Dynamics


A Python implementation of Physics-Informed Neural Networks (PINNs) for solving partial differential equations related to fluid dynamics, specifically the Navier-Stokes equations.

## Author
**Rakesh Dubey**

## Overview

This repository contains a TensorFlow implementation of Physics-Informed Neural Networks for solving the incompressible Navier-Stokes equations in 2D. The framework uses neural networks to approximate the solution while enforcing physics constraints through the loss function.

### Features

- Solves 2D incompressible Navier-Stokes equations
- Implements lid-driven cavity flow benchmark problem
- Uses automatic differentiation to compute PDE residuals
- Visualization of flow fields and training progress
- Configurable physics parameters (Reynolds number, domain size)

## Requirements

- Python 3.7+
- TensorFlow 2.4+
- NumPy
- Matplotlib

## Installation

```bash
# Clone the repository
git clone https://github.com/rakeshdubey/physics-based-deeplearning.git
cd physics-based-deeplearning

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install tensorflow numpy matplotlib
```

## Usage

Run the script with default parameters:

```bash
python physics_dl_script.py
```

### Command-line Arguments

The script accepts several command-line arguments:

- `--nu`: Kinematic viscosity (default: 0.01)
- `--epochs`: Number of training epochs (default: 10000)
- `--batch`: Batch size (default: 500)
- `--interior`: Number of interior collocation points (default: 10000)
- `--boundary`: Number of boundary points (default: 1000)

Example with custom parameters:

```bash
python physics_dl_script.py --nu 0.001 --epochs 20000 --batch 1000
```

## Physics Background

The code solves the incompressible Navier-Stokes equations:

1. **Continuity equation**: ∇·u = 0
2. **Momentum equations**: 
   - ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u
   - Where u is velocity, p is pressure, ρ is density, and μ is dynamic viscosity

For the lid-driven cavity problem, we use the following boundary conditions:
- Moving lid at the top (y=1) with u=1, v=0
- No-slip conditions (u=v=0) on all other walls

## How It Works

1. **Neural Network Architecture**: The model takes spatial coordinates (x,y) as input and outputs velocity components (u,v) and pressure (p).

2. **Physics-Informed Loss Function**: The loss function includes:
   - PDE residuals for the continuity equation
   - PDE residuals for the x and y momentum equations
   - Boundary condition enforcement

3. **Automatic Differentiation**: TensorFlow's automatic differentiation is used to compute spatial derivatives needed for the PDE residuals.

4. **Training Process**: The model is trained to minimize the combined loss using the Adam optimizer.

## Results

After training, the following results are saved in the `results` directory:

- Model weights at various epochs
- Velocity and pressure field visualizations
- Flow streamlines
- Loss history graph

## Example Output

The code generates visualizations of the flow field at regular intervals during training:

- u-velocity component
- v-velocity component
- Pressure field
- Streamlines colored by velocity magnitude

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707.

2. Cuomo, S., Di Cola, V. S., Giampaolo, F., Rozza, G., Raissi, M., & Piccialli, F. (2022). Scientific Machine Learning through Physics-Informed Neural Networks: Where we are and What's next. arXiv preprint arXiv:2201.05624.

<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/rakeshdubey/physics-based-deeplearning.svg?style=flat-square
[contributors-url]: https://github.com/rakeshdubey/physics-based-deeplearning/graphs/contributors
[license-shield]: https://img.shields.io/github/license/rakeshdubey/physics-based-deeplearning.svg?style=flat-square
[license-url]: https://github.com/rakeshdubey/physics-based-deeplearning/blob/master/LICENSE
