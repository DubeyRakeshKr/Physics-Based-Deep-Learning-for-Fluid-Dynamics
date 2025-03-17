#!/usr/bin/env python3
"""
Physics-Based Deep Learning Training Script

This script implements a basic physics-informed neural network (PINN) for solving
partial differential equations, with a focus on fluid dynamics problems.

Author: Rakesh Dubey
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os
import time

class NavierStokesSolver:
    """
    Neural network solver for 2D incompressible Navier-Stokes equations
    using Physics-Informed Neural Networks (PINNs).
    """
    def __init__(self, domain_size=(1.0, 1.0), nu=0.01, rho=1.0):
        """
        Initialize the Navier-Stokes solver.
        
        Args:
            domain_size: Tuple of (width, height) for the domain
            nu: Kinematic viscosity
            rho: Fluid density
        """
        self.width, self.height = domain_size
        self.nu = nu  # kinematic viscosity
        self.rho = rho  # density
        
        # Neural network parameters
        self.layers = [2, 64, 64, 64, 64, 3]  # Input: (x,y), Output: (u,v,p)
        
        # Build the neural network
        self.model = self._build_model()
        
        # Directory for saving results
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def _build_model(self):
        """Build the PINN model for Navier-Stokes"""
        # Input layer
        inputs = Input(shape=(2,))
        
        # Hidden layers
        x = inputs
        for layer_size in self.layers[1:-1]:
            x = Dense(layer_size, activation='tanh')(x)
        
        # Output layer: u, v, p
        outputs = Dense(self.layers[-1], activation=None)(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        return model
    
    def _compute_pde_residuals(self, x, y):
        """
        Compute the PDE residuals (continuity and momentum equations).
        Uses TensorFlow's automatic differentiation.
        """
        points = tf.stack([x, y], axis=1)
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(points)
            predictions = self.model(points)
            u, v, p = tf.split(predictions, 3, axis=1)
            
            # First-order derivatives
            u_grad = tape.gradient(u, points)
            v_grad = tape.gradient(v, points)
            p_grad = tape.gradient(p, points)
            
            u_x = u_grad[:, 0:1]
            u_y = u_grad[:, 1:2]
            v_x = v_grad[:, 0:1]
            v_y = v_grad[:, 1:2]
            p_x = p_grad[:, 0:1]
            p_y = p_grad[:, 1:2]
        
        # Second-order derivatives
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(points)
            u_x_tensor = tape2.gradient(u, points)[:, 0:1]
            u_y_tensor = tape2.gradient(u, points)[:, 1:2]
            v_x_tensor = tape2.gradient(v, points)[:, 0:1]
            v_y_tensor = tape2.gradient(v, points)[:, 1:2]
        
        u_xx = tape2.gradient(u_x_tensor, points)[:, 0:1]
        u_yy = tape2.gradient(u_y_tensor, points)[:, 1:2]
        v_xx = tape2.gradient(v_x_tensor, points)[:, 0:1]
        v_yy = tape2.gradient(v_y_tensor, points)[:, 1:2]
        
        # Continuity equation: div(u) = 0
        continuity = u_x + v_y
        
        # Momentum equations
        momentum_x = u * u_x + v * u_y + p_x / self.rho - self.nu * (u_xx + u_yy)
        momentum_y = u * v_x + v * v_y + p_y / self.rho - self.nu * (v_xx + v_yy)
        
        return continuity, momentum_x, momentum_y
    
    def custom_loss(self, x_interior, y_interior, x_boundary, y_boundary, u_boundary, v_boundary):
        """
        Custom loss function that combines PDE residuals and boundary conditions.
        """
        # PDE residuals
        continuity, momentum_x, momentum_y = self._compute_pde_residuals(x_interior, y_interior)
        
        # Calculate mean squared error for PDE residuals
        mse_continuity = tf.reduce_mean(tf.square(continuity))
        mse_momentum_x = tf.reduce_mean(tf.square(momentum_x))
        mse_momentum_y = tf.reduce_mean(tf.square(momentum_y))
        
        # Boundary conditions
        boundary_points = tf.stack([x_boundary, y_boundary], axis=1)
        boundary_pred = self.model(boundary_points)
        u_pred, v_pred, _ = tf.split(boundary_pred, 3, axis=1)
        
        # Calculate mean squared error for boundary conditions
        mse_u_boundary = tf.reduce_mean(tf.square(u_pred - u_boundary))
        mse_v_boundary = tf.reduce_mean(tf.square(v_pred - v_boundary))
        
        # Combine all losses with appropriate weights
        total_loss = (mse_continuity + mse_momentum_x + mse_momentum_y + 
                     10.0 * (mse_u_boundary + mse_v_boundary))
        
        return total_loss
    
    def train(self, num_interior=10000, num_boundary=1000, epochs=10000, batch_size=500):
        """
        Train the PINN model.
        
        Args:
            num_interior: Number of interior points for enforcing PDE
            num_boundary: Number of boundary points for enforcing BCs
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        print("Generating training points...")
        
        # Generate interior points (random sampling)
        x_interior = tf.random.uniform((num_interior, 1), 0, self.width)
        y_interior = tf.random.uniform((num_interior, 1), 0, self.height)
        
        # Generate boundary points (lid-driven cavity)
        x_boundary = np.zeros((num_boundary, 1))
        y_boundary = np.zeros((num_boundary, 1))
        u_boundary = np.zeros((num_boundary, 1))
        v_boundary = np.zeros((num_boundary, 1))
        
        # Bottom boundary (y=0)
        n_bottom = num_boundary // 4
        x_bottom = np.linspace(0, self.width, n_bottom).reshape(-1, 1)
        y_bottom = np.zeros_like(x_bottom)
        u_bottom = np.zeros_like(x_bottom)
        v_bottom = np.zeros_like(x_bottom)
        
        # Right boundary (x=width)
        n_right = num_boundary // 4
        x_right = np.ones_like(np.linspace(0, self.height, n_right).reshape(-1, 1)) * self.width
        y_right = np.linspace(0, self.height, n_right).reshape(-1, 1)
        u_right = np.zeros_like(x_right)
        v_right = np.zeros_like(x_right)
        
        # Top boundary (y=height) - Moving lid
        n_top = num_boundary // 4
        x_top = np.linspace(0, self.width, n_top).reshape(-1, 1)
        y_top = np.ones_like(x_top) * self.height
        u_top = np.ones_like(x_top)  # Moving lid with u=1
        v_top = np.zeros_like(x_top)
        
        # Left boundary (x=0)
        n_left = num_boundary - n_bottom - n_right - n_top
        x_left = np.zeros_like(np.linspace(0, self.height, n_left).reshape(-1, 1))
        y_left = np.linspace(0, self.height, n_left).reshape(-1, 1)
        u_left = np.zeros_like(x_left)
        v_left = np.zeros_like(x_left)
        
        # Combine all boundary points
        x_boundary = np.vstack([x_bottom, x_right, x_top, x_left])
        y_boundary = np.vstack([y_bottom, y_right, y_top, y_left])
        u_boundary = np.vstack([u_bottom, u_right, u_top, u_left])
        v_boundary = np.vstack([v_bottom, v_right, v_top, v_left])
        
        # Convert to TensorFlow tensors
        x_boundary = tf.convert_to_tensor(x_boundary, dtype=tf.float32)
        y_boundary = tf.convert_to_tensor(y_boundary, dtype=tf.float32)
        u_boundary = tf.convert_to_tensor(u_boundary, dtype=tf.float32)
        v_boundary = tf.convert_to_tensor(v_boundary, dtype=tf.float32)
        
        # Define training step
        @tf.function
        def train_step(x_int_batch, y_int_batch, x_bnd_batch, y_bnd_batch, u_bnd_batch, v_bnd_batch):
            with tf.GradientTape() as tape:
                loss = self.custom_loss(
                    x_int_batch, y_int_batch, 
                    x_bnd_batch, y_bnd_batch, 
                    u_bnd_batch, v_bnd_batch
                )
            
            # Get gradients and update model
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            
            return loss
        
        # Training loop
        start_time = time.time()
        loss_history = []
        
        print("Starting training...")
        for epoch in range(epochs):
            # Random batch sampling for interior points
            idx_int = np.random.choice(num_interior, batch_size, replace=False)
            x_int_batch = tf.gather(x_interior, idx_int)
            y_int_batch = tf.gather(y_interior, idx_int)
            
            # Random batch sampling for boundary points
            idx_bnd = np.random.choice(num_boundary, batch_size // 5, replace=False)
            x_bnd_batch = tf.gather(x_boundary, idx_bnd)
            y_bnd_batch = tf.gather(y_boundary, idx_bnd)
            u_bnd_batch = tf.gather(u_boundary, idx_bnd)
            v_bnd_batch = tf.gather(v_boundary, idx_bnd)
            
            # Train step
            loss = train_step(x_int_batch, y_int_batch, x_bnd_batch, y_bnd_batch, u_bnd_batch, v_bnd_batch)
            loss_history.append(loss.numpy())
            
            # Print progress
            if epoch % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch}/{epochs}, Loss: {loss.numpy():.6f}, Time: {elapsed:.2f}s")
                
                # Save intermediate results
                if epoch % 1000 == 0:
                    self.save_results(epoch)
        
        # Save final results
        self.save_results(epochs)
        
        # Plot loss history
        plt.figure(figsize=(10, 6))
        plt.semilogy(loss_history)
        plt.title('Loss History')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, 'loss_history.png'))
        
        print(f"Training completed in {time.time() - start_time:.2f} seconds")
        return loss_history
    
    def save_results(self, epoch):
        """Save the model and visualization of current results"""
        # Save model
        self.model.save_weights(os.path.join(self.results_dir, f"model_epoch_{epoch}.h5"))
        
        # Create a grid for visualization
        nx, ny = 100, 100
        x = np.linspace(0, self.width, nx)
        y = np.linspace(0, self.height, ny)
        X, Y = np.meshgrid(x, y)
        
        x_flat = X.flatten().reshape(-1, 1)
        y_flat = Y.flatten().reshape(-1, 1)
        points = np.hstack((x_flat, y_flat))
        
        # Predict on grid
        predictions = self.model.predict(points)
        u = predictions[:, 0].reshape(nx, ny)
        v = predictions[:, 1].reshape(nx, ny)
        p = predictions[:, 2].reshape(nx, ny)
        
        # Calculate velocity magnitude
        vel_mag = np.sqrt(u**2 + v**2)
        
        # Plot velocity field
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.contourf(X, Y, u, 50, cmap='jet')
        plt.colorbar(label='u velocity')
        plt.title(f'u velocity (Epoch {epoch})')
        plt.xlabel('x')
        plt.ylabel('y')
        
        plt.subplot(1, 3, 2)
        plt.contourf(X, Y, v, 50, cmap='jet')
        plt.colorbar(label='v velocity')
        plt.title(f'v velocity (Epoch {epoch})')
        plt.xlabel('x')
        plt.ylabel('y')
        
        plt.subplot(1, 3, 3)
        plt.contourf(X, Y, p, 50, cmap='jet')
        plt.colorbar(label='Pressure')
        plt.title(f'Pressure (Epoch {epoch})')
        plt.xlabel('x')
        plt.ylabel('y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'flow_field_{epoch}.png'))
        
        # Plot velocity vectors
        plt.figure(figsize=(10, 8))
        plt.streamplot(X, Y, u, v, density=1.5, color=vel_mag, cmap='jet', linewidth=1.5)
        plt.colorbar(label='Velocity magnitude')
        plt.title(f'Flow Streamlines (Epoch {epoch})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(os.path.join(self.results_dir, f'streamlines_{epoch}.png'))
        
        plt.close('all')

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Physics-based Deep Learning for Navier-Stokes Equations')
    parser.add_argument('--nu', type=float, default=0.01, help='Kinematic viscosity')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=500, help='Batch size')
    parser.add_argument('--interior', type=int, default=10000, help='Number of interior collocation points')
    parser.add_argument('--boundary', type=int, default=1000, help='Number of boundary points')
    
    args = parser.parse_args()
    
    print("Physics-Based Deep Learning for Navier-Stokes Equations")
    print("======================================================")
    print(f"Kinematic viscosity: {args.nu}")
    print(f"Training with {args.epochs} epochs, batch size {args.batch}")
    print(f"Using {args.interior} interior points and {args.boundary} boundary points")
    
    # Create and train the solver
    solver = NavierStokesSolver(nu=args.nu)
    solver.train(
        num_interior=args.interior, 
        num_boundary=args.boundary, 
        epochs=args.epochs, 
        batch_size=args.batch
    )
    
    print("Simulation completed. Results saved in the 'results' directory.")
