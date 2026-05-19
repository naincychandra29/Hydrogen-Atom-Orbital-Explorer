# To Run this Code: 
#         pip install numpy
#         pip install matplotlib
#         pip install scipy
# MAKE SURE to install these libraries before running
            
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm, genlaguerre
import math

# ---------------------
# PARAMETERS
# ---------------------
n, l, m = 5, 4, 4
grid_size = 200
lim = 30

# Memory-efficient grid
x = np.linspace(-lim, lim, grid_size)
y = np.linspace(-lim, lim, grid_size)
z = np.linspace(-lim, lim, grid_size)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# ---------------------
# HYDROGENIC WAVEFUNCTION
# ---------------------
def H_wavefunc(n, l, m, X, Y, Z):
    R = np.sqrt(X**2 + Y**2 + Z**2)
    R[R == 0] = 1e-12  # avoid division by zero

    # spherical coordinates
    rho = 2 * R / n
    theta = np.arccos(Z / R)
    phi = np.arctan2(Y, X)

    # associated Laguerre polynomial
    L = genlaguerre(n - l - 1, 2*l + 1)(rho)

    # spherical harmonics
    Ylm = sph_harm(m, l, phi, theta)

    # correct normalization constant
    coeff = np.sqrt(
        (2.0 / n)**3 * math.factorial(n-l-1) /
        (2*n * math.factorial(n+l))
    )

    psi = coeff * np.exp(-rho/2) * rho**l * L * Ylm
    return psi

# compute probability density
psi = H_wavefunc(n, l, m, X, Y, Z)
density = np.abs(psi)**2

# ---------------------
# VISUALIZATION: central slice
# ---------------------
mid = grid_size // 2
plt.imshow(density[mid], cmap='inferno')
plt.colorbar()
plt.title(f"Hydrogen Orbital |ψ|² Slice (n={n}, l={l}, m={m})")
plt.xlabel("x axis")
plt.ylabel("z axis")
plt.show()
