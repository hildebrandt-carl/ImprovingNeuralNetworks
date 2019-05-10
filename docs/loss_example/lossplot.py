# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.fftpack import ifft2, fft2
from scipy import interpolate
import numpy as np



# Make data.
X = np.arange(-1, 1, 0.3)
Y = np.arange(-1, 1, 0.3)
X, Y = np.meshgrid(X, Y)

H = np.exp(-.5*(X**2+Y**2)/3**2);
Z = ifft2(H**fft2(np.random.randn(7,7))).real;
Z = np.sin(Z)

Z = np.clip(Z,a_min=-1, a_max=0.2)
Z = Z * 2


# interpoloate the surface to smooth it
xnew, ynew = np.mgrid[-1:1:80j, -1:1:80j]
tck = interpolate.bisplrep(X, Y, Z, s=0)
znew = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)

# Plot the surface
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis,
                       linewidth=0, antialiased=True)
# Customize the z axis.
#ax.set_zlim(-0.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_xlabel("weight 1")
ax.set_ylabel("weight 2")
ax.set_zlabel("Loss Function")



# Plot the surface
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(xnew, ynew, znew, cmap=cm.viridis,
                       linewidth=0, antialiased=True)
# Customize the z axis.
#ax.set_zlim(-0.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_xlabel("weight 1")
ax.set_ylabel("weight 2")
ax.set_zlabel("Loss Function")


# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
