import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Configurazione dei dati
# Usiamo un intervallo più ampio data la natura trigonometrica
x = np.linspace(-np.pi, np.pi, 100)
y = np.linspace(-np.pi, np.pi, 100)
X, Y = np.meshgrid(x, y)

# Definizione della funzione Banded Trigonometric per n=2
# Problema 16 dal documento
# i=1: 1 * ((1 - cos(x)) + sin(0) - sin(y))
term_1 = 1 * ((1 - np.cos(X)) + 0 - np.sin(Y))

# i=2: 2 * ((1 - cos(y)) + sin(x) - sin(0))
term_2 = 2 * ((1 - np.cos(Y)) + np.sin(X) - 0)

Z = term_1 + term_2

# Plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Creazione della superficie
surf = ax.plot_surface(X, Y, Z, cmap=cm.plasma, linewidth=0, antialiased=False)

# Etichette e Titolo
ax.set_title('Problema 16: Banded Trigonometric Function (n=2)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('F(x, y)')

# Barra dei colori
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()