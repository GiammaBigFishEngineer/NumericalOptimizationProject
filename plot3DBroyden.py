import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Configurazione dei dati
# Definiamo un intervallo per x e y (solitamente tra -2 e 2 per questi problemi)
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)

# Definizione della funzione Broyden Tridiagonal per n=2
# Problema 31 dal documento
# f1 corrisponde a k=1, f2 corrisponde a k=2
# x_0 = 0 e x_3 = 0
term_f1 = (3 - 2*X)*X - 2*Y + 1
term_f2 = (3 - 2*Y)*Y - X + 1

# La funzione finale è 1/2 della somma dei quadrati
Z = 0.5 * (term_f1**2 + term_f2**2)

# Plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Creazione della superficie
surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=False)

# Etichette e Titolo
ax.set_title('Problema 31: Broyden Tridiagonal Function (n=2)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('F(x, y)')

# Barra dei colori
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()