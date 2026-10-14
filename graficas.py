"""
graficas.py
===========

Todas las funciones reciben una figura (matplotlib.figure.Figure) y dibujan
dentro de ella, para poder mostrarlas en la interfaz.
"""

from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from formulas import (suma_ponderada, coordenada_del_plano, media, valor_absoluto,
                      minimo, maximo, es_finito)

COLORES_CLASE = ["#1f77b4", "#ff7f0e", "#2ca02c"]
COLORES_PLANO = ["#9467bd", "#d62728", "#7f7f7f"]
COLORES_PESO = ["#1f77b4", "#ff7f0e", "#2ca02c", "#8c564b"]


# ----------------------------------------------------------------------
# Utilidades geométricas (sin librerías)
# ----------------------------------------------------------------------
def _rango(valores, margen=0.1):
    lo, hi = minimo(valores), maximo(valores)
    extra = (hi - lo) * margen if hi > lo else 1.0
    return lo - extra, hi + extra


def _todos_finitos(valores):
    """True si ningún valor es infinito ni NaN (M8)."""
    for v in valores:
        if not es_finito(v):
            return False
    return True


def _recortar(poligono, eje, limite, mayor_igual):
    """Recorta un polígono con la condición p[eje] >= limite (o <= limite).
    Algoritmo de Sutherland-Hodgman: se conservan los vértices que cumplen y
    se agregan los puntos donde una arista cruza el límite (interpolación lineal)."""
    def dentro(p):
        return p[eje] >= limite if mayor_igual else p[eje] <= limite

    resultado = []
    n = len(poligono)
    for i in range(n):
        actual, siguiente = poligono[i], poligono[(i + 1) % n]
        if dentro(actual):
            resultado.append(actual)
        if dentro(actual) != dentro(siguiente):
            t = (limite - actual[eje]) / (siguiente[eje] - actual[eje])
            resultado.append([actual[j] + t * (siguiente[j] - actual[j]) for j in range(3)])
    return resultado


def poligono_plano(w, theta, x0, umbral, limites):
    """Polígono del plano F10 (w1x1 + w2x2 + w3x3 + theta*x0 = umbral) dentro de la
    caja 'limites'. Se despeja la coordenada con el peso de mayor valor absoluto
    (para no dividir entre un número cercano a cero), se evalúa en las 4 esquinas
    del rectángulo de las otras dos coordenadas y se recorta a la caja."""
    k = 0
    for j in range(1, 3):
        if valor_absoluto(w[j]) > valor_absoluto(w[k]):
            k = j
    if valor_absoluto(w[k]) < 1e-12:
        return []
    a, b = [j for j in range(3) if j != k]
    esquinas = [(limites[a][0], limites[b][0]), (limites[a][1], limites[b][0]),
                (limites[a][1], limites[b][1]), (limites[a][0], limites[b][1])]
    poligono = []
    for va, vb in esquinas:
        punto = [0.0, 0.0, 0.0]
        punto[a], punto[b] = va, vb
        punto[k] = coordenada_del_plano(w, theta, x0, umbral, k, {a: va, b: vb})   # F10
        poligono.append(punto)
    poligono = _recortar(poligono, k, limites[k][0], True)
    if poligono:
        poligono = _recortar(poligono, k, limites[k][1], False)
    return poligono


def _preparar_3d(fig, vista):
    fig.clear()
    ax = fig.add_subplot(111, projection="3d")
    if vista:
        ax.view_init(elev=vista[0], azim=vista[1])
    return ax


# ----------------------------------------------------------------------
# 1. Espacio 3D
# ----------------------------------------------------------------------
def dibujar_clasificacion(fig, res, epoca=None, vista=None):
    X, etiquetas, nombres = res["X"], res["etiquetas"], res["nombres"]
    ax = _preparar_3d(fig, vista)
    for c, nombre in enumerate(nombres):
        pts = [x for x, e in zip(X, etiquetas) if e == c]
        ax.scatter([q[0] for q in pts], [q[1] for q in pts], [q[2] for q in pts],
                   color=COLORES_CLASE[c], s=12, label=nombre, depthshade=False)
        if pts:
            # Centroide de la nube (F16 en cada coordenada), marcado con una X
            centro = [media([q[j] for q in pts]) for j in range(3)]
            ax.scatter([centro[0]], [centro[1]], [centro[2]], marker="X", s=170,
                       color=COLORES_CLASE[c], edgecolors="black", linewidths=1.2,
                       depthshade=False, zorder=10)
    todos = X + ([res["punto"][0]] if res.get("punto") else [])
    limites = [_rango([x[j] for x in todos]) for j in range(3)]
    for i, p in enumerate(res["perceptrones"]):
        w, theta = p.pesos_en_epoca(p.epocas if epoca is None else epoca)
        poligono = poligono_plano(w, theta, p.x0, p.activacion.umbral, limites)
        if len(poligono) >= 3:
            color = COLORES_PLANO[i % 3]
            ax.add_collection3d(Poly3DCollection([poligono], facecolors=color,
                                                 edgecolors=color, alpha=0.35))
            ax.plot([], [], color=color, linewidth=8, alpha=0.5, label="Plano " + p.nombre)
    punto = res.get("punto")
    if punto:
        x, clase = punto
        ax.scatter([x[0]], [x[1]], [x[2]], marker="*", s=320, color=COLORES_CLASE[clase],
                   edgecolors="black", linewidths=1.2, depthshade=False, zorder=11,
                   label="Punto nuevo → " + res["nombres"][clase])
    ax.scatter([], [], marker="X", s=80, color="white", edgecolors="black",
               label="Centroide de cada nube")
    ax.set_xlim(*limites[0]); ax.set_ylim(*limites[1]); ax.set_zlim(*limites[2])
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("x3")
    ax.set_title(_titulo_epoca(res["titulo"], res["perceptrones"], epoca), fontsize=11)
    ax.legend(loc="upper left", fontsize=8)
    return ax


def dibujar_regresion(fig, res, epoca=None, vista=None):
    """2 entradas: puntos (x1, x2, y) y el plano y = w1x1 + w2x2 + theta*x0.
    3 entradas: son 4 dimensiones; se fija x3 en su media (F16) y a cada punto se
    le quita el efecto de x3:  y_corregido = y - w3 (x3 - media_x3)."""
    X, y, p = res["X"], res["y"], res["perceptrones"][0]
    w, theta = p.pesos_en_epoca(p.epocas if epoca is None else epoca)
    fig.clear()
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    if vista:
        ax.view_init(elev=vista[0], azim=vista[1])
    x1 = [x[0] for x in X]
    x2 = [x[1] for x in X]
    if p.n_entradas == 2:
        z, fijo, etiqueta_z = y, {}, "y"
        subtitulo = "Plano  y = w1 x1 + w2 x2 + θ·x0"
    else:
        media_x3 = media([x[2] for x in X])                                   # F16
        z = [yi - w[2] * (x[2] - media_x3) for x, yi in zip(X, y)]
        fijo, etiqueta_z = {2: media_x3}, "y corregido"
        subtitulo = "Corte del hiperplano con x3 = %.2f (media)" % media_x3
    l1, l2 = _rango(x1), _rango(x2)
    poligono = []
    for a, b in ((l1[0], l2[0]), (l1[1], l2[0]), (l1[1], l2[1]), (l1[0], l2[1])):
        entrada = [a, b] + ([fijo[2]] if fijo else [])
        poligono.append([a, b, suma_ponderada(w, entrada, theta, p.x0)])      # F1 = F3
    predicho = [suma_ponderada(w, x, theta, p.x0) for x in X]                   # F1 = F3

    # Si el perceptrón divergió, los pesos pueden ser infinitos o NaN (M8):
    # entonces solo se dibujan los datos, porque el plano no se puede dibujar.
    dibujable = _todos_finitos(z + [q[2] for q in poligono] + predicho)
    if not dibujable:
        z, etiqueta_z = y, "y"
        subtitulo = "Pesos infinitos o NaN: el perceptrón divergió (no hay plano)"
    ax.scatter(x1, x2, z, color=COLORES_CLASE[0], s=10, label="Datos", depthshade=False)
    zs = z
    if dibujable:
        ax.add_collection3d(Poly3DCollection([poligono], facecolors=COLORES_PLANO[1],
                                             edgecolors=COLORES_PLANO[1], alpha=0.35))
        ax.plot([], [], color=COLORES_PLANO[1], linewidth=8, alpha=0.5, label="Plano del perceptrón")
        zs = z + [q[2] for q in poligono]
    ax.set_xlim(*l1); ax.set_ylim(*l2); ax.set_zlim(*_rango(zs, 0.05))
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel(etiqueta_z)
    ax.set_title(subtitulo, fontsize=10)
    ax.legend(loc="upper left", fontsize=8)

    ax2 = fig.add_subplot(1, 2, 2)
    if dibujable:
        ax2.scatter(y, predicho, s=10, color=COLORES_CLASE[0], label="Instancias")
    else:
        ax2.text(0.5, 0.5, "Las predicciones son infinitas o NaN\n(el perceptrón divergió)",
                 ha="center", va="center", transform=ax2.transAxes)
    lo, hi = minimo(y), maximo(y)
    ax2.plot([lo, hi], [lo, hi], color=COLORES_PLANO[1], label="Ajuste perfecto (y = x)")
    ax2.set_xlabel("y real (esperado)")
    ax2.set_ylabel("y del perceptrón (obtenido)")
    ax2.set_title("Real vs. predicho", fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig.suptitle(_titulo_epoca(res["titulo"], [p], epoca), fontsize=11)
    fig.subplots_adjust(left=0.01, right=0.97, bottom=0.1, top=0.88, wspace=0.3)
    return ax


def dibujar_ejemplo(fig, res, epoca=None, vista=None):
    """Ejemplo de clase (2 entradas): recta w1 x1 + w2 x2 + theta*x0 = 0."""
    X, d, p = res["X"], res["d"], res["perceptrones"][0]
    w, theta = p.pesos_en_epoca(p.epocas if epoca is None else epoca)
    fig.clear()
    ax = fig.add_subplot(111)
    for valor, color in ((1, COLORES_CLASE[0]), (-1, COLORES_CLASE[1])):
        pts = [x for x, di in zip(X, d) if di == valor]
        ax.scatter([q[0] for q in pts], [q[1] for q in pts], color=color, s=60,
                   label="Salida deseada %d" % valor, zorder=3)
    for i, x in enumerate(X):
        ax.annotate("S%d" % (i + 1), (x[0], x[1]), textcoords="offset points", xytext=(6, 6))
    lx, ly = _rango([x[0] for x in X]), _rango([x[1] for x in X], 0.5)
    if valor_absoluto(w[1]) > 1e-12:
        xs = [lx[0], lx[1]]
        ys = [coordenada_del_plano(w, theta, p.x0, 0.0, 1, {0: v}) for v in xs]   # F10
        ax.plot(xs, ys, color=COLORES_PLANO[1], label="Recta de separación")
    ax.set_xlim(*lx); ax.set_ylim(*ly)
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.set_title(_titulo_epoca("Ejemplo de clase", [p], epoca), fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return ax


# ----------------------------------------------------------------------
# 2. Gráfica de las épocas
# ----------------------------------------------------------------------
def dibujar_epocas(fig, res, epoca=None):
    fig.clear()
    ax = fig.add_subplot(111)
    perceptrones = res["perceptrones"]
    for i, p in enumerate(perceptrones):
        ax.plot(list(range(len(p.historial_error))), p.historial_error, marker="o",
                markersize=3, color=COLORES_PLANO[i % 3], label="%s: %s" % (p.nombre, p.estado()))
    if epoca is not None:
        ax.axvline(epoca, color="black", linestyle="--", linewidth=1, label="Época mostrada")
    if any(p.divergio for p in perceptrones):
        ax.set_yscale("log")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Época")
    ax.set_ylabel("err = ½ Σ (d − y)²   (F7)")
    ax.set_title("Gráfica de las épocas: error por época", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


# ----------------------------------------------------------------------
# 3. Gráfica de los pesos
# ----------------------------------------------------------------------
def dibujar_pesos(fig, res, epoca=None):
    """Todos los pesos de cada perceptrón en una sola gráfica.
    Se registra el valor después de cada instancia; en el eje x se muestra
    la época (número de ajustes / número de instancias)."""
    fig.clear()
    perceptrones = res["perceptrones"]
    n = len(perceptrones)
    for i, p in enumerate(perceptrones):
        ax = fig.add_subplot(1, n, i + 1)
        xs = [k / p.n_instancias for k in range(len(p.historial_pesos))]
        nombres = ["w%d" % (j + 1) for j in range(p.n_entradas)] + ["θ (sesgo)"]
        for j, nombre in enumerate(nombres):
            ax.plot(xs, [h[j] for h in p.historial_pesos], linewidth=1.2,
                    color=COLORES_PESO[j % 4] if j < p.n_entradas else COLORES_PESO[3],
                    label=nombre)
        if epoca is not None:
            ax.axvline(epoca if epoca < p.epocas else p.epocas, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Época")
        if i == 0:
            ax.set_ylabel("Valor del peso")
        ax.set_title("Pesos de " + p.nombre, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Gráfica de los pesos (F5 y F6)", fontsize=11)


def _titulo_epoca(titulo, perceptrones, epoca):
    if epoca is None:
        return titulo + " — " + "; ".join(p.estado() for p in perceptrones)
    return "%s — época %d" % (titulo, epoca)
