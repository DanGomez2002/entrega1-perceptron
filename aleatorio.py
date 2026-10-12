"""
aleatorio.py
============
Simulación de datos SIN librerías de números aleatorios (no se usa random ni
numpy). Todas las fórmulas vienen de formulas.py (F11 a F15).

Solo se importa `time`, y únicamente para tomar una semilla distinta cada vez
que el usuario deja la semilla en blanco (no hace ningún cálculo matemático).
"""

import time

from formulas import (congruencial_lineal, uniforme_01, normal_box_muller,
                      uniforme_media_desviacion, exponencial_inversa,
                      binomial_estandarizada)

DISTRIBUCIONES = ("normal", "uniforme", "exponencial", "binomial")


class GeneradorAleatorio:
    """Generador congruencial lineal (F11) con las constantes de
    'Numerical Recipes':  a = 1664525,  c = 1013904223,  m = 2^32."""

    A = 1664525
    C = 1013904223
    M = 2 ** 32

    def __init__(self, semilla=None):
        if semilla is None:
            semilla = time.time_ns()
        self.X = int(semilla) % self.M

    def uniforme01(self):
        """F11: X(k+1) = (a X(k) + c) mod m ;  U = (X + 0.5) / m"""
        self.X = congruencial_lineal(self.X, self.A, self.C, self.M)
        return uniforme_01(self.X, self.M)

    def uniforme(self, a, b):
        """Número uniforme en (a, b):  a + (b - a) * U"""
        return a + (b - a) * self.uniforme01()

    def entero(self, n):
        """Entero en {0, 1, ..., n-1}"""
        return int(self.uniforme01() * n)

    def barajar(self, lista):
        """Algoritmo de Fisher-Yates: desordena la lista en el mismo sitio."""
        for i in range(len(lista) - 1, 0, -1):
            j = self.entero(i + 1)
            lista[i], lista[j] = lista[j], lista[i]

    # --- distribuciones (todas por media mu y desviación sigma) ---------
    def normal(self, mu, sigma):
        return normal_box_muller(self.uniforme01(), self.uniforme01(), mu, sigma)   # F12

    def uniforme_mu_sigma(self, mu, sigma):
        return uniforme_media_desviacion(self.uniforme01(), mu, sigma)              # F13

    def exponencial(self, mu, sigma):
        return exponencial_inversa(self.uniforme01(), mu, sigma)                    # F14

    def binomial(self, mu, sigma, ensayos=20, p=0.5):
        exitos = 0
        for _ in range(ensayos):            # n ensayos de Bernoulli
            if self.uniforme01() < p:
                exitos += 1
        return binomial_estandarizada(exitos, ensayos, p, mu, sigma)               # F15

    def muestra(self, distribucion, mu, sigma):
        if distribucion == "normal":
            return self.normal(mu, sigma)
        if distribucion == "uniforme":
            return self.uniforme_mu_sigma(mu, sigma)
        if distribucion == "exponencial":
            return self.exponencial(mu, sigma)
        if distribucion == "binomial":
            return self.binomial(mu, sigma)
        raise ValueError("Distribución desconocida: " + str(distribucion))


# ----------------------------------------------------------------------
# Generación de los conjuntos de datos
# ----------------------------------------------------------------------
def generar_clase(generador, n, centro, desviacion, distribucion):
    """n puntos 3D alrededor de 'centro'; cada coordenada con la distribución elegida."""
    puntos = []
    for _ in range(n):
        punto = []
        for c in centro:
            punto.append(generador.muestra(distribucion, c, desviacion))
        puntos.append(punto)
    return puntos


def generar_datos_clasificacion(generador, clases):
    """clases: lista de diccionarios {nombre, n, centro, desviacion, distribucion}.
    Devuelve X (puntos) y etiquetas (0, 1, 2 = índice de la clase)."""
    X, etiquetas = [], []
    for indice, clase in enumerate(clases):
        puntos = generar_clase(generador, clase["n"], clase["centro"],
                               clase["desviacion"], clase["distribucion"])
        X.extend(puntos)
        etiquetas.extend([indice] * len(puntos))
    return X, etiquetas


def generar_datos_regresion(generador, config):
    """Modelo lineal:      y = a1*x1 + a2*x2 + a3*x3 + b + ruido
       Modelo cuadrático:  y = a1*x1^2 + a2*x2^2 + a3*x3^2 + b + ruido
    (el cuadrático no es un plano: sirve para 'reventar' el perceptrón)."""
    X, y = [], []
    for _ in range(config["n"]):
        x = []
        for j in range(config["n_entradas"]):
            x.append(generador.muestra(config["distribucion_x"], config["centro_x"][j],
                                       config["desviacion_x"]))
        base = 0.0
        for j in range(config["n_entradas"]):
            if config["modelo"] == "cuadratico":
                base = base + config["coeficientes"][j] * x[j] * x[j]
            else:
                base = base + config["coeficientes"][j] * x[j]
        ruido = generador.muestra(config["distribucion_ruido"], 0.0,
                                  config["desviacion_ruido"])
        X.append(x)
        y.append(base + config["sesgo"] + ruido)
    return X, y
