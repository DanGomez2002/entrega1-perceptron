"""
perceptron.py
=============
Perceptrón (una sola neurona) implementado desde cero, SIN librerías.
Cada paso llama a su fórmula de formulas.py:

    S  = w1*x1 + w2*x2 + w3*x3 + theta*x0          F1  suma ponderada
    y  = f(S)                                      F2  escalón  /  F3 lineal
    e  = d - y                                     F4  error de la instancia
    wi = wi + gamma * e * xi                       F5  ajuste de cada peso
    theta = theta + gamma * e * x0                 F6  ajuste del sesgo
    err = 1/2 * sum(e^2)                           F7  error de la época

x0 es la entrada fija del sesgo:
    x0 = -1  ->  S = sum(wi*xi) - theta   (convención de la clase, por defecto)
    x0 = +1  ->  S = sum(wi*xi) + b       (convención de otros libros)
Las dos son equivalentes (ver informe): lo importante es usar el MISMO x0
en la suma (F1) y en el ajuste del sesgo (F6).

La misma clase sirve para clasificación (activación escalón) y para
regresión (activación lineal).
"""

from formulas import (suma_ponderada, funcion_escalon, funcion_lineal,
                      error_instancia, ajuste_peso, ajuste_sesgo, error_epoca,
                      valor_absoluto, minimo)

MAX_LINEAS_REGISTRO = 4000


class Escalon:
    """Función de activación escalón (F2) con umbral y salidas configurables."""

    def __init__(self, umbral=0.0, salida_alta=1, salida_baja=-1):
        self.umbral = umbral
        self.salida_alta = salida_alta
        self.salida_baja = salida_baja
        self.nombre = "escalón"

    def __call__(self, S):
        return funcion_escalon(S, self.umbral, self.salida_alta, self.salida_baja)


class Lineal:
    """Función de activación lineal (F3): y = S."""

    def __init__(self):
        self.umbral = 0.0
        self.nombre = "lineal"

    def __call__(self, S):
        return funcion_lineal(S)


class Perceptron:

    def __init__(self, n_entradas, tasa_aprendizaje, activacion, generador,
                 nombre="Perceptrón", x0=-1):
        self.n_entradas = n_entradas
        self.gamma = tasa_aprendizaje
        self.activacion = activacion
        self.generador = generador
        self.nombre = nombre
        self.x0 = x0
        # Pesos y sesgo iniciales aleatorios en (-1, 1)   (w0 <- rand())
        self.w = [generador.uniforme(-1.0, 1.0) for _ in range(n_entradas)]
        self.theta = generador.uniforme(-1.0, 1.0)
        # Historiales para las gráficas
        self.historial_error = []   # error de cada época (la 0 = pesos iniciales)
        self.historial_pesos = []   # [w1, ..., wn, theta] después de cada ajuste
        self.n_instancias = 0
        self.epocas = 0
        self.convergio = False
        self.estancado = False      # se detuvo porque el error dejó de bajar
        self.divergio = False

    # ------------------------------------------------------------------
    def suma(self, x):
        return suma_ponderada(self.w, x, self.theta, self.x0)          # F1

    def predecir(self, x):
        return self.activacion(self.suma(x))                            # F2 o F3

    def pesos_actuales(self):
        return self.w[:] + [self.theta]

    def pesos_en_epoca(self, k):
        """(w, theta) al final de la época k (k = 0: pesos iniciales)."""
        if k > self.epocas:
            k = self.epocas
        fila = self.historial_pesos[k * self.n_instancias]
        return fila[:-1], fila[-1]

    def error_conjunto(self, X, d):
        """F7 sobre TODO el conjunto con los pesos actuales (la 'repasadita')."""
        errores = []
        for x, di in zip(X, d):
            errores.append(error_instancia(di, self.predecir(x)))       # F4
        return error_epoca(errores)                                     # F7

    # ------------------------------------------------------------------
    def ajustar(self, x, d):
        """Una iteración: multiplicar, sumar, comparar y ajustar."""
        S = self.suma(x)                                                # F1
        y = self.activacion(S)                                          # F2 / F3
        e = error_instancia(d, y)                                       # F4
        if e != 0:
            for i in range(self.n_entradas):
                self.w[i] = ajuste_peso(self.w[i], self.gamma, e, x[i])      # F5
            self.theta = ajuste_sesgo(self.theta, self.gamma, e, self.x0)    # F6
        return S, y, e

    def entrenar(self, X, d, max_epocas=100, error_minimo=0.0,
                 tolerancia_cambio=None, paciencia=20, barajar=True, registro=None):
        """Entrena por épocas (una época = pasar una vez por todas las instancias).

        Se detiene cuando:
          - err <= error_minimo (convergió), o
          - en las últimas 'paciencia' épocas el error no mejoró más que
            'tolerancia_cambio' (proporción) respecto al mejor anterior
            (se estabilizó; se usa en regresión, donde el ruido no deja que
            err llegue a 0), o
          - el error explota (tasa de aprendizaje demasiado grande), o
          - se llega a max_epocas (no convergió).
        Si 'registro' es una lista, se le agregan las líneas del proceso paso a paso.
        """
        self.n_instancias = len(X)
        self.historial_error = [self.error_conjunto(X, d)]   # época 0
        self.historial_pesos = [self.pesos_actuales()]
        self.convergio = False
        self.estancado = False
        self.divergio = False
        self.epocas = 0
        orden = list(range(len(X)))

        for epoca in range(1, max_epocas + 1):
            if barajar:
                self.generador.barajar(orden)
            _anotar(registro, "\n=== %s | Época %d ===" % (self.nombre, epoca))

            for i in orden:
                antes = self.pesos_actuales()
                S, y, e = self.ajustar(X[i], d[i])
                self.historial_pesos.append(self.pesos_actuales())
                if registro is not None:
                    _anotar_paso(registro, i, X[i], S, y, d[i], e, antes,
                                 self.pesos_actuales())

            err = self.error_conjunto(X, d)
            self.historial_error.append(err)
            self.epocas = epoca
            _anotar(registro, "  -> err de la época %d = %.6f" % (epoca, err))

            if err != err or err > 1e100:                 # NaN o infinito
                self.divergio = True
                break
            if err <= error_minimo:
                self.convergio = True
                break
            if tolerancia_cambio is not None and len(self.historial_error) > paciencia:
                mejor_antes = minimo(self.historial_error[:-paciencia])
                mejor_reciente = minimo(self.historial_error[-paciencia:])
                if mejor_reciente >= (1.0 - tolerancia_cambio) * mejor_antes:
                    self.estancado = True
                    break
        return self.historial_error

    def estado(self):
        """Por qué terminó el entrenamiento. 'Se estabilizó' quiere decir que el
        error dejó de bajar (criterio de estancamiento): eso NO dice si el ajuste
        es bueno; para saberlo se miran el ECM (F8) y el R² (F19)."""
        if self.divergio:
            return "DIVERGIÓ en la época %d" % self.epocas
        if self.estancado:
            return "se estabilizó en %d época%s (el error dejó de bajar)" % (
                self.epocas, "" if self.epocas == 1 else "s")
        if self.convergio:
            return "convergió en %d época%s" % (self.epocas, "" if self.epocas == 1 else "s")
        return "NO convergió (%d épocas)" % self.epocas

    def ecuacion_plano(self):
        """Texto de F10:  w1*x1 + w2*x2 + w3*x3 + theta*x0 = umbral."""
        partes = []
        for j in range(self.n_entradas):
            partes.append("(%s)x%d" % (num(self.w[j]), j + 1))
        return "%s + (%s)(%d) = %s" % (" + ".join(partes), num(self.theta), self.x0,
                                       num(self.activacion.umbral))


# ----------------------------------------------------------------------
def num(v):
    """Número corto: decimales, o notación científica si es enorme (divergencia)."""
    if valor_absoluto(v) < 1e6:
        return "%.5f" % v
    return "%.3e" % v


def _anotar(registro, texto):
    if registro is None:
        return
    if len(registro) < MAX_LINEAS_REGISTRO:
        registro.append(texto)
    elif len(registro) == MAX_LINEAS_REGISTRO:
        registro.append("... (se omiten las demás líneas; reduzca instancias o épocas)")


def _anotar_paso(registro, i, x, S, y, d, e, antes, despues):
    xs = ", ".join("%.3f" % v for v in x)
    _anotar(registro, "  Instancia %d: x=(%s)  S=%s  y=%s  d=%s  e=d-y=%s"
            % (i + 1, xs, num(S), _f(y), _f(d), _f(e)))
    if antes != despues:
        _anotar(registro, "      ajuste [w..., theta]: [%s] -> [%s]"
                % (", ".join(num(v) for v in antes), ", ".join(num(v) for v in despues)))


def _f(v):
    return str(v) if isinstance(v, int) else num(v)


# ----------------------------------------------------------------------
# Tres clases con perceptrones INDEPENDIENTES (no es una red: las neuronas no
# se conectan entre sí ni se entrenan juntas)
# ----------------------------------------------------------------------
def entrenar_tres_clases(X, etiquetas, estrategia, crear_perceptron, parametros,
                         salida_alta=1, salida_baja=-1):
    """
    "cascada" (2 planos):
        P1: clase 1 (+) contra clases 2 y 3 (-)   -> entrena con TODOS los datos
        P2: clase 2 (+) contra clase 3 (-)        -> entrena SOLO con clases 2 y 3
    "uno_vs_resto" (3 planos):
        Pk: clase k (+) contra las otras dos (-)
    """
    if estrategia == "cascada":
        d1 = [salida_alta if c == 0 else salida_baja for c in etiquetas]
        p1 = crear_perceptron("P1 (clase 1 vs resto)")
        p1.entrenar(X, d1, **parametros)
        X2 = [x for x, c in zip(X, etiquetas) if c != 0]
        d2 = [salida_alta if c == 1 else salida_baja for c in etiquetas if c != 0]
        p2 = crear_perceptron("P2 (clase 2 vs clase 3)")
        p2.entrenar(X2, d2, **parametros)
        return [p1, p2]
    perceptrones = []
    for k in range(3):
        dk = [salida_alta if c == k else salida_baja for c in etiquetas]
        pk = crear_perceptron("P%d (clase %d vs resto)" % (k + 1, k + 1))
        pk.entrenar(X, dk, **parametros)
        perceptrones.append(pk)
    return perceptrones


def predecir_tres_clases(perceptrones, x, estrategia, salida_alta=1):
    """Cascada: si P1 se activa es clase 1; si no, P2 decide entre 2 y 3.
    Uno contra el resto: la clase de la neurona con mayor S - umbral."""
    if estrategia == "cascada":
        if perceptrones[0].predecir(x) == salida_alta:
            return 0
        if perceptrones[1].predecir(x) == salida_alta:
            return 1
        return 2
    mejor, mejor_valor = 0, None
    for k, p in enumerate(perceptrones):
        valor = p.suma(x) - p.activacion.umbral
        if mejor_valor is None or valor > mejor_valor:
            mejor, mejor_valor = k, valor
    return mejor
