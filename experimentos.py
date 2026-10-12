"""
experimentos.py
===============
Une la simulación de datos (aleatorio.py) con el perceptrón (perceptron.py).
No dibuja nada ni usa librerías: devuelve los resultados para que la
interfaz (main.py) los muestre.
"""

import copy

from aleatorio import (GeneradorAleatorio, generar_datos_clasificacion,
                       generar_datos_regresion)
from perceptron import (Perceptron, Escalon, Lineal, entrenar_tres_clases,
                        predecir_tres_clases, num)
from formulas import (exactitud, exactitud_por_clase, error_instancia, error_cuadratico_medio,
                      media, desviacion_estandar, coeficiente_determinacion)

# ======================================================================
# CONFIGURACIÓN POR DEFECTO
# ======================================================================
CONFIG_CLASIFICACION = {
    "clases": [
        {"nombre": "Clase 1", "n": 100, "centro": [2.0, 2.0, 2.0],
         "desviacion": 1.0, "distribucion": "normal"},
        {"nombre": "Clase 2", "n": 100, "centro": [8.0, 2.0, 5.0],
         "desviacion": 1.0, "distribucion": "normal"},
        {"nombre": "Clase 3", "n": 100, "centro": [5.0, 8.0, 8.0],
         "desviacion": 1.0, "distribucion": "normal"},
    ],
    "n_clases": 2,
    "estrategia": "cascada",   # "cascada" (2 planos) o "uno_vs_resto" (3 planos)
    "tasa": 0.1,               # gamma
    "max_epocas": 100,
    "error_minimo": 0.0,       # épsilon
    "umbral": 0.0,
    "salida_alta": 1,
    "salida_baja": -1,
    "x0": -1,                  # entrada del sesgo: -1 (clase) o +1
    "barajar": True,
    "registrar": False,        # guardar el proceso paso a paso
    "semilla": None,
}

CONFIG_REGRESION = {
    "n_entradas": 3,
    "n": 200,
    "centro_x": [5.0, 5.0, 5.0],
    "desviacion_x": 2.0,
    "distribucion_x": "uniforme",
    "coeficientes": [2.0, -1.5, 0.5],   # y = 2 x1 - 1.5 x2 + 0.5 x3 + 3 + ruido
    "sesgo": 3.0,
    "modelo": "lineal",                 # "lineal" o "cuadratico"
    "desviacion_ruido": 1.0,
    "distribucion_ruido": "normal",
    "tasa": 0.001,
    "max_epocas": 1000,
    "tolerancia_cambio": 0.0001,
    "x0": -1,
    "barajar": True,
    "registrar": False,
    "semilla": None,
}

# Datos del ejemplo de las diapositivas
EJEMPLO_X = [[-5.4, 6.1], [-5.9, 3.7], [-2.8, 2.9], [3.1, 7.7], [5.1, 6.3], [5.2, 4.1]]
EJEMPLO_D = [1, 1, 1, -1, -1, -1]          # clase 0 -> salida 1, clase 1 -> salida -1
EJEMPLO_W = [-2.10137, 3.11960]
EJEMPLO_THETA = 0.191448
EJEMPLO_GAMMA = 0.25


def config_clasificacion():
    return copy.deepcopy(CONFIG_CLASIFICACION)


def config_regresion():
    return copy.deepcopy(CONFIG_REGRESION)


# ======================================================================
def ejecutar_clasificacion(config):
    gen = GeneradorAleatorio(config["semilla"])
    clases = config["clases"][:config["n_clases"]]
    X, etiquetas = generar_datos_clasificacion(gen, clases)
    alta, baja = config["salida_alta"], config["salida_baja"]
    escalon = Escalon(config["umbral"], alta, baja)
    registro = [] if config["registrar"] else None
    parametros = {"max_epocas": config["max_epocas"], "error_minimo": config["error_minimo"],
                  "barajar": config["barajar"], "registro": registro}

    def crear(nombre):
        return Perceptron(3, config["tasa"], escalon, gen, nombre, config["x0"])

    if config["n_clases"] == 2:
        d = [alta if c == 0 else baja for c in etiquetas]
        p = crear("P1 (clase 1 vs clase 2)")
        p.entrenar(X, d, **parametros)
        perceptrones = [p]
        predicciones = [0 if p.predecir(x) == alta else 1 for x in X]
        titulo = "Clasificación de 2 clases"
    else:
        perceptrones = entrenar_tres_clases(X, etiquetas, config["estrategia"], crear,
                                            parametros, alta, baja)
        predicciones = [predecir_tres_clases(perceptrones, x, config["estrategia"], alta)
                        for x in X]
        titulo = "Clasificación de 3 clases (%s)" % config["estrategia"].replace("_", " ")

    # Comprobación de los datos: media (F16) y desviación (F17) que realmente
    # se obtuvieron en cada nube, comparadas con las que se pidieron.
    estadisticas = []
    for k, clase in enumerate(clases):
        pts = [x for x, e in zip(X, etiquetas) if e == k]
        estadisticas.append({
            "n": len(pts),
            "media": [media([q[j] for q in pts]) for j in range(3)],
            "desviacion": media([desviacion_estandar([q[j] for q in pts]) for j in range(3)]),
            "exactitud": exactitud_por_clase(predicciones, etiquetas, k)})          # F18

    return {"tipo": "clasificacion", "X": X, "etiquetas": etiquetas,
            "nombres": [c["nombre"] for c in clases], "perceptrones": perceptrones,
            "estrategia": config["estrategia"], "salida_alta": alta,
            "estadisticas": estadisticas, "punto": None,
            "exactitud": exactitud(predicciones, etiquetas),                       # F9
            "registro": registro, "titulo": titulo, "config": copy.deepcopy(config)}


def ejecutar_regresion(config):
    gen = GeneradorAleatorio(config["semilla"])
    X, y = generar_datos_regresion(gen, config)
    registro = [] if config["registrar"] else None
    p = Perceptron(config["n_entradas"], config["tasa"], Lineal(), gen, "Regresión",
                   config["x0"])
    p.entrenar(X, y, max_epocas=config["max_epocas"],
               tolerancia_cambio=config["tolerancia_cambio"],
               barajar=config["barajar"], registro=registro)
    predichos = [p.predecir(x) for x in X]                                         # F1 = F3
    errores = [error_instancia(yi, yp) for yi, yp in zip(y, predichos)]            # F4
    return {"tipo": "regresion", "X": X, "y": y, "perceptrones": [p],
            "ecm": error_cuadratico_medio(errores),                                # F8
            "r2": coeficiente_determinacion(y, predichos),                         # F19
            "registro": registro,
            "titulo": "Regresión (%d entradas, modelo %s)" % (config["n_entradas"],
                                                              config["modelo"]),
            "config": copy.deepcopy(config)}


def ejecutar_ejemplo_clase():
    """Reproduce el ejemplo de las diapositivas: S4 = 17,31523 (error) y,
    después del ajuste, S5 = -23,91494."""
    gen = GeneradorAleatorio(1)
    p = Perceptron(2, EJEMPLO_GAMMA, Escalon(0.0, 1, -1), gen, "Ejemplo de clase", x0=-1)
    p.w = EJEMPLO_W[:]
    p.theta = EJEMPLO_THETA
    registro = []
    p.entrenar(EJEMPLO_X, EJEMPLO_D, max_epocas=20, barajar=False, registro=registro)
    return {"tipo": "ejemplo", "X": EJEMPLO_X, "d": EJEMPLO_D, "perceptrones": [p],
            "registro": registro, "titulo": "Ejemplo de clase"}


# ======================================================================
def resumen(res):
    """Texto con los resultados y las fórmulas que se usaron."""
    L = ["RESULTADOS: " + res["titulo"], "=" * 60]
    for p in res["perceptrones"]:
        L.append("")
        L.append("%s -> %s" % (p.nombre, p.estado()))
        for j, wj in enumerate(p.w):
            L.append("   w%d = %s" % (j + 1, num(wj)))
        L.append("   theta = %s    (entrada del sesgo x0 = %d)" % (num(p.theta), p.x0))
        L.append("   Plano (F10): " + p.ecuacion_plano())
    L.append("")
    if res["tipo"] == "clasificacion":
        L.append("Exactitud global sobre los datos (F9): %.2f %%" % res["exactitud"])
        L.append("")
        L.append("Datos obtenidos en cada clase (comprobación de la simulación):")
        for nombre, clase, est in zip(res["nombres"], res["config"]["clases"], res["estadisticas"]):
            L.append("   %s: %d instancias | media pedida (%s) -> obtenida (%s)"
                     % (nombre, est["n"], ", ".join("%.2f" % v for v in clase["centro"]),
                        ", ".join("%.2f" % v for v in est["media"])))
            L.append("      desviación pedida %.2f -> obtenida %.2f (F17) | exactitud de la clase %.2f %% (F18)"
                     % (clase["desviacion"], est["desviacion"], est["exactitud"]))
    elif res["tipo"] == "regresion":
        cfg = res["config"]
        p = res["perceptrones"][0]
        L.append("Error cuadrático medio ECM (F8): %s   (varianza del ruido = %.4f)"
                 % (num(res["ecm"]), cfg["desviacion_ruido"] ** 2))
        L.append("Coeficiente de determinación R² (F19): %s   (1 = ajuste perfecto)"
                 % num(res["r2"]))
        L.append("")
        L.append("Comparación con el modelo real:")
        for j, a in enumerate(cfg["coeficientes"][:cfg["n_entradas"]]):
            L.append("   a%d real = %9.4f   |   w%d aprendido = %s" % (j + 1, a, j + 1, num(p.w[j])))
        L.append("   b  real = %9.4f   |   theta*x0     = %s"
                 % (cfg["sesgo"], num(p.theta * p.x0)))
    L += ["", "FÓRMULAS UTILIZADAS (formulas.py)", "-" * 60,
          "F1  S = w1*x1 + w2*x2 + w3*x3 + theta*x0",
          "F2  y = alta si S >= umbral; baja si S < umbral   (clasificación)",
          "F3  y = S                                           (regresión)",
          "F4  e = d - y",
          "F5  wi = wi + gamma * e * xi",
          "F6  theta = theta + gamma * e * x0",
          "F7  err = 1/2 * sum(e^2)   (gráfica de épocas)",
          "F10 plano: w1*x1 + w2*x2 + w3*x3 + theta*x0 = umbral",
          "F16 media = (1/N) * sum(x)          F17 sigma = sqrt((1/N) * sum((x - media)^2))",
          "F18 exactitud de la clase k = 100 * aciertos de k / N_k",
          "F19 R² = 1 - sum((y - y_pred)^2) / sum((y - media)^2)   (regresión)"]
    return "\n".join(L)


def clasificar_punto(res, x):
    """Clasifica un punto nuevo con los perceptrones ya entrenados."""
    alta = res["salida_alta"]
    if len(res["perceptrones"]) == 1:
        return 0 if res["perceptrones"][0].predecir(x) == alta else 1
    return predecir_tres_clases(res["perceptrones"], x, res["estrategia"], alta)
