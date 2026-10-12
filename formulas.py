"""
formulas.py
===========
TODAS las fórmulas matemáticas del proyecto, escritas a mano.

No se importa ninguna librería: ni math, ni numpy, ni random. Hasta la raíz
cuadrada, el logaritmo natural, el coseno y el número pi se calculan con sus
fórmulas (Newton-Raphson y series de Taylor).

Cada función lleva el número de fórmula (F1, F2, ...) que se usa en el informe.

    PERCEPTRÓN
    F1  Suma ponderada .............. S = w1*x1 + w2*x2 + w3*x3 + theta*x0
    F2  Función escalón ............. y = alta si S >= umbral, si no baja
    F3  Función lineal .............. y = S
    F4  Error de una instancia ...... e = d - y
    F5  Ajuste de un peso ........... wi(nuevo) = wi + gamma * e * xi
    F6  Ajuste del sesgo ............ theta(nuevo) = theta + gamma * e * x0
    F7  Error de la época ........... err = 1/2 * sum(e^2)
    F8  Error cuadrático medio ...... ECM = (1/N) * sum(e^2)
    F9  Exactitud ................... 100 * aciertos / N
    F10 Plano de separación ......... w1*x1 + w2*x2 + w3*x3 + theta*x0 = umbral

    SIMULACIÓN DE DATOS
    F11 Generador congruencial ...... X(k+1) = (a*X(k) + c) mod m ;  U = (X + 0.5) / m
    F12 Normal (Box-Muller) ......... Z = sqrt(-2 ln U1) * cos(2 pi U2) ; X = mu + sigma*Z
    F13 Uniforme .................... X = mu - a + 2a*U ,  a = sigma*sqrt(3)
    F14 Exponencial (inversa) ....... X = (mu - sigma) - sigma * ln(U)
    F15 Binomial estandarizada ...... X = mu + sigma * (B - n p) / sqrt(n p (1 - p))
    F16 Media ....................... media = (1/N) * sum(x)
    F17 Desviación estándar ......... sigma = sqrt( (1/N) * sum((x - media)^2) )
    F18 Exactitud por clase ......... 100 * aciertos de la clase k / N_k
    F19 Coeficiente R² .............. R² = 1 - sum((y - y_pred)^2) / sum((y - media)^2)

    FUNCIONES MATEMÁTICAS BÁSICAS
    M1  Valor absoluto
    M2  Raíz cuadrada (Newton-Raphson)
    M3  Arcotangente (serie de Taylor)  ->  M4  pi (fórmula de Machin)
    M5  Logaritmo natural (serie de atanh)
    M6  Coseno (serie de Taylor)
    M7  Mínimo y máximo de una lista (recorrido simple)
    M8  Número finito (ni infinito ni NaN)
"""

# =====================================================================
# FUNCIONES MATEMÁTICAS BÁSICAS (sin librerías)
# =====================================================================
def valor_absoluto(v):
    """M1:  |v| = v si v >= 0,  -v si v < 0"""
    if v >= 0:
        return v
    return -v


def raiz_cuadrada(a):
    """M2: Newton-Raphson para r^2 = a:
           r(k+1) = 1/2 * ( r(k) + a / r(k) )
    Se repite hasta que r deja de cambiar.
    Mientras r está lejos de la raíz, cada paso solo divide r a la mitad; un
    número tan grande como 1e308 necesita unas 520 mitades antes de acercarse,
    por eso se permiten hasta 1100 pasos (los números normales terminan en pocos)."""
    if a < 0:
        raise ValueError("No existe raíz cuadrada real de un número negativo")
    if a == 0:
        return 0.0
    r = a if a >= 1 else 1.0
    for _ in range(1100):
        r_nuevo = 0.5 * (r + a / r)
        if valor_absoluto(r_nuevo - r) <= 1e-16 * r_nuevo:
            return r_nuevo
        r = r_nuevo
    return r


def arcotangente(z):
    """M3: serie de Taylor (válida para |z| < 1):
           atan(z) = z - z^3/3 + z^5/5 - z^7/7 + ..."""
    suma = 0.0
    potencia = z          # z^(2n+1)
    n = 0
    while True:
        termino = potencia / (2 * n + 1)
        if n % 2 == 1:
            termino = -termino
        suma += termino
        if valor_absoluto(termino) < 1e-18:
            return suma
        potencia = potencia * z * z
        n += 1


# M4: número pi con la fórmula de Machin:  pi = 16*atan(1/5) - 4*atan(1/239)
PI = 16.0 * arcotangente(1.0 / 5.0) - 4.0 * arcotangente(1.0 / 239.0)


def _serie_ln(m):
    """ln(m) = 2 * [ z + z^3/3 + z^5/5 + ... ],  z = (m - 1) / (m + 1)
    (serie de la arcotangente hiperbólica; converge rápido si m está cerca de 1)."""
    z = (m - 1.0) / (m + 1.0)
    z2 = z * z
    potencia = z
    suma = 0.0
    n = 0
    while True:
        termino = potencia / (2 * n + 1)
        suma += termino
        if valor_absoluto(termino) < 1e-18:
            return 2.0 * suma
        potencia = potencia * z2
        n += 1


LN2 = _serie_ln(2.0)   # ln(2) calculado con la misma serie


def logaritmo_natural(a):
    """M5: se escribe a = m * 2^k con 1 <= m < 2, entonces
           ln(a) = k * ln(2) + ln(m)
    y ln(m) se calcula con la serie de la arcotangente hiperbólica."""
    if a <= 0:
        raise ValueError("El logaritmo solo existe para números positivos")
    k = 0
    m = a
    while m >= 2.0:
        m = m / 2.0
        k += 1
    while m < 1.0:
        m = m * 2.0
        k -= 1
    return k * LN2 + _serie_ln(m)


def coseno(angulo):
    """M6: serie de Taylor:
           cos(t) = 1 - t^2/2! + t^4/4! - t^6/6! + ...
    Antes se lleva el ángulo al intervalo [-pi, pi] restando vueltas completas (2 pi)."""
    vuelta = 2.0 * PI
    t = angulo - int(angulo / vuelta) * vuelta
    if t > PI:
        t -= vuelta
    elif t < -PI:
        t += vuelta
    suma = 0.0
    termino = 1.0         # t^0 / 0!
    n = 0
    while valor_absoluto(termino) > 1e-18:
        suma += termino
        # de t^(2n)/(2n)! se pasa a -t^(2n+2)/(2n+2)!
        termino = -termino * t * t / ((2 * n + 1) * (2 * n + 2))
        n += 1
    return suma


def minimo(valores):
    """M7: el menor valor de la lista, comparando uno por uno."""
    menor = valores[0]
    for v in valores:
        if v < menor:
            menor = v
    return menor


def maximo(valores):
    """M7: el mayor valor de la lista, comparando uno por uno."""
    mayor = valores[0]
    for v in valores:
        if v > mayor:
            mayor = v
    return mayor


INFINITO = float("inf")


def es_finito(v):
    """M8: v es un número finito si no es NaN ni infinito.
    NaN es el único valor que no es igual a sí mismo (v != v), y el infinito
    se reconoce comparándolo con INFINITO. Sirve para rechazar datos como
    'nan' o 'inf' y para no dibujar pesos que se volvieron infinitos o NaN
    cuando el perceptrón diverge."""
    if v != v:
        return False
    return v != INFINITO and v != -INFINITO


# =====================================================================
# FÓRMULAS DEL PERCEPTRÓN
# =====================================================================
def suma_ponderada(w, x, theta, x0):
    """F1:  S = w1*x1 + w2*x2 + ... + wn*xn + theta*x0

    Con x0 = -1 (convención de la clase):  S = w1*x1 + w2*x2 + w3*x3 - theta
    Con x0 = +1 (otra bibliografía):       S = w1*x1 + w2*x2 + w3*x3 + b"""
    S = 0.0
    for i in range(len(w)):
        S = S + w[i] * x[i]
    S = S + theta * x0
    return S


def funcion_escalon(S, umbral, salida_alta, salida_baja):
    """F2:  y = salida_alta  si S >= umbral
            y = salida_baja  si S <  umbral"""
    if S >= umbral:
        return salida_alta
    return salida_baja


def funcion_lineal(S):
    """F3:  y = S   (activación de la regresión)"""
    return S


def error_instancia(d, y):
    """F4:  e = d - y   (d = valor esperado, y = valor obtenido)"""
    return d - y


def ajuste_peso(wi, gamma, e, xi):
    """F5:  wi(nuevo) = wi + gamma * (d - y) * xi"""
    return wi + gamma * e * xi


def ajuste_sesgo(theta, gamma, e, x0):
    """F6:  theta(nuevo) = theta + gamma * (d - y) * x0
    Es la MISMA regla de los pesos: la entrada que acompaña al sesgo es x0."""
    return theta + gamma * e * x0


def error_epoca(errores):
    """F7:  err = 1/2 * [ e1^2 + e2^2 + ... + eN^2 ]"""
    suma = 0.0
    for e in errores:
        suma = suma + e * e
    return 0.5 * suma


def error_cuadratico_medio(errores):
    """F8:  ECM = (1/N) * [ e1^2 + e2^2 + ... + eN^2 ]"""
    suma = 0.0
    for e in errores:
        suma = suma + e * e
    return suma / len(errores)


def exactitud(predicciones, reales):
    """F9:  exactitud (%) = 100 * (número de aciertos) / N"""
    aciertos = 0
    for p, r in zip(predicciones, reales):
        if p == r:
            aciertos += 1
    return 100.0 * aciertos / len(reales)


def coordenada_del_plano(w, theta, x0, umbral, k, otros_valores):
    """F10: el plano de separación es donde S = umbral:
                w1*x1 + w2*x2 + w3*x3 + theta*x0 = umbral
    Despejando la coordenada k (las demás se conocen):
                xk = ( umbral - theta*x0 - sum(wj*xj, j != k) ) / wk"""
    suma_otros = 0.0
    for j, valor in otros_valores.items():
        suma_otros = suma_otros + w[j] * valor
    return (umbral - theta * x0 - suma_otros) / w[k]


def media(valores):
    """F16:  media = (x1 + x2 + ... + xN) / N"""
    suma = 0.0
    for v in valores:
        suma = suma + v
    return suma / len(valores)


def desviacion_estandar(valores):
    """F17:  sigma = sqrt( (1/N) * [ (x1 - media)^2 + ... + (xN - media)^2 ] )
    Se usa para comprobar que cada nube simulada tiene la desviación pedida."""
    m = media(valores)
    suma = 0.0
    for v in valores:
        suma = suma + (v - m) * (v - m)
    return raiz_cuadrada(suma / len(valores))


def exactitud_por_clase(predicciones, reales, clase):
    """F18:  exactitud de la clase k (%) = 100 * aciertos de la clase k / N_k
    Muestra si una clase pequeña (desbalance) queda mal clasificada aunque la
    exactitud global se vea alta."""
    aciertos = 0
    total = 0
    for p, r in zip(predicciones, reales):
        if r == clase:
            total += 1
            if p == r:
                aciertos += 1
    if total == 0:
        return 0.0
    return 100.0 * aciertos / total


def coeficiente_determinacion(reales, predichos):
    """F19:  R² = 1 - [ sum((y - y_pred)^2) ] / [ sum((y - media)^2) ]
    Dice qué fracción de la variación de y explica el plano de la regresión:
    1 es un ajuste perfecto y 0 equivale a predecir siempre la media.
    Si todos los y son iguales no hay variación que explicar y se devuelve 0."""
    m = media(reales)
    suma_residuos = 0.0
    suma_total = 0.0
    for y, y_pred in zip(reales, predichos):
        suma_residuos = suma_residuos + (y - y_pred) * (y - y_pred)
        suma_total = suma_total + (y - m) * (y - m)
    if suma_total == 0:
        return 0.0
    return 1.0 - suma_residuos / suma_total


# =====================================================================
# FÓRMULAS DE LA SIMULACIÓN DE DATOS
# =====================================================================
def congruencial_lineal(X, a, c, m):
    """F11:  X(k+1) = (a * X(k) + c) mod m"""
    return (a * X + c) % m


def uniforme_01(X, m):
    """F11:  U = (X + 0.5) / m   ->  número en el intervalo abierto (0, 1)"""
    return (X + 0.5) / m


def normal_box_muller(U1, U2, mu, sigma):
    """F12:  Z = sqrt(-2 ln U1) * cos(2 pi U2)   ~ Normal(0, 1)
             X = mu + sigma * Z"""
    Z = raiz_cuadrada(-2.0 * logaritmo_natural(U1)) * coseno(2.0 * PI * U2)
    return mu + sigma * Z


def uniforme_media_desviacion(U, mu, sigma):
    """F13:  X = (mu - a) + 2a * U,   a = sigma * sqrt(3)
    porque la varianza de una uniforme en [-a, a] es a^2 / 3."""
    a = sigma * raiz_cuadrada(3.0)
    return (mu - a) + 2.0 * a * U


def exponencial_inversa(U, mu, sigma):
    """F14:  E = -sigma * ln(U)  tiene media sigma y desviación sigma.
             X = (mu - sigma) + E   ->  media mu y desviación sigma."""
    E = -sigma * logaritmo_natural(U)
    return (mu - sigma) + E


def binomial_estandarizada(exitos, ensayos, p, mu, sigma):
    """F15:  X = mu + sigma * (B - n p) / sqrt(n p (1 - p))
    B = número de éxitos en n ensayos de Bernoulli con probabilidad p."""
    z = (exitos - ensayos * p) / raiz_cuadrada(ensayos * p * (1.0 - p))
    return mu + sigma * z
