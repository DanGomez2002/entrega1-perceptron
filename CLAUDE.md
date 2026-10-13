# Contexto del proyecto: Perceptron_Computacional_I

Proyecto de Carlos Daniel Gómez Murcia (Daniel), estudiante de Ingeniería de Sistemas, para el curso Inteligencia Computacional 1 (Computacional I) de la Universidad de la Amazonia, con el profesor Jesús Emilio Pinto Lopera. Es la primera entrega del curso: un simulador del perceptrón en Python que resuelve clasificación de dos y tres clases y regresión sobre datos simulados en tres dimensiones, acompañado de un informe en Word. El script y el informe se envían por correo antes de la clase del 7 de octubre de 2026, y ese día hay sustentación, en la que una sola persona expone, por lo que todo el código debe poder explicarse línea por línea.

Esta carpeta es la versión unificada y definitiva. Este es el único archivo de contexto del proyecto: cubre tanto el código como la app de estudio («Estudio del perceptrón · Sustentación.html», en la raíz del proyecto). Combina dos versiones anteriores (la primera Perceptron_Computacional_I y perceptron_simulador, hecha con Claude Code). La carpeta perceptron_simulador quedó superada: no se debe volver a ella ni mezclar código de allí.

## REGLA OBLIGATORIA: cero librerías para la matemática

Las librerías solo se permiten para lo visual: tkinter para la ventana y matplotlib para dibujar. Toda la matemática del perceptrón y de la simulación está escrita a mano. En ningún archivo se puede importar math, random, numpy, scipy, statistics, sklearn, pandas ni nada que calcule por nosotros.

Tampoco se usan las funciones integradas de Python que calculan, como min, max, sum o abs, en la lógica ni en las gráficas. En su lugar están minimo, maximo, valor_absoluto, media y desviacion_estandar de formulas.py. Si hace falta una operación nueva, se escribe primero en formulas.py, con su código de fórmula y un docstring que la explique, y luego se usa desde los demás archivos.

Los únicos import que no son de dibujo, y no hacen matemática, son tres:
- `time` en aleatorio.py, que solo toma una semilla cuando el usuario la deja vacía.
- `copy` en experimentos.py, que duplica la configuración por defecto.
- `ctypes` en main.py, que solo mejora la nitidez de la ventana en Windows.

Antes de dar por terminado cualquier cambio hay que revisar los import de todos los archivos y buscar min(, max(, sum( y abs( fuera de comentarios. Si una petición choca con esta regla, hay que avisarle a Daniel en lugar de saltársela.

## Lo que pidió el profesor (según la grabación de clase)

**Datos y entrenamiento:**
- Generar nubes de puntos 3D con distribuciones de probabilidad propias.
- Entrenar un perceptrón de regresión que ajuste un plano a la nube y perceptrones de clasificación de 2 y 3 clases.
- Para tres clases se permiten varias neuronas, siempre independientes y sin formar una red.

**Activación y salidas:**
- Activación escalón con umbral configurable.
- El sesgo se resta, así que x0 = −1 en el ajuste.
- Las salidas pueden ser 1/−1, 1/0 «o las que quieran».

**Cambios en vivo para «reventar» el perceptrón:**
- Desde la interfaz se cambian las instancias por clase, la media en los tres ejes, la desviación, la distribución de cada clase, las épocas y la tasa de aprendizaje.
- Para reventarlo se acercan las clases, se aumenta σ o se desbalancea 100 contra 10.

**Tres salidas visuales obligatorias:**
- El espacio 3D con los planos.
- El error por época.
- Una sola gráfica con todos los pesos.

**Informe:** empieza con introducción y sigue con un desarrollo metodológico que explique el código.

## Estructura (dependencias en un solo sentido, de abajo hacia arriba)

| Archivo | Capa | Contenido | Importa |
|---|---|---|---|
| formulas.py | Matemática | Todas las fórmulas F1 a F19 y las funciones básicas M1 a M8 | Nada |
| aleatorio.py | Simulación | Clase GeneradorAleatorio (congruencial lineal con constantes de Numerical Recipes: a = 1664525, c = 1013904223, m = 2^32), barajado Fisher-Yates, distribuciones normal, uniforme, exponencial y binomial (por media μ y desviación σ) y generación de los conjuntos de datos | formulas, time |
| perceptron.py | Modelo | Clases Escalon, Lineal y Perceptron (suma, ajustar, entrenar con criterios de parada), entrenar_tres_clases y predecir_tres_clases (cascada y uno_vs_resto) | formulas |
| experimentos.py | Coordinación | Configuración por defecto, ejecutar_clasificacion (con estadísticas y exactitud por clase), ejecutar_regresion, ejecutar_ejemplo_clase, clasificar_punto y resumen | módulos propios, copy |
| graficas.py | Visual | Espacio 3D con planos, centroides (X) y punto nuevo (estrella); gráfica de épocas; gráfica de pesos; recorte de planos con Sutherland-Hodgman escrito a mano | formulas, matplotlib |
| main.py | Visual | Interfaz tkinter. Es el archivo que se ejecuta | experimentos, graficas, aleatorio, formulas, tkinter, matplotlib |

También están LEEME.txt, requirements.txt (solo matplotlib), la carpeta .vscode (launch, tasks, settings, extensions) y el informe Informe_Perceptron_Carlos_Gomez_Murcia.docx.

### Fórmulas de formulas.py (los códigos se repiten en el código, la interfaz y el informe)

**Perceptrón:**
- F1 suma ponderada: S = w1·x1 + w2·x2 + w3·x3 + θ·x0
- F2 escalón: y = alta si S ≥ umbral, si no baja
- F3 lineal: y = S
- F4 error: e = d − y
- F5 ajuste de peso: wi ← wi + γ·e·xi
- F6 ajuste del sesgo: θ ← θ + γ·e·x0
- F7 error de la época: err = ½ Σe²
- F8 error cuadrático medio: ECM = (1/N) Σe²
- F9 exactitud: 100 · aciertos / N
- F10 plano: Σwi·xi + θ·x0 = umbral

**Simulación de datos:**
- F11 congruencial: X ← (aX + c) mod m, con U = (X + 0,5)/m
- F12 normal (Box-Muller)
- F13 uniforme con a = σ√3
- F14 exponencial por transformada inversa: X = (μ − σ) − σ ln U
- F15 binomial estandarizada (20 ensayos, p = 0,5)

**Evaluación:**
- F16 media
- F17 desviación estándar
- F18 exactitud por clase: 100 · aciertos_k / N_k
- F19 coeficiente de determinación: R² = 1 − Σ(y − ŷ)² / Σ(y − media)² (coeficiente_determinacion; se calcula en ejecutar_regresion y sale en el resumen y la barra de estado)

**Funciones básicas:**
- M1 valor absoluto
- M2 raíz cuadrada por Newton-Raphson (hasta 1100 pasos, para que también converja con números enormes)
- M3 arcotangente por serie de Taylor
- M4 π por la fórmula de Machin
- M5 logaritmo natural por la serie de atanh con reducción de rango
- M6 coseno por serie de Taylor con reducción a [−π, π]
- M7 mínimo y máximo
- M8 número finito (es_finito: ni NaN ni infinito), usado para validar datos y para no dibujar pesos infinitos

## Cómo ejecutar

1. En VS Code: Archivo > Abrir carpeta > Perceptron_Computacional_I.
2. Si no existe la carpeta .venv, abrir Terminal > Ejecutar tarea > «Preparar entorno», que crea el entorno e instala matplotlib. El equivalente manual en Windows es:

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

3. Elegir el intérprete .venv (Ctrl+Shift+P > Python: Select Interpreter).
4. Abrir main.py y presionar F5, o Ctrl+Shift+B, que ejecuta la tarea «Abrir el simulador».

Desde la terminal: `.venv\Scripts\python main.py`.

Ojo: si se ejecuta con el Python del sistema, falla porque matplotlib solo está en el .venv.

Para probar la lógica sin abrir la ventana se puede usar experimentos.py directamente:

```
.venv\Scripts\python -c "import experimentos as ex; c = ex.config_clasificacion(); c['semilla'] = 7; r = ex.ejecutar_clasificacion(c); print(ex.resumen(r))"
```

Funciona igual con ex.config_regresion() y ex.ejecutar_regresion(c), y con ex.ejecutar_ejemplo_clase().

## Lo que hace la interfaz (main.py)

### Pestaña Clasificación
- **Datos de cada clase:** 2 o 3 clases; por clase se fijan instancias, media en x1, x2 y x3, desviación y distribución.
- **Escenarios** (ESCENARIOS al inicio de main.py):
  - Separadas
  - Cercanas
  - Solapadas (con exponencial en la clase 2)
  - Desbalanceadas (100, 10 y 100 instancias)
- **Fila Reventar:**
  - Acercar clases y Alejar clases (método _mover_medias, factor 0,75 o 1,33 respecto al centro común).
  - Aumentar σ y Reducir σ (método _escalar_sigma, factor 1,3).
  - Los cuatro botones entrenan de inmediato.
- **Parámetros del perceptrón:**
  - tasa γ, épocas máximas, error mínimo ε y umbral;
  - salidas del escalón, que admiten cualquier par de números siempre que la salida alta sea mayor que la baja (leer_salida y la validación en _config_clasificacion; con las salidas al revés la regla de ajuste no aprende);
  - validaciones de la entrada: leer_float rechaza nan e inf, leer_no_negativo exige σ ≥ 0 y leer_tasa exige γ > 0 (también en la pestaña Regresión);
  - x0 (−1 o +1), estrategia de tres clases y semilla;
  - barajar instancias y registrar el proceso paso a paso.
- **Clasificar un punto nuevo:** después de entrenar, dice la clase de un punto y lo marca con una estrella.

### Pestaña Regresión
2 o 3 entradas, modelo lineal o cuadrático, coeficientes, ruido y distribución. Por defecto el modelo es y = 2x1 − 1,5x2 + 0,5x3 + 3 + ruido, con tasa 0,001 y 1000 épocas, y se detiene por estancamiento (mejora menor a 0,01 % en 20 épocas). En ese caso Perceptron.estancado queda en True y estado() dice «se estabilizó en N épocas (el error dejó de bajar)», no «convergió»: que el error deje de bajar no dice si el ajuste es bueno, eso lo dicen el ECM y el R².

### Pestaña Ejemplo de clase
Reproduce las cuentas de las diapositivas: S4 = 17,31522, ajuste a w = (−3,65137; −0,73040) y θ = 0,691448, y S5 = −23,91495.

### Panel de resultados
- **Pestañas:** Espacio 3D, Épocas, Pesos, Proceso paso a paso, y Resumen y fórmulas. El resumen trae media y σ pedidas frente a obtenidas y la exactitud de cada clase.
- **Barra de estado:** estado de cada perceptrón y exactitud global y por clase; en regresión, ECM y R².
- **Debajo de las pestañas:** la barra «Época mostrada», el botón «Animar entrenamiento» y «Reiniciar todo» (Ctrl+R).

## Resultados de referencia (semilla 7, usados en el informe)

Si un cambio altera estos números, el informe deja de coincidir y hay que avisarle a Daniel.

**Clasificación:**

| Escenario | Clases y estrategia | Resultado | Exactitud global | Por clase |
|---|---|---|---|---|
| Separadas | 2 clases | Convergió en 6 épocas | 100 % | — |
| Solapadas | 2 clases | No convergió (100 épocas) | 61,5 % | 69 % y 54 % |
| Separadas | 3 clases, cascada | Convergió en 1 y 1 épocas | 100 % | — |
| Separadas | 3 clases, uno contra el resto | Convergió en 1, 13 y 9 épocas | 100 % | — |
| Cercanas | 3 clases, cascada | No convergió | 77 % | 63, 94 y 74 % |
| Cercanas | 2 clases (100/100) | No convergió | 89 % | 81 y 97 % |
| Desbalanceadas | 2 clases (100/10) | No convergió | 95,5 % | 100 y 50 % |

**Regresión:**
- **Lineal:** se estabilizó en 265 épocas. Aprendió w = (2,002; −1,495; 0,454) y un sesgo θ·x0 = 3,169, con ECM 1,076 y R² 0,964.
- **Cuadrático:** se estabilizó en 181 épocas (el informe explica que el ajuste es deficiente por el ECM), con ECM 81,5 y R² 0,972.
- **Efecto de la tasa:** con γ = 0,005 se estabiliza en 100 épocas, con 0,01 en 63, con 0,02 en 24, y con 0,05 diverge en la primera época.

## Relación con el informe

El informe (unas 50 páginas, formato APA, Times New Roman 12 a doble espacio) describe este código. Cita por nombre los archivos, las funciones, los códigos de fórmula F1 a F19 y M1 a M8, y los resultados de arriba, e incluye listados extraídos del código real. El índice, la lista de figuras y la de tablas son texto escrito a mano (no campos de Word): sus números de página se corrigieron el 2 de octubre de 2026 comparándolos con la paginación real de Word, y hay que revisarlos si se agrega texto. Las Figuras 12, 13 y 15 se regeneraron con graficas.py (semilla 7, 130 dpi, mismos píxeles; las 3D con vista (20, −60)). El 12 de octubre de 2026, tras una revisión externa, se corrigieron cuatro frases del informe: graficas.py es «el módulo que dibuja con matplotlib», no el único que lo importa (main.py también lo importa para incrustar las figuras); en Solapadas, lo que invade a la clase 1 son los puntos de la exponencial acumulados cerca de su mínimo, no su cola larga; el ECM de 1,076 está muy cerca del de mínimos cuadrados, 1,063, y no es «el menor error posible»; y el barajado se desactiva en el ejemplo de las diapositivas. Se comprobó con Word que las 52 páginas y los números del índice no cambiaron. Para cambios así en el .docx se reemplaza el texto directamente en word/document.xml y se compara la paginación con Word por COM.

Cualquier cambio en nombres de funciones, fórmulas, valores por defecto o resultados debe reportarse a Daniel para actualizar el informe. No se debe modificar el .docx desde Claude Code sin que él lo pida. Queda pendiente que Daniel confirme el nombre oficial del curso en la portada.

## Forma de trabajar

- Editar los archivos directamente en esta carpeta.
- Después de cada cambio, ejecutar main.py y comprobar que abre sin errores y que los tres modos entrenan.
- Mantener LEEME.txt al día con cada funcionalidad nueva.
- Respetar el estilo actual: comentarios, docstrings y textos de la interfaz en español, con nombres claros que Daniel pueda explicar en la sustentación.
- Cada fórmula nueva lleva su código (F19, M8...) en el docstring.
- Los decimales en la interfaz se aceptan con coma o con punto.
- En explicaciones para Daniel o textos del informe, usar párrafos fluidos que conecten unas ideas con otras, en lugar de frases cortas y sueltas.

## App de estudio (Estudio del perceptrón · Sustentación.html)

El archivo «Estudio del perceptrón · Sustentación.html», en la raíz del proyecto (no en una carpeta estudio/), es una aplicación web de una sola página que Daniel usa para preparar la sustentación. No se entrega al profesor, pero explica este código, así que **cada vez que el código cambie hay que actualizarla** para que lo que Daniel estudia coincida con lo que va a sustentar. Si Daniel pide un cambio en el proyecto, al terminar se le pregunta si también quiere actualizar la app.

### Cómo está hecha

Es un solo archivo HTML con el CSS y el JavaScript adentro, sin frameworks ni compilación; lo único externo son las fuentes de Google Fonts. Se abre con doble clic y guarda el avance de Daniel en localStorage con la clave `estudio-perceptron-v1`, que no se debe cambiar porque Daniel perdería su avance.

El JavaScript empieza con los datos y luego trae la lógica. Para reflejar cambios del proyecto casi siempre basta con editar los datos:

| Constante | Contenido | Formato |
|---|---|---|
| `MODULOS` | 7 módulos: neurona, aprendizaje, separabilidad, tresclases, regresion, nucleo, interfaz | `{ id, titulo, teoria (HTML), archivo, codigo, idea (HTML), demo }`; `codigo` es un fragmento real dentro de String.raw; `demo` es `null`, `"calculadora"`, `"lienzo"` o `"newton"` |
| `ARCHIVOS` | Descripción de cada .py para el mapa de inicio | `{ capa, texto, usa }` |
| `NIVELES` | 3 niveles de evaluación: conceptos; código y cálculos; casos en vivo | `tipo: "opcion"` con `opciones: [[texto, esCorrecta, explicación], ...]` (una sola correcta; cada opción explica por qué es correcta o incorrecta) o `tipo: "numero"` con `respuesta`, `tolerancia`, `explicacion` |
| `SUSTENTACION` | Preguntas del profesor con respuesta modelo | `{ tema, p, r }` |
| `ERRORES_COMUNES` | Errores al exponer | `[título, explicación]` |
| `ANALISIS` y sus `ANALISIS.push(...)` | Análisis línea por línea | `{ archivo, titulo, lineas: [[String.raw\`línea\`, "explicación"], ...], decision }` |
| `MEJORAS`, `BUENAS_PRACTICAS` | Cierre del análisis de código | Listas de textos |
| `REPASO`, `FRASES_CLAVE` | Tabla de fórmulas y frases clave | `[código, fórmula, para qué]` y textos |
| `FUNDAMENTOS` | 14 lecciones de la sección «Desde cero», para quien no sabe nada: problema, coordenadas, suma ponderada, rectas y planos, símbolos, estadística, azar, aprendizaje, series, Python I y II, tkinter y matplotlib, recorrido de un clic y ejecución | `{ id, titulo, html, ejemplos: [{ archivo, codigo, nota }], demo, preguntas }`; `demo` es `null`, `"recta"`, `"histograma"`, `"generador"` o `"newton"`; `preguntas` son de opción como en `NIVELES`, pero no cuentan para los niveles |
| `GLOSARIO` | Términos de la pestaña Glosario de «Desde cero» | `[término, definición]` |
| `FUNCIONES` | Catálogo de todas las funciones, clases y constantes de cada .py, que aparece en el análisis de código | `{ "archivo.py": [[nombre, qué hace], ...] }` |

El párrafo de resultados del repaso está escrito dentro de la función `vistaRepaso`. La lógica son las funciones:
- `crearLienzo`: el perceptrón 2D animado.
- `crearCalculadora`: F1 a F6 con la instancia 4 del ejemplo de clase.
- `crearNewton`: la raíz cuadrada paso a paso.
- `crearRecta`, `crearHistograma` y `crearGenerador` (con `crearCongruencial`): las demostraciones de «Desde cero»; `crearMiniQuiz`: las preguntas al final de cada lección.
- `vistaInicio`, `vistaFundamentos` (con `vistaGlosario`), `vistaEstudio`, `vistaEvaluacion`, `vistaSustentacion`, `vistaCodigo` y `vistaRepaso`: dibujan cada sección.

El avance de «Desde cero» se guarda en `estado.fundamentos`, dentro de la misma clave de localStorage.

Solo se tocan si hace falta un tipo de contenido nuevo.

### Procedimiento de actualización

1. Identificar qué cambió en el código: funciones, fórmulas, valores por defecto, interfaz y resultados.
2. Recalcular con la semilla 7 los resultados que pudieron cambiar, usando experimentos.py. Nunca se escribe un número que no haya salido de una ejecución real.
3. Buscar en todo el HTML cada dato o nombre que cambió y corregir todas sus apariciones. Un mismo resultado suele aparecer en el módulo, en las lecciones de «Desde cero», en preguntas de varios niveles, en la sustentación, en los errores comunes y en el repaso.
4. Copiar los fragmentos de código tal cual de los .py, con su sangría, y revisar las explicaciones de cada línea.
5. Para algo nuevo, agregar lo siguiente:
   - una mención en el módulo correspondiente y en `ARCHIVOS`;
   - al menos una pregunta en `NIVELES`;
   - una pregunta con respuesta modelo en `SUSTENTACION`;
   - su fila en `FUNCIONES` si es una función, clase o constante (el catálogo debe cubrir todo lo que hay en cada .py);
   - un bloque de `ANALISIS` si es una función importante;
   - su fila en `REPASO` si es una fórmula;
   - una explicación sencilla en la lección de `FUNDAMENTOS` que corresponda, y su término en `GLOSARIO`, si introduce un concepto o una palabra técnica nueva.
6. Validar y avisarle a Daniel qué secciones cambiaron, para que vuelva a estudiarlas.

### Reglas de contenido

- Cada afirmación debe poder verificarse en el código actual.
- La app debe reforzar que la matemática está en formulas.py y que tkinter y matplotlib solo dibujan.
- Se escribe en español, con coma decimal y párrafos fluidos.
- Las respuestas modelo van en primera persona, como Daniel hablándole al profesor.
- El nivel 3 sigue siendo de casos en vivo y cada nivel se desbloquea con 70 %.
- Dentro de String.raw no puede haber comillas invertidas ni `${`.
- En los textos entre comillas dobles se escapan las comillas internas o se usan «comillas angulares».

### Validación

```
node -e "const fs=require('fs');const h=fs.readFileSync('Estudio del perceptrón · Sustentación.html','utf8');new Function(h.split('<script>')[1].split('</script>')[0]);console.log('sintaxis ok')"
```

Luego se abre en el navegador con la consola (F12) abierta y se recorren las siete secciones: el perceptrón 2D entrena, las 14 lecciones de «Desde cero» con sus demostraciones y el glosario, los 7 módulos con la calculadora y el Newton, una pregunta de cada tipo, el simulacro y el acordeón, el filtro por archivo del análisis y la tabla del repaso. También se prueban el cambio de tema y una ventana angosta. No debe haber errores en la consola, salvo un posible aviso de fuentes sin internet.

## Qué se envía al profesor

- El informe en Word.
- Una carpeta comprimida con los seis archivos .py, LEEME.txt, requirements.txt y la carpeta .vscode.

No se envían la carpeta .venv (pesa mucho y solo sirve en el computador donde se creó), las carpetas __pycache__, la app de estudio («Estudio del perceptrón · Sustentación.html») ni este archivo CLAUDE.md.
