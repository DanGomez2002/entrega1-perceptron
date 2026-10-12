"""
main.py
=======
Interfaz gráfica del proyecto: Perceptrón para CLASIFICACIÓN (2 y 3 clases)
y para REGRESIÓN con datos simulados de 3 entradas.

Ejecutar:  python main.py      (en VS Code: F5)

La interfaz usa tkinter (incluido en Python) y muestra las gráficas de
matplotlib dentro de la ventana. Todos los cálculos del perceptrón están en
formulas.py y perceptron.py, sin librerías.

Autor: Carlos Daniel Gómez Murcia — Computacional I
"""

import tkinter as tk
from tkinter import ttk, messagebox

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import experimentos as ex
import graficas as gr
from aleatorio import DISTRIBUCIONES
from formulas import media, maximo, es_finito

AZUL = "#1F3A5F"
FONDO = "#F4F6FA"

# Escenarios para "reventar" el perceptrón: (centros, desviaciones, distribuciones, instancias)
ESCENARIOS = {
    "Separadas": ([[2, 2, 2], [8, 2, 5], [5, 8, 8]], [1.0, 1.0, 1.0],
                  ["normal", "normal", "normal"], [100, 100, 100]),
    "Cercanas": ([[2, 2, 2], [4.5, 2, 3], [3, 4.5, 4]], [1.2, 1.2, 1.2],
                 ["normal", "normal", "normal"], [100, 100, 100]),
    "Solapadas": ([[2, 2, 2], [3, 2, 2.5], [2.5, 3, 3]], [1.5, 1.5, 1.5],
                  ["normal", "exponencial", "normal"], [100, 100, 100]),
    # Desbalance pedido por el profesor: 100 contra 10
    "Desbalanceadas": ([[2, 2, 2], [4.5, 2, 3], [3, 4.5, 4]], [1.2, 1.2, 1.2],
                       ["normal", "normal", "normal"], [100, 10, 100]),
}


# ----------------------------------------------------------------------
# Lectura de los campos (acepta coma o punto decimal)
# ----------------------------------------------------------------------
def leer_float(var, nombre):
    """Lee un número; rechaza 'nan', 'inf' y números que no caben (M8)."""
    texto = var.get().strip().replace(",", ".")
    try:
        valor = float(texto)
    except ValueError:
        raise ValueError("'%s' debe ser un número (se escribió: '%s')." % (nombre, var.get()))
    if not es_finito(valor):
        raise ValueError("'%s' debe ser un número finito (se escribió: '%s')." % (nombre, var.get()))
    return valor


def leer_no_negativo(var, nombre):
    """Lee un número que no puede ser negativo (por ejemplo, una desviación σ)."""
    valor = leer_float(var, nombre)
    if valor < 0:
        raise ValueError("'%s' no puede ser negativa (se escribió: '%s')." % (nombre, var.get()))
    return valor


def leer_tasa(var):
    """La tasa γ debe ser mayor que cero: con γ = 0 los pesos no cambian y con
    γ < 0 cada ajuste empuja los pesos en la dirección contraria."""
    valor = leer_float(var, "Tasa de aprendizaje")
    if valor <= 0:
        raise ValueError("La tasa de aprendizaje γ debe ser mayor que cero.")
    return valor


def leer_int(var, nombre, minimo=None):
    texto = var.get().strip()
    try:
        real = float(texto.replace(",", "."))
    except ValueError:
        raise ValueError("'%s' debe ser un número entero (se escribió: '%s')." % (nombre, texto))
    if not es_finito(real):
        raise ValueError("'%s' debe ser un número entero finito (se escribió: '%s')." % (nombre, texto))
    valor = int(real)
    if minimo is not None and valor < minimo:
        raise ValueError("'%s' debe ser al menos %d." % (nombre, minimo))
    return valor


def leer_salida(var, nombre):
    """Las salidas del escalón pueden ser cualquier número (1/-1, 1/0, 5/2...).
    Si es entero se guarda como entero para que se lea mejor en los resultados."""
    v = leer_float(var, nombre)
    return int(v) if v == int(v) else v


def leer_semilla(var):
    texto = var.get().strip()
    if texto == "":
        return None
    try:
        return int(texto)
    except ValueError:
        raise ValueError("La semilla debe ser un entero o quedar vacía (aleatoria).")


def fmt(v):
    return ("%g" % v).replace(".", ",")


# ----------------------------------------------------------------------
class MarcoDesplazable(ttk.Frame):
    """Marco con barra de desplazamiento vertical (para pantallas pequeñas)."""

    def __init__(self, padre):
        super().__init__(padre)
        self.lienzo = tk.Canvas(self, highlightthickness=0, background=FONDO)
        barra = ttk.Scrollbar(self, orient="vertical", command=self.lienzo.yview)
        self.interior = ttk.Frame(self.lienzo, padding=8)
        self.interior.bind("<Configure>", lambda e: self.lienzo.configure(
            scrollregion=self.lienzo.bbox("all")))
        self._ventana = self.lienzo.create_window((0, 0), window=self.interior, anchor="nw")
        self.lienzo.bind("<Configure>", lambda e: self.lienzo.itemconfigure(
            self._ventana, width=e.width))
        self.lienzo.configure(yscrollcommand=barra.set)
        self.lienzo.pack(side="left", fill="both", expand=True)
        barra.pack(side="right", fill="y")
        self.interior.bind("<Enter>", lambda e: self.lienzo.bind_all("<MouseWheel>", self._rueda))
        self.interior.bind("<Leave>", lambda e: self.lienzo.unbind_all("<MouseWheel>"))

    def _rueda(self, evento):
        self.lienzo.yview_scroll(int(-evento.delta / 120), "units")


class PestanaGrafica:
    """Pestaña con una figura de matplotlib y su barra de herramientas
    (zoom, mover, guardar imagen). En 3D se puede rotar con el mouse."""

    def __init__(self, cuaderno, titulo):
        self.marco = ttk.Frame(cuaderno)
        cuaderno.add(self.marco, text=titulo)
        self.fig = Figure(figsize=(9, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.marco)
        self.barra = NavigationToolbar2Tk(self.canvas, self.marco, pack_toolbar=False)
        self.barra.update()
        self.barra.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    def vista_3d(self):
        for ax in self.fig.axes:
            if hasattr(ax, "elev"):
                return ax.elev, ax.azim
        return None


class PestanaTexto:
    def __init__(self, cuaderno, titulo):
        self.marco = ttk.Frame(cuaderno)
        cuaderno.add(self.marco, text=titulo)
        self.texto = tk.Text(self.marco, wrap="none", font=("Consolas", 10),
                             background="white", padx=8, pady=8)
        sy = ttk.Scrollbar(self.marco, orient="vertical", command=self.texto.yview)
        sx = ttk.Scrollbar(self.marco, orient="horizontal", command=self.texto.xview)
        self.texto.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        sy.pack(side="right", fill="y")
        sx.pack(side="bottom", fill="x")
        self.texto.pack(side="left", fill="both", expand=True)

    def mostrar(self, contenido):
        self.texto.configure(state="normal")
        self.texto.delete("1.0", "end")
        self.texto.insert("1.0", contenido)
        self.texto.configure(state="disabled")


# ----------------------------------------------------------------------
class Aplicacion:

    def __init__(self, raiz):
        self.raiz = raiz
        raiz.title("Perceptrón — Clasificación y Regresión | Computacional I")
        raiz.geometry("1500x900")
        raiz.minsize(1100, 700)
        raiz.configure(background=FONDO)
        self._estilos()

        self.res = None
        self.epoca = None
        self.animando = False
        self.pendientes = set()

        encabezado = tk.Frame(raiz, background=AZUL)
        encabezado.pack(side="top", fill="x")
        tk.Label(encabezado, text="Perceptrón: clasificación y regresión con datos 3D",
                 font=("Segoe UI", 15, "bold"), fg="white", bg=AZUL).pack(side="left", padx=14, pady=8)
        tk.Label(encabezado, text="Carlos Daniel Gómez Murcia — Computacional I — Entrega 1",
                 font=("Segoe UI", 10), fg="#D6E0F0", bg=AZUL).pack(side="right", padx=14)
        tk.Button(encabezado, text="⟲  Reiniciar todo", command=self.reiniciar,
                  font=("Segoe UI", 10, "bold"), fg="white", bg="#C0392B",
                  activebackground="#E74C3C", activeforeground="white", relief="flat",
                  cursor="hand2", padx=12, pady=3).pack(side="right", padx=6)
        raiz.bind("<Control-r>", lambda e: self.reiniciar())

        panel = ttk.PanedWindow(raiz, orient="horizontal")
        panel.pack(fill="both", expand=True, padx=6, pady=6)
        izquierda = ttk.Frame(panel, width=665)   # ancho inicial del panel de parámetros
        izquierda.pack_propagate(False)
        derecha = ttk.Frame(panel)
        panel.add(izquierda, weight=0)
        panel.add(derecha, weight=1)

        self.nb_config = ttk.Notebook(izquierda)
        self.nb_config.pack(fill="both", expand=True)
        self._pestana_clasificacion()
        self._pestana_regresion()
        self._pestana_ejemplo()

        self.nb_res = ttk.Notebook(derecha)
        self.nb_res.pack(fill="both", expand=True)
        self.g_espacio = PestanaGrafica(self.nb_res, "  Espacio 3D  ")
        self.g_epocas = PestanaGrafica(self.nb_res, "  Épocas  ")
        self.g_pesos = PestanaGrafica(self.nb_res, "  Pesos  ")
        self.t_proceso = PestanaTexto(self.nb_res, "  Proceso paso a paso  ")
        self.t_resumen = PestanaTexto(self.nb_res, "  Resumen y fórmulas  ")
        self.nb_res.bind("<<NotebookTabChanged>>", lambda e: self._redibujar_visible())
        self._barra_epocas(derecha)
        self.t_resumen.mostrar("Configure los datos a la izquierda y presione ENTRENAR.")

    # ------------------------------------------------------------------
    def _estilos(self):
        estilo = ttk.Style()
        estilo.theme_use("clam")
        estilo.configure(".", background=FONDO, font=("Segoe UI", 10))
        estilo.configure("TNotebook.Tab", padding=(10, 5))
        estilo.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"), foreground=AZUL)
        estilo.configure("Accion.TButton", font=("Segoe UI", 11, "bold"), foreground="white",
                         background=AZUL, padding=8)
        estilo.map("Accion.TButton", background=[("active", "#2E5A8C")])
        estilo.configure("Encabezado.TLabel", font=("Segoe UI", 9, "bold"))

    def _campo(self, marco, fila, texto, valor, ancho=10, columna=0):
        ttk.Label(marco, text=texto).grid(row=fila, column=columna, sticky="w", pady=2)
        var = tk.StringVar(value=valor)
        ttk.Entry(marco, textvariable=var, width=ancho).grid(row=fila, column=columna + 1,
                                                             sticky="w", padx=4, pady=2)
        return var

    def _lista(self, marco, fila, texto, opciones, valor, ancho=12, columna=0):
        ttk.Label(marco, text=texto).grid(row=fila, column=columna, sticky="w", pady=2)
        var = tk.StringVar(value=valor)
        ttk.Combobox(marco, textvariable=var, values=opciones, state="readonly",
                     width=ancho).grid(row=fila, column=columna + 1, sticky="w", padx=4, pady=2)
        return var

    # ------------------------------------------------------------------
    # Pestaña CLASIFICACIÓN
    # ------------------------------------------------------------------
    def _pestana_clasificacion(self):
        cfg = ex.config_clasificacion()
        m = MarcoDesplazable(self.nb_config)
        self.nb_config.add(m, text=" Clasificación ")
        f = m.interior

        datos = ttk.LabelFrame(f, text="Datos simulados (3 entradas por instancia)", padding=8)
        datos.pack(fill="x", pady=4)
        fila_n = ttk.Frame(datos)
        fila_n.grid(row=0, column=0, columnspan=7, sticky="w", pady=(0, 6))
        ttk.Label(fila_n, text="Número de clases:").pack(side="left")
        self.c_nclases = tk.IntVar(value=cfg["n_clases"])
        for n in (2, 3):
            ttk.Radiobutton(fila_n, text=str(n), value=n, variable=self.c_nclases,
                            command=self._activar_clase3).pack(side="left", padx=6)

        for col, txt in enumerate(["Clase", "Instancias", "Media x1", "Media x2", "Media x3",
                                   "Desv.", "Distribución"]):
            ttk.Label(datos, text=txt, style="Encabezado.TLabel").grid(row=1, column=col, padx=2)
        self.c_clases = []
        self.fila3 = []
        for i, clase in enumerate(cfg["clases"]):
            color = gr.COLORES_CLASE[i]
            tk.Label(datos, text=" %d " % (i + 1), bg=color, fg="white",
                     font=("Segoe UI", 9, "bold")).grid(row=i + 2, column=0, pady=2)
            fila = {"n": tk.StringVar(value=str(clase["n"])),
                    "centro": [tk.StringVar(value=fmt(v)) for v in clase["centro"]],
                    "desviacion": tk.StringVar(value=fmt(clase["desviacion"])),
                    "distribucion": tk.StringVar(value=clase["distribucion"])}
            widgets = [ttk.Entry(datos, textvariable=fila["n"], width=7)]
            widgets += [ttk.Entry(datos, textvariable=v, width=6) for v in fila["centro"]]
            widgets.append(ttk.Entry(datos, textvariable=fila["desviacion"], width=5))
            widgets.append(ttk.Combobox(datos, textvariable=fila["distribucion"],
                                        values=DISTRIBUCIONES, state="readonly", width=11))
            for col, wdg in enumerate(widgets):
                wdg.grid(row=i + 2, column=col + 1, padx=2, pady=2)
            self.c_clases.append(fila)
            if i == 2:
                self.fila3 = widgets

        esc = ttk.Frame(datos)
        esc.grid(row=6, column=0, columnspan=7, sticky="w", pady=(8, 0))
        ttk.Label(esc, text="Escenarios:").pack(side="left")
        for nombre in ESCENARIOS:
            ttk.Button(esc, text=nombre,
                       command=lambda n=nombre: self._escenario(n)).pack(side="left", padx=2)
        rev = ttk.Frame(datos)
        rev.grid(row=7, column=0, columnspan=7, sticky="w", pady=(6, 0))
        ttk.Label(rev, text="Reventar:").pack(side="left")
        for texto, comando in (("Acercar clases", lambda: self._mover_medias(0.75)),
                               ("Alejar clases", lambda: self._mover_medias(1.33)),
                               ("Aumentar σ", lambda: self._escalar_sigma(1.3)),
                               ("Reducir σ", lambda: self._escalar_sigma(1 / 1.3))):
            ttk.Button(rev, text=texto, command=comando).pack(side="left", padx=2)

        per = ttk.LabelFrame(f, text="Perceptrón", padding=8)
        per.pack(fill="x", pady=4)
        self.c_tasa = self._campo(per, 0, "Tasa de aprendizaje γ", fmt(cfg["tasa"]))
        self.c_epocas = self._campo(per, 1, "Número máximo de épocas", str(cfg["max_epocas"]))
        self.c_eps = self._campo(per, 2, "Error mínimo ε", fmt(cfg["error_minimo"]))
        self.c_umbral = self._campo(per, 3, "Umbral del escalón", fmt(cfg["umbral"]))
        self.c_alta = self._campo(per, 4, "Salida si S ≥ umbral", str(cfg["salida_alta"]))
        self.c_baja = self._campo(per, 5, "Salida si S < umbral", str(cfg["salida_baja"]))
        self.c_x0 = self._lista(per, 6, "Entrada del sesgo x0", ("-1", "+1"), "-1", 6)
        self.c_estrategia = self._lista(per, 7, "Estrategia (3 clases)",
                                        ("cascada", "uno_vs_resto"), cfg["estrategia"])
        self.c_semilla = self._campo(per, 8, "Semilla (vacía = aleatoria)", "")
        self.c_barajar = tk.BooleanVar(value=True)
        self.c_registrar = tk.BooleanVar(value=False)
        ttk.Checkbutton(per, text="Barajar instancias en cada época",
                        variable=self.c_barajar).grid(row=9, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(per, text="Registrar proceso paso a paso",
                        variable=self.c_registrar).grid(row=10, column=0, columnspan=2, sticky="w")
        ttk.Label(per, text="S = w1·x1 + w2·x2 + w3·x3 + θ·x0   (x0 = −1  →  S = Σwx − θ)",
                  foreground="#555").grid(row=11, column=0, columnspan=2, sticky="w", pady=(6, 0))

        ttk.Button(f, text="▶  ENTRENAR CLASIFICACIÓN", style="Accion.TButton",
                   command=self.entrenar_clasificacion).pack(fill="x", pady=10)

        prueba = ttk.LabelFrame(f, text="Clasificar un punto nuevo (después de entrenar)", padding=8)
        prueba.pack(fill="x", pady=4)
        self.c_punto = []
        for j in range(3):
            ttk.Label(prueba, text="x%d" % (j + 1)).grid(row=0, column=2 * j, padx=(0, 2))
            var = tk.StringVar(value="5")
            ttk.Entry(prueba, textvariable=var, width=7).grid(row=0, column=2 * j + 1, padx=(0, 8))
            self.c_punto.append(var)
        ttk.Button(prueba, text="Clasificar", command=self.clasificar_punto).grid(row=0, column=6)
        self._activar_clase3()

    def _activar_clase3(self):
        estado = "normal" if self.c_nclases.get() == 3 else "disabled"
        for wdg in self.fila3:
            if isinstance(wdg, ttk.Combobox):
                wdg.configure(state="readonly" if estado == "normal" else "disabled")
            else:
                wdg.configure(state=estado)

    def _escenario(self, nombre):
        centros, desv, distrib, instancias = ESCENARIOS[nombre]
        for fila, c, s, d, n in zip(self.c_clases, centros, desv, distrib, instancias):
            for var, v in zip(fila["centro"], c):
                var.set(fmt(v))
            fila["desviacion"].set(fmt(s))
            fila["distribucion"].set(d)
            fila["n"].set(str(n))

    def _mover_medias(self, factor):
        """Acerca (factor < 1) o aleja (factor > 1) cada centro respecto al centro
        común de las clases activas:  c_nuevo = centro_comun + factor * (c - centro_comun)."""
        try:
            filas = self.c_clases[:self.c_nclases.get()]
            centros = [[leer_float(v, "Media") for v in fila["centro"]] for fila in filas]
        except ValueError as error:
            messagebox.showerror("Dato no válido", str(error))
            return
        comun = [media([c[j] for c in centros]) for j in range(3)]           # F16
        for fila, c in zip(filas, centros):
            for j, var in enumerate(fila["centro"]):
                var.set("%.3g" % (comun[j] + factor * (c[j] - comun[j])))
        self.entrenar_clasificacion()

    def _escalar_sigma(self, factor):
        """Multiplica la desviación de cada clase activa por el factor."""
        try:
            for fila in self.c_clases[:self.c_nclases.get()]:
                s = leer_float(fila["desviacion"], "Desviación") * factor
                fila["desviacion"].set("%.3g" % s)
        except ValueError as error:
            messagebox.showerror("Dato no válido", str(error))
            return
        self.entrenar_clasificacion()

    def clasificar_punto(self):
        if self.res is None or self.res["tipo"] != "clasificacion":
            messagebox.showinfo("Primero entrene", "Entrene una clasificación antes de probar un punto.")
            return
        try:
            x = [leer_float(v, "x%d" % (j + 1)) for j, v in enumerate(self.c_punto)]
        except ValueError as error:
            messagebox.showerror("Dato no válido", str(error))
            return
        clase = ex.clasificar_punto(self.res, x)
        self.res["punto"] = (x, clase)
        self.pendientes.add("espacio")
        self.nb_res.select(0)
        self._redibujar_visible()
        self.estado.configure(text="El punto (%s) se clasifica como %s (estrella en el espacio 3D)."
                              % (", ".join(fmt(v) for v in x), self.res["nombres"][clase]))

    def _config_clasificacion(self):
        cfg = ex.config_clasificacion()
        cfg["n_clases"] = self.c_nclases.get()
        for i in range(cfg["n_clases"]):
            fila, clase = self.c_clases[i], cfg["clases"][i]
            etiqueta = "clase %d" % (i + 1)
            clase["n"] = leer_int(fila["n"], "Instancias de la " + etiqueta, 1)
            clase["centro"] = [leer_float(v, "Media x%d de la %s" % (j + 1, etiqueta))
                               for j, v in enumerate(fila["centro"])]
            clase["desviacion"] = leer_no_negativo(fila["desviacion"], "Desviación de la " + etiqueta)
            clase["distribucion"] = fila["distribucion"].get()
        cfg["tasa"] = leer_tasa(self.c_tasa)
        cfg["max_epocas"] = leer_int(self.c_epocas, "Número máximo de épocas", 1)
        cfg["error_minimo"] = leer_float(self.c_eps, "Error mínimo")
        cfg["umbral"] = leer_float(self.c_umbral, "Umbral")
        cfg["salida_alta"] = leer_salida(self.c_alta, "Salida alta")
        cfg["salida_baja"] = leer_salida(self.c_baja, "Salida baja")
        if cfg["salida_alta"] == cfg["salida_baja"]:
            raise ValueError("Las dos salidas del escalón deben ser diferentes.")
        # La regla wi = wi + gamma*(d - y)*xi supone que la salida alta es la mayor:
        # si estuvieran al revés, cada ajuste movería el plano en la dirección contraria.
        if cfg["salida_alta"] < cfg["salida_baja"]:
            raise ValueError("La salida si S ≥ umbral debe ser MAYOR que la salida si S < umbral\n"
                             "(por ejemplo 1/-1 o 1/0). Con las salidas al revés, la regla de\n"
                             "ajuste empuja los pesos en la dirección contraria y no aprende.")
        cfg["x0"] = int(self.c_x0.get())
        cfg["estrategia"] = self.c_estrategia.get()
        cfg["semilla"] = leer_semilla(self.c_semilla)
        cfg["barajar"] = self.c_barajar.get()
        cfg["registrar"] = self.c_registrar.get()
        return cfg

    # ------------------------------------------------------------------
    # Pestaña REGRESIÓN
    # ------------------------------------------------------------------
    def _pestana_regresion(self):
        cfg = ex.config_regresion()
        m = MarcoDesplazable(self.nb_config)
        self.nb_config.add(m, text=" Regresión ")
        f = m.interior

        datos = ttk.LabelFrame(f, text="Datos simulados", padding=8)
        datos.pack(fill="x", pady=4)
        fila_n = ttk.Frame(datos)
        fila_n.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 6))
        ttk.Label(fila_n, text="Número de entradas:").pack(side="left")
        self.r_nentradas = tk.IntVar(value=cfg["n_entradas"])
        for n in (2, 3):
            ttk.Radiobutton(fila_n, text=str(n), value=n, variable=self.r_nentradas,
                            command=self._activar_x3).pack(side="left", padx=6)
        self.r_n = self._campo(datos, 1, "Instancias", str(cfg["n"]), 8)

        ttk.Label(datos, text="", style="Encabezado.TLabel").grid(row=2, column=0)
        for j, txt in enumerate(["x1", "x2", "x3"]):
            ttk.Label(datos, text=txt, style="Encabezado.TLabel").grid(row=2, column=j + 1)
        ttk.Label(datos, text="Media de las entradas").grid(row=3, column=0, sticky="w")
        ttk.Label(datos, text="Coeficientes reales a").grid(row=4, column=0, sticky="w")
        self.r_centro = [tk.StringVar(value=fmt(v)) for v in cfg["centro_x"]]
        self.r_coef = [tk.StringVar(value=fmt(v)) for v in cfg["coeficientes"]]
        self.col_x3 = []
        for j in range(3):
            e1 = ttk.Entry(datos, textvariable=self.r_centro[j], width=7)
            e2 = ttk.Entry(datos, textvariable=self.r_coef[j], width=7)
            e1.grid(row=3, column=j + 1, padx=2, pady=2)
            e2.grid(row=4, column=j + 1, padx=2, pady=2)
            if j == 2:
                self.col_x3 = [e1, e2]
        self.r_desv = self._campo(datos, 5, "Desviación de las entradas", fmt(cfg["desviacion_x"]), 8)
        self.r_dist = self._lista(datos, 6, "Distribución de las entradas", DISTRIBUCIONES,
                                  cfg["distribucion_x"])
        self.r_b = self._campo(datos, 7, "Término independiente b", fmt(cfg["sesgo"]), 8)
        self.r_modelo = self._lista(datos, 8, "Modelo real", ("lineal", "cuadratico"), cfg["modelo"])
        self.r_ruido = self._campo(datos, 9, "Desviación del ruido", fmt(cfg["desviacion_ruido"]), 8)
        self.r_dist_ruido = self._lista(datos, 10, "Distribución del ruido", DISTRIBUCIONES,
                                        cfg["distribucion_ruido"])
        self.r_formula = ttk.Label(datos, foreground="#555")
        self.r_formula.grid(row=11, column=0, columnspan=4, sticky="w", pady=(6, 0))

        per = ttk.LabelFrame(f, text="Perceptrón (activación lineal)", padding=8)
        per.pack(fill="x", pady=4)
        self.r_tasa = self._campo(per, 0, "Tasa de aprendizaje γ", fmt(cfg["tasa"]))
        self.r_epocas = self._campo(per, 1, "Número máximo de épocas", str(cfg["max_epocas"]))
        self.r_tol = self._campo(per, 2, "Mejora mínima en 20 épocas", fmt(cfg["tolerancia_cambio"]))
        self.r_x0 = self._lista(per, 3, "Entrada del sesgo x0", ("-1", "+1"), "-1", 6)
        self.r_semilla = self._campo(per, 4, "Semilla (vacía = aleatoria)", "")
        self.r_barajar = tk.BooleanVar(value=True)
        self.r_registrar = tk.BooleanVar(value=False)
        ttk.Checkbutton(per, text="Barajar instancias en cada época",
                        variable=self.r_barajar).grid(row=5, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(per, text="Registrar proceso paso a paso",
                        variable=self.r_registrar).grid(row=6, column=0, columnspan=2, sticky="w")

        ttk.Button(f, text="▶  ENTRENAR REGRESIÓN", style="Accion.TButton",
                   command=self.entrenar_regresion).pack(fill="x", pady=10)
        self.r_modelo.trace_add("write", lambda *a: self._activar_x3())
        self._activar_x3()

    def _activar_x3(self):
        tres = self.r_nentradas.get() == 3
        for wdg in self.col_x3:
            wdg.configure(state="normal" if tres else "disabled")
        potencia = "²" if self.r_modelo.get() == "cuadratico" else ""
        terminos = ["a%d·x%d%s" % (j, j, potencia) for j in range(1, 4 if tres else 3)]
        self.r_formula.configure(text="Modelo real:  y = " + " + ".join(terminos) + " + b + ruido")

    def _config_regresion(self):
        cfg = ex.config_regresion()
        ne = self.r_nentradas.get()
        cfg["n_entradas"] = ne
        cfg["n"] = leer_int(self.r_n, "Instancias", 2)
        cfg["centro_x"] = [leer_float(self.r_centro[j], "Media de x%d" % (j + 1)) for j in range(ne)]
        cfg["coeficientes"] = [leer_float(self.r_coef[j], "Coeficiente a%d" % (j + 1))
                               for j in range(ne)]
        cfg["desviacion_x"] = leer_no_negativo(self.r_desv, "Desviación de las entradas")
        cfg["distribucion_x"] = self.r_dist.get()
        cfg["sesgo"] = leer_float(self.r_b, "Término independiente b")
        cfg["modelo"] = self.r_modelo.get()
        cfg["desviacion_ruido"] = leer_no_negativo(self.r_ruido, "Desviación del ruido")
        cfg["distribucion_ruido"] = self.r_dist_ruido.get()
        cfg["tasa"] = leer_tasa(self.r_tasa)
        cfg["max_epocas"] = leer_int(self.r_epocas, "Número máximo de épocas", 1)
        cfg["tolerancia_cambio"] = leer_float(self.r_tol, "Mejora mínima")
        cfg["x0"] = int(self.r_x0.get())
        cfg["semilla"] = leer_semilla(self.r_semilla)
        cfg["barajar"] = self.r_barajar.get()
        cfg["registrar"] = self.r_registrar.get()
        return cfg

    # ------------------------------------------------------------------
    # Pestaña EJEMPLO DE CLASE
    # ------------------------------------------------------------------
    def _pestana_ejemplo(self):
        f = ttk.Frame(self.nb_config, padding=10)
        self.nb_config.add(f, text=" Ejemplo de clase ")
        ttk.Label(f, text="Reproduce el ejercicio de las diapositivas para\n"
                          "comprobar que el código hace las mismas cuentas.",
                  justify="left").pack(anchor="w", pady=(0, 8))
        tabla = ttk.Treeview(f, columns=("x1", "x2", "clase", "d"), show="headings", height=6)
        for col, txt in zip(("x1", "x2", "clase", "d"), ("x1", "x2", "Clase", "Salida d")):
            tabla.heading(col, text=txt)
            tabla.column(col, width=80, anchor="center")
        for i, (x, d) in enumerate(zip(ex.EJEMPLO_X, ex.EJEMPLO_D)):
            tabla.insert("", "end", values=(fmt(x[0]), fmt(x[1]), 0 if d == 1 else 1, d))
        tabla.pack(anchor="w")
        ttk.Label(f, justify="left", text=(
            "\nPesos iniciales:  w1 = −2,10137   w2 = 3,11960   θ = 0,191448\n"
            "Tasa de aprendizaje γ = 0,25     x0 = −1\n\n"
            "S = w1·x1 + w2·x2 − θ\n"
            "wi = wi + γ (d − y) xi\n"
            "θ  = θ + γ (d − y) x0\n\n"
            "Resultado esperado (diapositivas):\n"
            "   S4 = 17,31523  → error (y = 1, d = −1)\n"
            "   S5 = −23,91494 (con los pesos ya ajustados)")).pack(anchor="w")
        ttk.Button(f, text="▶  REPRODUCIR EJEMPLO", style="Accion.TButton",
                   command=self.reproducir_ejemplo).pack(fill="x", pady=12)

    # ------------------------------------------------------------------
    # Barra de épocas (animación)
    # ------------------------------------------------------------------
    def _barra_epocas(self, padre):
        self.estado = ttk.Label(padre, text="", foreground=AZUL, font=("Segoe UI", 10, "bold"),
                                padding=(6, 0, 6, 4))
        self.estado.pack(side="bottom", fill="x")
        barra = ttk.Frame(padre, padding=(4, 6))
        barra.pack(side="bottom", fill="x")
        ttk.Label(barra, text="Época mostrada:").pack(side="left")
        self.escala = tk.Scale(barra, from_=0, to=1, orient="horizontal", resolution=1,
                               showvalue=True, length=320, background=FONDO,
                               highlightthickness=0, command=self._mover_epoca)
        self.escala.pack(side="left", padx=6)
        ttk.Button(barra, text="▶ Animar entrenamiento", command=self.animar).pack(side="left", padx=3)
        ttk.Button(barra, text="■ Detener", command=self.detener).pack(side="left", padx=3)
        ttk.Button(barra, text="Resultado final", command=self.mostrar_final).pack(side="left", padx=3)

    def _max_epocas(self):
        return maximo([p.epocas for p in self.res["perceptrones"]])

    def _mover_epoca(self, valor):
        if self.res is None:
            return
        nueva = int(float(valor))
        if nueva >= self._max_epocas() and not self.animando:
            nueva = None               # última época = resultado final
        if nueva == self.epoca:
            return                     # ya está dibujada
        self.epoca = nueva
        self.pendientes = {"espacio", "epocas", "pesos"}
        self._redibujar_visible()

    def animar(self):
        if self.res is None:
            return
        self.animando = True
        self.escala.set(0)
        self._paso_animacion(0)

    def _paso_animacion(self, k):
        if not self.animando or self.res is None:
            return
        self.escala.set(k)
        self._mover_epoca(k)
        if k < self._max_epocas():
            espera = maximo([80, int(6000 / maximo([self._max_epocas(), 1]))])   # ~6 s en total
            self.raiz.after(espera, lambda: self._paso_animacion(k + 1))
        else:
            self.animando = False

    def detener(self):
        self.animando = False

    def mostrar_final(self):
        self.animando = False
        if self.res is None:
            return
        self.escala.set(self._max_epocas())
        self.epoca = None
        self.pendientes = {"espacio", "epocas", "pesos"}
        self._redibujar_visible()

    # ------------------------------------------------------------------
    # Reiniciar: valores por defecto y resultados en blanco
    # ------------------------------------------------------------------
    def reiniciar(self, preguntar=True):
        if preguntar and not messagebox.askyesno(
                "Reiniciar todo",
                "Se borrarán los resultados y todos los parámetros volverán a sus "
                "valores iniciales.\n\n¿Desea continuar?"):
            return
        self.animando = False

        # --- Clasificación ---
        cfg = ex.config_clasificacion()
        self.c_nclases.set(cfg["n_clases"])
        for fila, clase in zip(self.c_clases, cfg["clases"]):
            fila["n"].set(str(clase["n"]))
            for var, v in zip(fila["centro"], clase["centro"]):
                var.set(fmt(v))
            fila["desviacion"].set(fmt(clase["desviacion"]))
            fila["distribucion"].set(clase["distribucion"])
        self.c_tasa.set(fmt(cfg["tasa"]))
        self.c_epocas.set(str(cfg["max_epocas"]))
        self.c_eps.set(fmt(cfg["error_minimo"]))
        self.c_umbral.set(fmt(cfg["umbral"]))
        self.c_alta.set(str(cfg["salida_alta"]))
        self.c_baja.set(str(cfg["salida_baja"]))
        self.c_x0.set("-1")
        self.c_estrategia.set(cfg["estrategia"])
        self.c_semilla.set("")
        self.c_barajar.set(cfg["barajar"])
        self.c_registrar.set(cfg["registrar"])
        self._activar_clase3()

        # --- Regresión ---
        cfg = ex.config_regresion()
        self.r_nentradas.set(cfg["n_entradas"])
        self.r_n.set(str(cfg["n"]))
        for var, v in zip(self.r_centro, cfg["centro_x"]):
            var.set(fmt(v))
        for var, v in zip(self.r_coef, cfg["coeficientes"]):
            var.set(fmt(v))
        self.r_desv.set(fmt(cfg["desviacion_x"]))
        self.r_dist.set(cfg["distribucion_x"])
        self.r_b.set(fmt(cfg["sesgo"]))
        self.r_modelo.set(cfg["modelo"])
        self.r_ruido.set(fmt(cfg["desviacion_ruido"]))
        self.r_dist_ruido.set(cfg["distribucion_ruido"])
        self.r_tasa.set(fmt(cfg["tasa"]))
        self.r_epocas.set(str(cfg["max_epocas"]))
        self.r_tol.set(fmt(cfg["tolerancia_cambio"]))
        self.r_x0.set("-1")
        self.r_semilla.set("")
        self.r_barajar.set(cfg["barajar"])
        self.r_registrar.set(cfg["registrar"])
        self._activar_x3()

        # --- Resultados en blanco ---
        self.res = None
        self.epoca = None
        self.pendientes = set()
        for pestana in (self.g_espacio, self.g_epocas, self.g_pesos):
            pestana.fig.clear()
            pestana.barra.set_message("")
            pestana.canvas.draw_idle()
        self.t_proceso.mostrar("")
        self.t_resumen.mostrar("Configure los datos a la izquierda y presione ENTRENAR.")
        self.escala.configure(to=1)
        self.escala.set(0)
        self.estado.configure(text="Todo reiniciado: valores por defecto y sin resultados.")
        self.nb_config.select(0)
        self.nb_res.select(0)

    # ------------------------------------------------------------------
    # Entrenamiento y dibujo
    # ------------------------------------------------------------------
    def _ejecutar(self, funcion):
        self.animando = False
        self.raiz.configure(cursor="watch")
        self.estado.configure(text="Entrenando...")
        self.raiz.update()
        try:
            self.res = funcion()
        except ValueError as error:
            messagebox.showerror("Dato no válido", str(error))
            return
        finally:
            self.raiz.configure(cursor="")
        self.epoca = None
        self.escala.configure(to=maximo([self._max_epocas(), 1]))
        self.escala.set(self._max_epocas())
        self.epoca = None
        self.pendientes = {"espacio", "epocas", "pesos"}
        self._redibujar_visible()
        registro = self.res.get("registro")
        if registro:
            self.t_proceso.mostrar("\n".join(registro))
        else:
            self.t_proceso.mostrar("Active 'Registrar proceso paso a paso' antes de entrenar para\n"
                                   "ver S, y, d, e y el ajuste de los pesos en cada instancia.\n"
                                   "(Recomendado con pocas instancias, por ejemplo 10 por clase.)")
        self.t_resumen.mostrar(ex.resumen(self.res))
        partes = [p.estado() for p in self.res["perceptrones"]]
        extra = ""
        if self.res["tipo"] == "clasificacion":
            por_clase = ", ".join("%s %.0f %%" % (n.replace("Clase ", "C"), e["exactitud"])
                                  for n, e in zip(self.res["nombres"], self.res["estadisticas"]))
            extra = "  |  exactitud %.1f %% (%s)" % (self.res["exactitud"], por_clase)
        elif self.res["tipo"] == "regresion":
            extra = "  |  ECM %s  |  R² %s" % (ex.num(self.res["ecm"]), ex.num(self.res["r2"]))
        self.estado.configure(text="; ".join(partes) + extra)

    def entrenar_clasificacion(self):
        self._ejecutar(lambda: ex.ejecutar_clasificacion(self._config_clasificacion()))

    def entrenar_regresion(self):
        self._ejecutar(lambda: ex.ejecutar_regresion(self._config_regresion()))

    def reproducir_ejemplo(self):
        self._ejecutar(ex.ejecutar_ejemplo_clase)
        self.nb_res.select(self.t_proceso.marco)

    def _redibujar_visible(self):
        if self.res is None:
            return
        actual = self.nb_res.index(self.nb_res.select())
        claves = {0: "espacio", 1: "epocas", 2: "pesos"}
        if actual not in claves or claves[actual] not in self.pendientes:
            return
        clave = claves[actual]
        self.pendientes.discard(clave)
        if clave == "espacio":
            pestana = self.g_espacio
            vista = pestana.vista_3d()
            dibujo = {"clasificacion": gr.dibujar_clasificacion, "regresion": gr.dibujar_regresion,
                      "ejemplo": gr.dibujar_ejemplo}[self.res["tipo"]]
            dibujo(pestana.fig, self.res, self.epoca, vista)
        elif clave == "epocas":
            pestana = self.g_epocas
            gr.dibujar_epocas(pestana.fig, self.res, self.epoca)
        else:
            pestana = self.g_pesos
            gr.dibujar_pesos(pestana.fig, self.res, self.epoca)
        if not (clave == "espacio" and self.res["tipo"] == "regresion"):
            pestana.fig.tight_layout()
        pestana.canvas.draw_idle()


def main():
    try:  # pantallas de alta resolución en Windows: texto nítido
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    raiz = tk.Tk()
    Aplicacion(raiz)
    raiz.mainloop()


if __name__ == "__main__":
    main()
