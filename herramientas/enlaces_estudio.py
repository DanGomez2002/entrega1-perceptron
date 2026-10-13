"""
enlaces_estudio.py
==================
Herramienta de apoyo para la app de estudio (no forma parte del simulador ni
se envía al profesor).

Lee los seis .py del proyecto y escribe dentro de «Estudio del perceptrón ·
Sustentación.html» el bloque de datos que usan los enlaces a VS Code:

    ENLACES_DEF     línea donde se define cada función, clase, método y
                    constante de cada archivo.
    ENLACES_LINEAS  para cada línea de código que la app muestra en sus
                    fragmentos, el archivo y la línea donde aparece en los .py.

Hay que ejecutarla cada vez que cambie el código, para que los enlaces sigan
apuntando a la línea correcta:

    .venv\\Scripts\\python herramientas\\enlaces_estudio.py
"""

import ast
import json
import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
HTML = RAIZ / "Estudio del perceptrón · Sustentación.html"
ARCHIVOS = ["formulas.py", "aleatorio.py", "perceptron.py", "experimentos.py",
            "graficas.py", "main.py"]
INICIO = "/* ENLACES-INICIO"
FIN = "/* ENLACES-FIN */"


def definiciones(texto):
    """Nombre -> línea de cada def, class y constante (los métodos como Clase.metodo)."""
    resultado = {}

    def recorrer(nodos, prefijo):
        for nodo in nodos:
            if isinstance(nodo, (ast.FunctionDef, ast.ClassDef)):
                resultado.setdefault(prefijo + nodo.name, nodo.lineno)
                if isinstance(nodo, ast.ClassDef):
                    recorrer(nodo.body, prefijo + nodo.name + ".")
            elif isinstance(nodo, ast.Assign):
                for destino in nodo.targets:
                    if isinstance(destino, ast.Name):
                        resultado.setdefault(prefijo + destino.id, nodo.lineno)

    recorrer(ast.parse(texto).body, "")
    return resultado


def main():
    html = HTML.read_text(encoding="utf-8")
    fuentes = {a: (RAIZ / a).read_text(encoding="utf-8") for a in ARCHIVOS}

    enlaces_def = {a: definiciones(t) for a, t in fuentes.items()}

    # Todas las líneas de los fragmentos String.raw`...` que muestra la app
    sin_bloque = re.sub(re.escape(INICIO) + r".*?" + re.escape(FIN), "", html, flags=re.S)
    buscadas = set()
    for fragmento in re.findall(r"String\.raw`([^`]*)`", sin_bloque):
        for linea in fragmento.split("\n"):
            if linea.strip():
                buscadas.add(linea.strip())
    indice = {}
    for archivo, texto in fuentes.items():
        for numero, linea in enumerate(texto.split("\n"), start=1):
            clave = linea.strip()
            if clave in buscadas:
                indice.setdefault(clave, []).append([archivo, numero])
    faltan = sorted(b for b in buscadas if b not in indice)

    bloque = (INICIO + " (generado por herramientas/enlaces_estudio.py; no editar a mano) */\n"
              "const RAIZ_PROYECTO = " + json.dumps(RAIZ.as_posix()) + ";\n"
              "const ENLACES_DEF = " + json.dumps(enlaces_def, ensure_ascii=False) + ";\n"
              "const ENLACES_LINEAS = " + json.dumps(indice, ensure_ascii=False) + ";\n"
              + FIN)
    if INICIO in html:
        html = re.sub(re.escape(INICIO) + r".*?" + re.escape(FIN), lambda m: bloque, html, flags=re.S)
    else:
        html = html.replace("<script>\n", "<script>\n" + bloque + "\n", 1)
    HTML.write_text(html, encoding="utf-8")

    print("Definiciones:", sum(len(d) for d in enlaces_def.values()))
    print("Líneas de fragmentos encontradas:", len(indice), "de", len(buscadas))
    for linea in faltan:
        print("  NO ENCONTRADA:", linea)


if __name__ == "__main__":
    main()
