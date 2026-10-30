"""Catálogo de formas y sus parámetros: la interfaz se arma a partir de esto.

Cada parámetro pertenece a un grupo (se muestran como secciones) y cada forma
trae perfiles de calidad que cambian varios parámetros de un clic. Los valores
por defecto son los del perfil «Máxima».
"""

from __future__ import annotations


def num(key, label, default, lo, hi, step=0.1, unit="mm", hint="", group=""):
    return {"clave": key, "nombre": label, "tipo": "numero", "defecto": default, "min": lo, "max": hi, "paso": step, "unidad": unit,
            "ayuda": hint, "grupo": group}


def choice(key, label, default, options, hint="", group=""):
    return {"clave": key, "nombre": label, "tipo": "opcion", "defecto": default, "opciones": options, "ayuda": hint, "grupo": group}


def flag(key, label, default, hint="", group=""):
    return {"clave": key, "nombre": label, "tipo": "si_no", "defecto": default, "ayuda": hint, "grupo": group}


# clave → (título, plegado al abrir)
GRUPOS = {
    "calidad": ("Calidad", False),
    "foto": ("Foto para la IA", False),
    "muestreo": ("Muestreo avanzado", True),
    "forma": ("Forma", False),
    "relieve": ("Relieve", False),
    "detalle": ("Detalle", False),
    "tonos": ("Tonos", False),
    "colores": ("Colores", False),
    "profundidad": ("Profundidad y resolución", True),
    "resolucion": ("Resolución", True),
    "tamano": ("Tamaño y base", False),
    "superficie": ("Superficie", False),
    "malla": ("Malla", True),
}

CONTORNOS = [
    ["silueta", "Silueta"], ["rectangulo", "Rectángulo"], ["circulo", "Círculo"], ["hexagono", "Hexágono"],
    ["escudo", "Escudo"], ["estrella", "Estrella"], ["corazon", "Corazón"],
]
CARAS = [[250_000, "250 mil"], [500_000, "500 mil"], [1_000_000, "1 millón"], [2_000_000, "2 millones"], [0, "Sin reducir"]]
CARAS_AYUDA = "Tope de triángulos del STL. Menos pesa menos; «Sin reducir» guarda todo el detalle (archivos grandes)."
DETALLE_MAX = [[0.7, "0,7 MP"], [1.5, "1,5 MP"], [3.0, "3 MP (lento)"]]
DETALLE_MAX_AYUDA = "Tope de puntos de la malla. Si la pieza es grande, la resolución se ajusta sola para no pasarse."

RELIEF_COMMON = [
    choice("fuente", "Relieve desde", "profundidad", [["profundidad", "Profundidad (IA)"], ["luminancia", "Luz y sombra"]],
           "Profundidad usa Depth Anything V2; luz y sombra sirve para dibujos planos.", "relieve"),
    flag("invertir", "Invertir relieve", False, group="relieve"),
    choice("curva", "Curva", "linear", [["linear", "Lineal"], ["gamma", "Gamma"], ["s-curve", "Curva S"]],
           "Cómo se reparte la altura: la curva S marca más los volúmenes del medio.", "relieve"),
    num("gamma", "Gamma", 1.0, 0.2, 3.0, 0.1, "", "Solo con la curva Gamma: menos de 1 levanta lo bajo, más de 1 lo hunde.", "relieve"),
    num("detalle", "Detalle fino", 0.35, 0, 1, 0.05, "", "Suma ojos, bordes y texturas de la foto al relieve.", "detalle"),
    choice("tamano_detalle", "Tamaño del detalle", "medio", [["fino", "Fino (pelo, poros)"], ["medio", "Medio (ojos, bordes)"], ["grueso", "Grueso (formas)"]],
           "Qué tan grandes son los rasgos de la foto que se suman al relieve.", "detalle"),
    num("contraste_local", "Contraste local", 0.0, 0, 1, 0.05, "", "Saca relieve de las zonas planas (CLAHE) sin quemar lo más alto ni lo más bajo.", "detalle"),
    num("suavizado", "Suavizado", 0.6, 0, 5, 0.1, "px", "Quita el ruido de la profundidad; más alto deja la superficie más lisa.", "detalle"),
    num("bisel_mm", "Bisel del borde", 0.8, 0, 3, 0.1, "mm", "Baja el relieve suave hacia el borde en vez de cortarlo a pico.", "detalle"),
    num("contorno_suave", "Suavizar contorno", 0.2, 0, 1.5, 0.05, "mm", "Redondea los escalones del contorno de la silueta.", "detalle"),
    choice("modelo_profundidad", "Modelo de profundidad", "base",
           [["small", "Small · 99 MB, rápido"], ["base", "Base · 371 MB, más detalle"], ["large", "Large · 1,3 GB, máximo detalle"]],
           "Base y Large se descargan solos la primera vez y son de uso no comercial. Si no hay internet, usa Small.", "profundidad"),
    choice("resolucion_profundidad", "Resolución de profundidad", 728,
           [[518, "518 (rápida)"], [728, "728"], [1022, "1022 (fina)"], [1288, "1288 (muy fina, lenta)"]],
           "Con cuántos píxeles mira la foto el modelo de profundidad. Más alto ve detalles más chicos.", "profundidad"),
    num("resolucion_mm", "Resolución", 0.15, 0.08, 0.5, 0.01, "mm", "Tamaño de cada punto de la malla. Más chico = más detalle.", "profundidad"),
    choice("detalle_max", "Detalle máximo", 1.5, DETALLE_MAX, DETALLE_MAX_AYUDA, "profundidad"),
]

FIGURE_PARAMS = [
    choice("calidad", "Calidad", "maxima", [["maxima", "Máxima (más lenta)"], ["alta", "Alta"], ["rapida", "Rápida"]],
           "Máxima: malla de 512 y 50 pasos. Tarda más, pero se parece más a la foto.", "calidad"),
    num("intentos", "Intentos (mejor de N)", 1, 1, 4, 1, "",
        "Genera varias figuras con semillas distintas y se queda con la que más se parece a la silueta de la foto. Cada intento suma el tiempo de una figura.",
        "calidad"),
    num("encuadre", "Tamaño del sujeto", 85, 70, 95, 1, "%", "Cuánto del cuadro ocupa el sujeto en la imagen que recibe Hunyuan3D (85 % es lo de Tencent).", "foto"),
    flag("realzar", "Realzar la foto", False, "Contraste automático y nitidez suave antes de mandarla a la IA; ayuda con fotos lavadas o borrosas.", "foto"),
    choice("algoritmo", "Superficie", "suave", [["suave", "Suave (surface net)"], ["rapido", "Rápida (escalonada)"]], group="muestreo"),
    num("umbral", "Umbral de superficie", 0.6, 0.3, 0.9, 0.05, "",
        "Más bajo engrosa la figura y recupera partes finas (orejas, colas, dedos); más alto la afina.", "muestreo"),
    choice("latentes", "Detalle de la forma", 0,
           [[0, "Según el modelo"], [3072, "3072"], [4096, "4096"], [6144, "6144 (experimental)"], [8192, "8192 (experimental)"]],
           "Cuántos «tokens» describen la forma. Hunyuan3D 2.1 se entrenó con 4096: más puede sumar detalle o deformarla.", "muestreo"),
    num("pasos", "Pasos de muestreo", 0, 0, 80, 1, "", "0 = según la calidad.", "muestreo"),
    num("guia", "Guía (CFG)", 0, 0, 12, 0.5, "", "0 = la del modelo (2.1: 5 · 2.0: 8). Más alto se pega más a la foto, demasiado la deforma.", "muestreo"),
    choice("muestreador", "Muestreador", "euler",
           [["euler", "Euler (recomendado)"], ["heun", "Heun (más fino, el doble de lento)"], ["dpmpp_2m", "DPM++ 2M"], ["uni_pc", "UniPC"]],
           group="muestreo"),
    choice("programador", "Programador", "normal", [["normal", "Normal"], ["simple", "Simple"], ["beta", "Beta"], ["sgm_uniform", "SGM uniforme"]],
           group="muestreo"),
    num("shift", "Desplazamiento (shift)", 1.0, 0.5, 5, 0.1, "", "Reparto de los pasos del muestreo. 1 = plantilla oficial de ComfyUI.", "muestreo"),
    num("semilla", "Semilla", 0, 0, 2**31 - 1, 1, "", "0 = aleatoria. Con varios intentos se usan semilla, semilla+1…", "muestreo"),
]

FIGURE_PRINT = [
    num("tamano_mm", "Tamaño", 100, 10, 300, 1, group="tamano"),
    choice("eje", "Medido por", "alto", [["alto", "Altura"], ["largo", "Lado más largo"]], group="tamano"),
    choice("base", "Base", "ninguna", [["ninguna", "Sin base"], ["disco", "Disco"], ["rectangular", "Rectangular"]], group="tamano"),
    num("base_mm", "Alto de la base", 3, 1, 10, 0.5, group="tamano"),
    num("aplanar_mm", "Aplanar abajo", 1.0, 0, 6, 0.1, "mm", "Corta una rebanada para que pegue a la cama.", "tamano"),
    num("rotacion_z", "Girar", 0, 0, 360, 15, "°", group="tamano"),
    flag("color", "Colores de la foto", True, "Pinta la figura con los colores de la imagen (vista previa y OBJ a color).", "superficie"),
    choice("pulido", "Pulido de superficie", "suave", [["ninguno", "Ninguno"], ["suave", "Suave"], ["medio", "Medio"], ["fuerte", "Fuerte"]],
           "Quita el escalonado de la malla sin encoger la pieza. Más fuerte también borra detalle fino.", "superficie"),
    num("engrosar_mm", "Engrosar partes finas", 0, 0, 1.5, 0.1, "mm",
        "Engrosa toda la superficie para que orejas, colas y dedos aguanten. Cada lado crece esto.", "superficie"),
    flag("reparar", "Reparar malla", True, "Cierra agujeros y arregla caras para que el slicer la acepte.", "malla"),
    flag("solida", "Rehacer siempre por vóxeles", False, "Quita caras internas y huecos escondidos aunque la malla parezca sana.", "malla"),
    choice("voxeles", "Detalle al reparar", 512, [[256, "256 (rápido)"], [384, "384"], [512, "512"], [640, "640 (lento)"]],
           "Resolución con la que se rehace la malla cuando hace falta. Más alto conserva más detalle.", "malla"),
    num("fragmentos", "Fragmentos sueltos", 2, 0, 10, 0.5, "%", "Borra pedacitos sueltos más chicos que este % de la pieza; 0 los conserva todos.", "malla"),
    choice("caras_max", "Triángulos máximos", 1_000_000, CARAS, CARAS_AYUDA, "malla"),
    flag("dividir", "Dividir si no cabe", False, group="tamano"),
]

COMMON_PRINT = [
    num("rotacion_z", "Girar", 0, 0, 360, 15, "°", group="tamano"),
    flag("dividir", "Dividir si no cabe", False, group="tamano"),
    choice("caras_max", "Triángulos máximos", 1_000_000, CARAS, CARAS_AYUDA, "malla"),
]


def profile(key, label, hint, params=None, printing=None):
    return {"clave": key, "nombre": label, "ayuda": hint, "parametros": params or {}, "impresion": printing or {}}


def quality_profiles(params: dict[str, tuple], printing: dict[str, tuple], names=("rapida", "equilibrada", "maxima", "ultra")) -> list[dict]:
    """Arma los perfiles a partir de {clave: (valor rápida, equilibrada, máxima, ultra)}."""
    labels = {
        "rapida": ("Rápida", "Para probar: menos detalle, sale en una fracción del tiempo."),
        "equilibrada": ("Equilibrada", "Buen detalle sin esperar tanto."),
        "maxima": ("Máxima", "La de siempre: el máximo detalle que vale la pena imprimir."),
        "ultra": ("Ultra", "Todo al tope, aunque tarde mucho más y los archivos pesen más."),
    }
    out = []
    for i, key in enumerate(names):
        out.append(profile(key, *labels[key], {k: v[i] for k, v in params.items()}, {k: v[i] for k, v in printing.items()}))
    return out


RELIEF_PROFILES = quality_profiles(
    {"modelo_profundidad": ("small", "small", "base", "large"), "resolucion_profundidad": (518, 728, 728, 1022),
     "resolucion_mm": (0.3, 0.2, 0.15, 0.1), "detalle_max": (0.7, 0.7, 1.5, 3.0)},
    {"caras_max": (250_000, 500_000, 1_000_000, 2_000_000)},
)
GRID_PROFILES = quality_profiles(
    {"resolucion_mm": (0.3, 0.2, 0.15, 0.1), "detalle_max": (0.7, 0.7, 1.5, 3.0)},
    {"caras_max": (250_000, 500_000, 1_000_000, 2_000_000)},
)

FORMS = [
    {
        "clave": "figura",
        "nombre": "Figura 3D (IA)",
        "descripcion": "Modelo completo generado por IA con Hunyuan3D. Para personajes, mascotas, juguetes y objetos.",
        "requiere": "generador",
        "parametros": FIGURE_PARAMS,
        "impresion": FIGURE_PRINT,
        "perfiles": quality_profiles(
            {"calidad": ("rapida", "alta", "maxima", "maxima"), "intentos": (1, 1, 1, 3), "pasos": (0, 0, 0, 70)},
            {"voxeles": (256, 384, 512, 640), "caras_max": (250_000, 500_000, 1_000_000, 2_000_000), "solida": (False, False, False, True)},
        ),
    },
    {
        "clave": "relieve",
        "nombre": "Relieve",
        "descripcion": "Placa con la imagen en relieve, en silueta o dentro de una forma.",
        "parametros": [
            choice("contorno", "Contorno", "silueta", CONTORNOS, group="forma"),
            num("tamano_mm", "Tamaño (lado mayor)", 80, 20, 250, 1, group="forma"),
            num("base_mm", "Grosor de la base", 2.0, 0.6, 10, 0.1, group="forma"),
            num("relieve_mm", "Alto del relieve", 3.0, 0.4, 15, 0.1, group="forma"),
            num("borde_mm", "Borde", 2.0, 0, 10, 0.5, group="forma"),
            flag("aro", "Aro elevado en placas", True, group="forma"),
            *RELIEF_COMMON,
        ],
        "perfiles": RELIEF_PROFILES,
    },
    {
        "clave": "litofania",
        "nombre": "Litofanía",
        "descripcion": "Lámina que revela la foto con luz detrás: plana, curva o cilindro para lámpara.",
        "parametros": [
            choice("forma", "Forma", "plana", [["plana", "Plana"], ["curva", "Curva"], ["cilindro", "Cilindro (lámpara)"]], group="forma"),
            num("ancho_mm", "Ancho", 100, 40, 250, 1, group="forma"),
            num("diametro_mm", "Diámetro (cilindro)", 80, 40, 200, 1, group="forma"),
            num("angulo", "Ángulo (curva)", 120, 30, 300, 5, "°", group="forma"),
            num("grosor_min_mm", "Grosor mínimo", 0.8, 0.4, 2.0, 0.1, group="forma"),
            num("grosor_max_mm", "Grosor máximo", 3.0, 1.5, 6.0, 0.1, group="forma"),
            num("marco_mm", "Marco", 3.0, 0, 10, 0.5, group="forma"),
            flag("invertir", "Invertir", False, group="tonos"),
            num("gamma", "Gamma de tonos", 1.0, 0.4, 2.5, 0.05, "", "Menos de 1 aclara las sombras; más de 1 las oscurece.", "tonos"),
            num("contraste_local", "Contraste local", 0.15, 0, 1, 0.05, "", "Marca los detalles en zonas de luz pareja (cielos, piel) sin quemar los extremos.", "tonos"),
            num("nitidez", "Nitidez", 0.25, 0, 1.5, 0.05, "", "Enfoca bordes y texturas para que se lean con la luz detrás.", "tonos"),
            num("suavizado", "Suavizado", 0.6, 0, 3, 0.1, "px", "Quita el grano de la foto antes de enfocar.", "tonos"),
            num("resolucion_mm", "Resolución", 0.15, 0.08, 0.5, 0.01, "mm", "Tamaño de cada punto. 0,15 o menos lee detalles finos.", "resolucion"),
            choice("detalle_max", "Detalle máximo", 1.5, DETALLE_MAX, DETALLE_MAX_AYUDA, "resolucion"),
        ],
        "perfiles": GRID_PROFILES,
    },
    {
        "clave": "llavero",
        "nombre": "Llavero",
        "descripcion": "Silueta o medalla pequeña con argolla para la cadenita.",
        "parametros": [
            choice("contorno", "Contorno", "silueta", CONTORNOS, group="forma"),
            num("tamano_mm", "Tamaño", 50, 20, 100, 1, group="forma"),
            num("base_mm", "Grosor", 2.5, 1.2, 6, 0.1, group="forma"),
            num("relieve_mm", "Relieve", 1.5, 0, 6, 0.1, group="forma"),
            num("borde_mm", "Borde", 2.0, 0.5, 6, 0.5, group="forma"),
            num("argolla_mm", "Argolla (diámetro)", 9.0, 5, 16, 0.5, group="forma"),
            num("agujero_mm", "Agujero", 4.5, 2, 10, 0.5, group="forma"),
            flag("aro", "Aro elevado en placas", True, group="forma"),
            *RELIEF_COMMON,
        ],
        "perfiles": RELIEF_PROFILES,
    },
    {
        "clave": "iman",
        "nombre": "Imán de nevera",
        "descripcion": "Pieza plana con bolsillo trasero para un imán de neodimio.",
        "parametros": [
            choice("contorno", "Contorno", "silueta", CONTORNOS, group="forma"),
            num("tamano_mm", "Tamaño", 60, 20, 150, 1, group="forma"),
            num("base_mm", "Grosor", 3.0, 2, 8, 0.1, group="forma"),
            num("relieve_mm", "Relieve", 2.0, 0, 8, 0.1, group="forma"),
            num("borde_mm", "Borde", 2.0, 0, 8, 0.5, group="forma"),
            num("iman_diametro_mm", "Imán: diámetro", 10.2, 4, 30, 0.1, group="forma"),
            num("iman_alto_mm", "Imán: alto", 3.2, 1, 6, 0.1, group="forma"),
            flag("aro", "Aro elevado en placas", True, group="forma"),
            *RELIEF_COMMON,
        ],
        "perfiles": RELIEF_PROFILES,
    },
    {
        "clave": "pixel",
        "nombre": "Pixel art multicolor",
        "descripcion": "Bloques de colores; cada color es una pieza del 3MF para su filamento (AMS/MMU).",
        "parametros": [
            num("pixeles", "Píxeles (lado mayor)", 32, 8, 96, 1, "", group="forma"),
            num("colores", "Colores", 4, 2, 8, 1, "", group="forma"),
            num("tamano_mm", "Tamaño", 100, 30, 250, 1, group="forma"),
            num("alto_mm", "Alto de los píxeles", 2.0, 0.6, 6, 0.1, group="forma"),
            num("placa_mm", "Placa de fondo", 1.2, 0, 4, 0.1, "mm", "0 = sin placa", "forma"),
            flag("con_fondo", "Incluir el fondo", False, group="forma"),
            choice("espacio_color", "Agrupar colores", "lab", [["lab", "Como los ve el ojo (Lab)"], ["rgb", "RGB"]],
                   "Lab separa mejor pieles, cielos y sombras parecidas.", "colores"),
            choice("muestreo", "Color de cada píxel", "dominante", [["dominante", "El dominante (nítido)"], ["promedio", "El promedio (suave)"]],
                   "Dominante evita colores mezclados en los bordes; promedio se ve más suave.", "colores"),
            flag("limpiar", "Quitar píxeles sueltos", True, "Un píxel sin vecinos de su color toma el de alrededor: menos cambios de filamento.", "colores"),
        ],
        "perfiles": quality_profiles(
            {"espacio_color": ("rgb", "lab", "lab"), "muestreo": ("promedio", "promedio", "dominante"), "limpiar": (False, True, True)},
            {"caras_max": (250_000, 500_000, 1_000_000)},
            names=("rapida", "equilibrada", "maxima"),
        ),
    },
    {
        "clave": "cortador",
        "nombre": "Cortador de galletas",
        "descripcion": "Filo con la silueta y pestaña para empujar. Mejor con formas simples.",
        "parametros": [
            num("tamano_mm", "Tamaño", 80, 30, 200, 1, group="forma"),
            num("pared_mm", "Filo", 1.0, 0.6, 2.0, 0.1, group="forma"),
            num("alto_mm", "Alto", 14, 8, 25, 0.5, group="forma"),
            num("pestana_mm", "Pestaña", 4, 0, 10, 0.5, group="forma"),
            num("pestana_alto_mm", "Alto de la pestaña", 2, 1, 4, 0.1, group="forma"),
            num("detalle_min_mm", "Detalle mínimo", 1.5, 0.4, 5, 0.1, "mm", "Salientes más finos que esto se quitan: el filo no los puede seguir.", "detalle"),
            num("resolucion_mm", "Resolución", 0.15, 0.08, 0.5, 0.01, "mm", "Tamaño de cada punto del contorno. Más chico = filo más fiel.", "detalle"),
            choice("detalle_max", "Detalle máximo", 1.5, DETALLE_MAX, DETALLE_MAX_AYUDA, "detalle"),
        ],
        "perfiles": GRID_PROFILES,
    },
]

for _f in FORMS:
    _f.setdefault("impresion", COMMON_PRINT)
    _used = [p["grupo"] for p in [*_f["parametros"], *_f["impresion"]]]
    _f["grupos"] = [{"clave": g, "nombre": GRUPOS[g][0], "plegado": GRUPOS[g][1]} for g in dict.fromkeys(_used)]
