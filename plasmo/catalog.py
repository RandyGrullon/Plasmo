"""Catálogo de formas y sus parámetros: la interfaz se arma a partir de esto."""

from __future__ import annotations


def num(key, label, default, lo, hi, step=0.1, unit="mm", hint=""):
    return {"clave": key, "nombre": label, "tipo": "numero", "defecto": default, "min": lo, "max": hi, "paso": step, "unidad": unit, "ayuda": hint}


def choice(key, label, default, options, hint=""):
    return {"clave": key, "nombre": label, "tipo": "opcion", "defecto": default, "opciones": options, "ayuda": hint}


def flag(key, label, default, hint=""):
    return {"clave": key, "nombre": label, "tipo": "si_no", "defecto": default, "ayuda": hint}


CONTORNOS = [
    ["silueta", "Silueta"], ["rectangulo", "Rectángulo"], ["circulo", "Círculo"], ["hexagono", "Hexágono"],
    ["escudo", "Escudo"], ["estrella", "Estrella"], ["corazon", "Corazón"],
]
FUENTE = choice("fuente", "Relieve desde", "profundidad", [["profundidad", "Profundidad (IA)"], ["luminancia", "Luz y sombra"]],
                "Profundidad usa Depth Anything V2; luz y sombra sirve para dibujos planos.")
RELIEF_COMMON = [
    FUENTE,
    num("detalle", "Detalle fino", 0.35, 0, 1, 0.05, "", "Suma ojos, bordes y texturas de la foto al relieve."),
    num("bisel_mm", "Bisel del borde", 0.8, 0, 3, 0.1, "mm", "Baja el relieve suave hacia el borde en vez de cortarlo a pico."),
    flag("invertir", "Invertir relieve", False),
    choice("curva", "Curva", "linear", [["linear", "Lineal"], ["gamma", "Gamma"], ["s-curve", "Curva S"]]),
    num("gamma", "Gamma", 1.0, 0.2, 3.0, 0.1, ""),
    num("suavizado", "Suavizado", 0.6, 0, 5, 0.1, "px"),
]

FORMS = [
    {
        "clave": "figura",
        "nombre": "Figura 3D (IA)",
        "descripcion": "Modelo completo generado por IA con Hunyuan3D. Para personajes, mascotas, juguetes y objetos.",
        "requiere": "generador",
        "parametros": [
            choice("calidad", "Calidad", "maxima", [["maxima", "Máxima (más lenta)"], ["alta", "Alta"], ["rapida", "Rápida"]],
                   "Máxima: malla de 512 y 50 pasos. Tarda más, pero se parece más a la foto."),
            choice("algoritmo", "Superficie", "suave", [["suave", "Suave (surface net)"], ["rapido", "Rápida (escalonada)"]]),
            num("pasos", "Pasos de muestreo", 0, 0, 80, 1, "", "0 = según la calidad"),
            num("guia", "Guía (CFG)", 0, 0, 12, 0.5, "", "0 = la del modelo (2.1: 5 · 2.0: 8)"),
            num("semilla", "Semilla", 0, 0, 2**31 - 1, 1, "", "0 = aleatoria"),
        ],
        "impresion": [
            num("tamano_mm", "Tamaño", 100, 10, 300, 1),
            choice("eje", "Medido por", "alto", [["alto", "Altura"], ["largo", "Lado más largo"]]),
            flag("color", "Colores de la foto", True, "Pinta la figura con los colores de la imagen (vista previa y OBJ a color)."),
            choice("pulido", "Pulido de superficie", "suave", [["ninguno", "Ninguno"], ["suave", "Suave"], ["fuerte", "Fuerte"]],
                   "Quita el escalonado de la malla sin encoger la pieza."),
            choice("base", "Base", "ninguna", [["ninguna", "Sin base"], ["disco", "Disco"], ["rectangular", "Rectangular"]]),
            num("base_mm", "Alto de la base", 3, 1, 10, 0.5),
            num("aplanar_mm", "Aplanar abajo", 1.0, 0, 6, 0.1, "mm", "Corta una rebanada para que pegue a la cama."),
            num("rotacion_z", "Girar", 0, 0, 360, 15, "°"),
            flag("reparar", "Reparar malla", True),
            flag("dividir", "Dividir si no cabe", False),
        ],
    },
    {
        "clave": "relieve",
        "nombre": "Relieve",
        "descripcion": "Placa con la imagen en relieve, en silueta o dentro de una forma.",
        "parametros": [
            choice("contorno", "Contorno", "silueta", CONTORNOS),
            num("tamano_mm", "Tamaño (lado mayor)", 80, 20, 250, 1),
            num("base_mm", "Grosor de la base", 2.0, 0.6, 10, 0.1),
            num("relieve_mm", "Alto del relieve", 3.0, 0.4, 15, 0.1),
            num("borde_mm", "Borde", 2.0, 0, 10, 0.5),
            flag("aro", "Aro elevado en placas", True),
            *RELIEF_COMMON,
            num("resolucion_mm", "Resolución", 0.2, 0.1, 0.5, 0.05),
        ],
    },
    {
        "clave": "litofania",
        "nombre": "Litofanía",
        "descripcion": "Lámina que revela la foto con luz detrás: plana, curva o cilindro para lámpara.",
        "parametros": [
            choice("forma", "Forma", "plana", [["plana", "Plana"], ["curva", "Curva"], ["cilindro", "Cilindro (lámpara)"]]),
            num("ancho_mm", "Ancho", 100, 40, 250, 1),
            num("diametro_mm", "Diámetro (cilindro)", 80, 40, 200, 1),
            num("angulo", "Ángulo (curva)", 120, 30, 300, 5, "°"),
            num("grosor_min_mm", "Grosor mínimo", 0.8, 0.4, 2.0, 0.1),
            num("grosor_max_mm", "Grosor máximo", 3.0, 1.5, 6.0, 0.1),
            num("marco_mm", "Marco", 3.0, 0, 10, 0.5),
            flag("invertir", "Invertir", False),
            num("suavizado", "Suavizado", 0.6, 0, 3, 0.1, "px"),
            num("resolucion_mm", "Resolución", 0.2, 0.1, 0.5, 0.05),
        ],
    },
    {
        "clave": "llavero",
        "nombre": "Llavero",
        "descripcion": "Silueta o medalla pequeña con argolla para la cadenita.",
        "parametros": [
            choice("contorno", "Contorno", "silueta", CONTORNOS),
            num("tamano_mm", "Tamaño", 50, 20, 100, 1),
            num("base_mm", "Grosor", 2.5, 1.2, 6, 0.1),
            num("relieve_mm", "Relieve", 1.5, 0, 6, 0.1),
            num("borde_mm", "Borde", 2.0, 0.5, 6, 0.5),
            num("argolla_mm", "Argolla (diámetro)", 9.0, 5, 16, 0.5),
            num("agujero_mm", "Agujero", 4.5, 2, 10, 0.5),
            flag("aro", "Aro elevado en placas", True),
            *RELIEF_COMMON,
        ],
    },
    {
        "clave": "iman",
        "nombre": "Imán de nevera",
        "descripcion": "Pieza plana con bolsillo trasero para un imán de neodimio.",
        "parametros": [
            choice("contorno", "Contorno", "silueta", CONTORNOS),
            num("tamano_mm", "Tamaño", 60, 20, 150, 1),
            num("base_mm", "Grosor", 3.0, 2, 8, 0.1),
            num("relieve_mm", "Relieve", 2.0, 0, 8, 0.1),
            num("borde_mm", "Borde", 2.0, 0, 8, 0.5),
            num("iman_diametro_mm", "Imán: diámetro", 10.2, 4, 30, 0.1),
            num("iman_alto_mm", "Imán: alto", 3.2, 1, 6, 0.1),
            flag("aro", "Aro elevado en placas", True),
            *RELIEF_COMMON,
        ],
    },
    {
        "clave": "pixel",
        "nombre": "Pixel art multicolor",
        "descripcion": "Bloques de colores; cada color es una pieza del 3MF para su filamento (AMS/MMU).",
        "parametros": [
            num("pixeles", "Píxeles (lado mayor)", 32, 8, 96, 1, ""),
            num("colores", "Colores", 4, 2, 8, 1, ""),
            num("tamano_mm", "Tamaño", 100, 30, 250, 1),
            num("alto_mm", "Alto de los píxeles", 2.0, 0.6, 6, 0.1),
            num("placa_mm", "Placa de fondo", 1.2, 0, 4, 0.1, "mm", "0 = sin placa"),
            flag("con_fondo", "Incluir el fondo", False),
        ],
    },
    {
        "clave": "cortador",
        "nombre": "Cortador de galletas",
        "descripcion": "Filo con la silueta y pestaña para empujar. Mejor con formas simples.",
        "parametros": [
            num("tamano_mm", "Tamaño", 80, 30, 200, 1),
            num("pared_mm", "Filo", 1.0, 0.6, 2.0, 0.1),
            num("alto_mm", "Alto", 14, 8, 25, 0.5),
            num("pestana_mm", "Pestaña", 4, 0, 10, 0.5),
            num("pestana_alto_mm", "Alto de la pestaña", 2, 1, 4, 0.1),
        ],
    },
]

COMMON_PRINT = [
    num("rotacion_z", "Girar", 0, 0, 360, 15, "°"),
    flag("dividir", "Dividir si no cabe", False),
]
for _f in FORMS:
    _f.setdefault("impresion", COMMON_PRINT)
