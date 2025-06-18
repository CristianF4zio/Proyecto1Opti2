import cv2
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# Detecta bordes en una imagen utilizando el algoritmo de Canny.
# Retorna: bordes (np.array): Imagen binaria con los bordes detectados. img (np.array): Imagen original en escala de grises.


def detectar_bordes(ruta_imagen):

    img = cv2.imread(
        ruta_imagen, cv2.IMREAD_GRAYSCALE)  # Lee imagen en escala de grises
    if img is None:
        raise ValueError(f"No se encontró la imagen en {ruta_imagen}")
    bordes = cv2.Canny(img, 50, 150)  # Detecta los bordes con Canny
    return bordes, img


# Permite al usuario seleccionar un contorno desde la terminal.
# Muestra visualmente los puntos y la interpolación con splines cúbicos.

def extraer_contorno_superior_interactivo(bordes, imagen_original):

    # Encuentra contornos externos.
    contornos, _ = cv2.findContours(
        bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidatos = []
    print(f"Total de contornos encontrados: {len(contornos)}")

    # Filtra y evalua contornos candidatos.
    for idx, c in enumerate(contornos):
        if len(c) < 300:
            continue  # Ignorar contornos pequeños
        x_vals = c[:, 0, 0]
        y_vals = c[:, 0, 1]
        ancho = x_vals.max() - x_vals.min()
        y_prom = np.mean(y_vals)
        y_min = y_vals.min()
        candidatos.append((idx, ancho, y_prom, y_min, c))

    if not candidatos:
        raise ValueError(
            "No se encontraron contornos suficientemente grandes.")

    # Muestra la tabla de contornos.
    print("\nContornos candidatos encontrados:\n")
    print(f"{'Idx':<5} {'Ancho':<8} {'Y_prom':<10} {'Y_min':<8} {'Puntos':<8}")
    print("-" * 45)
    for idx, ancho, y_prom, y_min, c in candidatos:
        print(f"{idx:<5} {ancho:<8.1f} {y_prom:<10.1f} {y_min:<8.1f} {len(c):<8}")
    print("\n")

    continuar = True
    while continuar:
        # Solicitarel índice (Idx) al usuario.
        try:
            idx_input = int(
                input("Ingresa el índice (Idx) del contorno a usar: "))
            seleccionado = [c for c in candidatos if c[0] == idx_input][0]
        except (ValueError, IndexError):
            print("Índice inválido. Usando el contorno más ancho por defecto.")

            # Ordenar por ancho descendente.
            candidatos.sort(key=lambda x: -x[1])
            seleccionado = candidatos[0]

        idx_seleccionado, ancho, y_prom, y_min, contorno_superior = seleccionado
        print(
            f"\nContorno seleccionado: #{idx_seleccionado} (ancho: {ancho:.1f}, y_prom: {y_prom:.1f})")

        # Extrae coordenadas X e Y del contorno.
        x = contorno_superior[:, 0, 0]
        y = contorno_superior[:, 0, 1]

        # Ordena los puntos de izquierda a derecha (por X ascendente).
        indices_ordenados = np.argsort(x)
        x = x[indices_ordenados]
        y = y[indices_ordenados]

        # Elimina valores duplicados en X para evitar errores en interpolación.
        x_unicos, indices_unicos = np.unique(x, return_index=True)
        y_unicos = y[indices_unicos]

        print(f"Puntos extraídos: {len(x_unicos)}")

        # Interpolación cúbica
        cs = CubicSpline(x_unicos, y_unicos)
        x_nuevo = np.linspace(
            x_unicos.min(), x_unicos.max(), 3 * len(x_unicos))
        y_nuevo = cs(x_nuevo)

        # Grafica los resultados.
        plt.figure(figsize=(12, 6))

        # Imagen original con puntos y spline.
        plt.subplot(1, 2, 1)
        plt.imshow(imagen_original, cmap='gray')
        plt.title(f'Contorno #{idx_seleccionado} - Imagen Original')
        plt.scatter(x_unicos, y_unicos, s=10, c='red', label='Puntos')
        plt.plot(x_nuevo, y_nuevo, c='green', label='Spline Cúbico')
        plt.legend()

        # Imagen de bordes.
        plt.subplot(1, 2, 2)
        plt.imshow(bordes, cmap='gray')
        plt.title('Bordes Detectados')

        plt.tight_layout()
        # Guarda resultado con el nombre "resultado_spline.png".
        plt.savefig("resultado_spline.png", dpi=300)
        print("Imagen guardada como resultado_spline.png")
        plt.show()

        # Pregunta si se desea intentar con otro contorno
        respuesta = input(
            "¿Quieres probar otro contorno? (s/n): ").strip().lower()
        if respuesta != 's':
            continuar = False


# Interpola los puntos dados utilizando splines cúbicos.
# Retorna: x_nuevo (np.array): Puntos X interpolados. y_nuevo (np.array): Puntos Y interpolados.

def interpolar_spline(x, y, factor=3):

    cs = CubicSpline(x, y)
    x_nuevo = np.linspace(x.min(), x.max(), factor * len(x))
    y_nuevo = cs(x_nuevo)
    return x_nuevo, y_nuevo
