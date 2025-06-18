import matplotlib.pyplot as plt
import numpy as np
from utils import detectar_bordes, extraer_contorno_superior_interactivo

# - Carga y procesa una imagen.
# - Detecta bordes.
# - Muestra contornos candidatos.
# - Permite al usuario seleccionar uno.
# - Interpola con splines cúbicos y muestra los resultados.


def main():
    ruta_imagen = 'src/guinea-pig.jpg'

    try:
        # Detecta bordes en la imagen.
        bordes, imagen_original = detectar_bordes(ruta_imagen)

        # Muestra modo interactivo para elegir contorno y graficar spline.
        extraer_contorno_superior_interactivo(bordes, imagen_original)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
