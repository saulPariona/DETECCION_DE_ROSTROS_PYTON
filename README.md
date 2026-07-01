# Sistema de Reconocimiento Facial para Control de Asistencia

Este proyecto es un sistema de visión artificial en tiempo real diseñado para identificar estudiantes mediante biometría facial. Utiliza el algoritmo LBPH (Local Binary Patterns Histograms) de la librería OpenCV.

---

## 1. Requisitos del Sistema

Antes de iniciar, asegúrate de contar con los siguientes elementos:

* Python 3.x instalado.
* Cámara web funcional (integrada o externa).
* Dependencias de Python: Instala los módulos requeridos ejecutando el siguiente comando en tu terminal:

```bash
pip install opencv-contrib-python numpy
```

Nota importante: Es obligatorio instalar opencv-contrib-python en lugar de la versión estándar, ya que el algoritmo LBPH se encuentra en los módulos extendidos de OpenCV.

---

## 2. Estructura del Proyecto

Organiza tus directorios y archivos de la siguiente manera para que el script localice las rutas de forma automática:

```text
asistencia_rostros/
├── opencv-files/
│   └── OpenCV-Face-Recognition-Python.py  # Script principal de ejecución
└── training-data/
    ├── Nombre_Estudiante_1/               # Carpeta con 5-10 fotos (.jpg o .png)
    └── Nombre_Estudiante_2/               # Carpeta con 5-10 fotos (.jpg o .png)
```

---

## 3. Configuración del Dataset

Para garantizar una alta precisión en el entrenamiento del modelo, sigue estas pautas:

1. Carpetas de usuarios: Dentro del directorio training-data/, crea una subcarpeta por cada estudiante. El nombre de la carpeta será la etiqueta que se mostrará en pantalla al reconocer el rostro.
2. Calidad de imagen: Agrega entre 5 y 10 fotos por persona. Asegúrate de que los rostros estén bien iluminados, centrados y mirando al frente.
3. Filtro de ruido: El sistema incluye un filtro de tamaño mínimo (w > 50 px y h > 50 px) para ignorar fondos o detecciones falsas y enfocarse solo en rostros reales.

---

## 4. Cómo Ejecutar el Sistema

1. Abre tu terminal o consola de comandos en la raíz del proyecto (asistencia_rostros/).
2. Ejecuta el script principal con el siguiente comando:

```bash
python opencv-files/OpenCV-Face-Recognition-Python.py
```

3. El sistema entrenará el modelo con las imágenes cargadas y abrirá automáticamente una ventana con el flujo de video en vivo.
4. Para salir: Presiona la tecla "q" mientras la ventana de video está activa.

---

## 5. Solución de Problemas Frecuentes

* FileNotFoundError (Error de ruta): Verifica que la consola esté abierta exactamente en la carpeta raíz asistencia_rostros/ antes de lanzar el comando.
* Detecciones falsas en el fondo: Si el sistema confunde objetos con rostros, incrementa el parámetro minNeighbors (por ejemplo, súbelo de 7 a 8 o 9) dentro de la función detect_face.
* Error AttributeError: module 'cv2' has no attribute 'face': Esto ocurre si instalaste la versión incorrecta de OpenCV. Desinstálala y vuelve a instalar la versión extendida ejecutando:
    ```bash
    pip uninstall opencv-python
    pip install opencv-contrib-python
    ```
