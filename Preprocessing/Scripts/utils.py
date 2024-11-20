import nibabel as nib
import matplotlib.pyplot as plt
import os

def load_nii(archivo):
    """
    Carga un archivo .nii (NIfTI) y devuelve la imagen como un array de NumPy.
    
    :param archivo: Ruta al archivo .nii
    :return: Imagen cargada como un array de NumPy
    """
    imagen_nifti = nib.load(archivo)
    return imagen_nifti
  
def extract_folder(archivo, indice=1):
    """
    Extrae una carpeta específica de la ruta del archivo.
    
    :param archivo: Ruta del archivo
    :param indice: Índice de la carpeta deseada (0 es la carpeta raíz)
    :return: Nombre de la carpeta especificada
    """
    partes_ruta = os.path.normpath(archivo).split(os.sep)
    if indice < len(partes_ruta):
        return partes_ruta[indice]
    else:
        return "Carpeta no encontrada"

def show_subplots(imagen1, archivo1, imagen2, archivo2, capa=50, indice_carpeta=1):
    """
    Muestra una capa específica de dos imágenes de resonancia magnética en subgráficas.
    
    :param imagen1: Primer array de imagen cargada
    :param archivo1: Ruta del primer archivo para extraer el nombre de la carpeta
    :param imagen2: Segundo array de imagen cargada
    :param archivo2: Ruta del segundo archivo para extraer el nombre de la carpeta
    :param capa: Índice de la capa a mostrar
    :param indice_carpeta: Índice de la carpeta a mostrar en el título
    """
    carpeta1 = extract_folder(archivo1, indice_carpeta)
    carpeta2 = extract_folder(archivo2, indice_carpeta)
    
    plt.figure(figsize=(12, 6))
    
    # Primer subplot
    plt.subplot(1, 2, 1)
    plt.imshow(imagen1[:, :, capa], cmap='gray')
    plt.title(f'Capa {capa} - Carpeta: {carpeta1}')
    plt.axis('off')
    
    # Segundo subplot
    plt.subplot(1, 2, 2)
    plt.imshow(imagen2[:, :, capa], cmap='gray')
    plt.title(f'Capa {capa} - Carpeta: {carpeta2}')
    plt.axis('off')
    
    plt.show()