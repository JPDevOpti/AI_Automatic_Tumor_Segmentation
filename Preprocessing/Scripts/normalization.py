import nibabel as nib
import numpy as np

def normalize_image(img):
    """
    Normaliza la escala de grises de una imagen y guarda el resultado.

    Parameters:
    - image_path (nibabel.nifti1.Nifti1Image): Imagen a normalizar.
    returns:
    - normalized_img (nibabel.nifti1.Nifti1Image): Imagen normalizada.
    """
    # Cargar la imagen
    data = img.get_fdata()

    # Normalizar la escala de grises
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)

    # Crear una nueva imagen con los datos normalizados
    normalized_img = nib.Nifti1Image(normalized_data, img.affine, img.header)
    
    return normalized_img