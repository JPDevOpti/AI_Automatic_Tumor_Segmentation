import os
from nilearn.image import resample_to_img

def reorient_image(reference_image, image_to_reorient):
    """
    Reorienta una imagen a la orientación de una imagen de referencia y guarda el resultado.

    Parameters:
    reference_image (nibabel.nifti1.Nifti1Image): Imagen de referencia para la reorientación.
    image_to_reorient (nibabel.nifti1.Nifti1Image): Imagen a reorientar.
    returns: 
    reoriented_img (nibabel.nifti1.Nifti1Image): Imagen reorientada.
    """
    try:
        # Comprobar los tipos de datos
        """ print("Tipo de imagen a reorientar:", type(reference_image))  # Debería mostrar <class 'nibabel.nifti1.Nifti1Image'>
        print("Tipo de imagen de referencia:", type(image_to_reorient))  # Debería mostrar <class 'nibabel.nifti1.Nifti1Image'> """
        
        # Mostrar información básica de las imágenes
        """ print("Imagen a reorientar:")
        print("Tipo de datos:", reference_image.get_data_dtype())
        print("Forma:", reference_image.shape)
        
        print("\nImagen de referencia:")
        print("Tipo de datos:", image_to_reorient.get_data_dtype())
        print("Forma:", image_to_reorient.shape) """

        # Reorientar la imagen
        reoriented_img = resample_to_img(image_to_reorient, reference_image, interpolation='nearest')
        #reoriented_img = resample_to_img(reference_image, image_to_reorient, interpolation='nearest')
  
        return reoriented_img

    except Exception as e:
        print(f"Error al cargar las imágenes o realizar la reorientación: {e}")