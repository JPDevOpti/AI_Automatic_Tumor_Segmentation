import os
import tkinter as tk
from tkinter import filedialog
from Scripts.utils import load_nii
from Scripts.co_registration import reorient_image
from Scripts.normalization import normalize_image
from Scripts.brain_extraction import extract_brain
from Scripts.filter import apply_gaussian_filter

def preprocess_image(input_image_path, reference_image_path, output_path):
    """
    Preprocesses the input NIfTI image by reorienting, normalizing, and extracting the brain.

    Args:
        input_image_path (str): Path to the input NIfTI image.
        reference_image_path (str): Path to the reference NIfTI image for reorientation.
        output_path (str): Directory where the output images will be saved.

    Returns:
        str: Path to the brain-extracted NIfTI image.
    """

    reference_nifti_image = load_nii(reference_image_path)
    input_nifti_image = load_nii(input_image_path)

    reoriented_nifti_image = reorient_image(reference_nifti_image, input_nifti_image)
    normalized_nifti_image = normalize_image(reoriented_nifti_image)

    # save the reoriented and normalized image to a temporary folder
    temp_output_path = os.path.join(output_path, 'temp.nii')
    normalized_nifti_image.to_filename(temp_output_path)

    # extract the brain from the normalized image
    extract_brain(temp_output_path, temp_output_path)
    
    #apply gaussian filter, save the result in the same path
    apply_gaussian_filter(temp_output_path, temp_output_path)

    # move the processed image to the final output path
    finished_output_path = os.path.join(output_path, f'{os.path.splitext(os.path.basename(input_image_path))[0]}_brain_extracted.nii')
    os.rename(temp_output_path, finished_output_path)
    print(f"Proceso de preprocesamiento finalizado. Imagen guardada en {finished_output_path}")

    return finished_output_path

def process_folder(input_folder, reference_image_path, output_folder):
    """
    Process all NIfTI files in the input folder and its subfolders that end with 'T1GD_unstripped.nii.gz'.

    Args:
        input_folder (str): Path to the input folder containing NIfTI files.
        reference_image_path (str): Path to the reference NIfTI image for reorientation.
        output_folder (str): Path to the output folder where processed images will be saved.
    """
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('T1GD_unstripped.nii.gz'):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_folder)
                output_path = os.path.join(output_folder, relative_path)
                os.makedirs(output_path, exist_ok=True)
                
                try:
                    result = preprocess_image(input_path, reference_image_path, output_path)
                    print(f"---- Procesado: {input_path} -> {result} \n")
                except Exception as e:
                    print(f"Error al procesar {input_path}: {str(e)}")

def select_folder():
    """
    Open a dialog to select a folder and return the selected path.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    folder_path = filedialog.askdirectory(title="Seleccione la carpeta de entrada")
    return folder_path

if __name__ == "__main__":
    reference_image_path = os.path.abspath('../DataBase/Preprocessed_Subjects/UPENN-GBM-00001_11_T1GD.nii')

    if not os.path.isfile(reference_image_path):
        print(f"Error: El archivo de referencia no se encuentra en {reference_image_path}")
        exit()
    
    input_folder = select_folder()
    if not input_folder:
        print("No se seleccionó ninguna carpeta. Saliendo...")
        exit()
    
    output_folder = os.path.join(os.path.dirname(input_folder), 'outputs')
    os.makedirs(output_folder, exist_ok=True)
    
    process_folder(input_folder, reference_image_path, output_folder)
    print("Procesamiento completo.")
