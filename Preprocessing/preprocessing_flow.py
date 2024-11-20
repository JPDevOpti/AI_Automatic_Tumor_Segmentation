from Scripts.utils import load_nii
from Scripts.co_registration import reorient_image
from Scripts.normalization import normalize_image
from Scripts.brain_extraction import extract_brain
import os

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
  brain_extracted_output_path = os.path.join(output_path, 'brain_extracted.nii')
  extract_brain(temp_output_path, brain_extracted_output_path)

  print(f"Proceso de preprocesamiento finalizado. Imagen guardada en {brain_extracted_output_path}")

  # delete the temporary file
  os.remove(temp_output_path)
  
  return brain_extracted_output_path



reference_image_path = '../DataBase/Preprocessed_Subjects/UPENN-GBM-00001_11_T2.nii'  # temporal reference for reorientation
input_image_path = '../DataBase/No_Preprocessed_Subject/UPENN-GBM-00096_11_T2_unstripped.nii'
output_path = './outputs'

result = preprocess_image(input_image_path, reference_image_path, output_path)
print(result)