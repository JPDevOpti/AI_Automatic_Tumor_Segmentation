import fsl.wrappers.bet as bet
import os

#Setup FSL environment (for brain extraction)

# Ensure FSLDIR is set
os.environ['FSLDIR'] = '/Users/imeag/fsl'
os.environ['PATH'] = os.environ['FSLDIR'] + '/bin:' + os.environ['PATH']

# Set FSLOUTPUTTYPE environment variable
#os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['FSLOUTPUTTYPE'] = 'NIFTI'

def extract_brain(input_path, output_path):
    """
    Extrae el cerebro de una imagen y guarda el resultado.

    Parameters:
    - input_path (str): Ruta del archivo .nii de la imagen.
    - output_path (str): Ruta donde se guardará la imagen con el cerebro extraído.
    """
    # Extraer el cerebro
    bet(input_path, output_path, f=0.32)