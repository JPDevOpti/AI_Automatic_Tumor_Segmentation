import numpy as np
import nibabel as nib
# import the Gaussian filter function from the scipy library
import scipy.ndimage as ndimage 

SIGMA = 1

def apply_gaussian_filter(input_path, output_path, sigma=SIGMA):
    """
    Applies a Gaussian filter to a T2-weighted MRI image in NIfTI format and saves the result.
    
    Parameters:
    - input_path (str): Path to the input NIfTI file.
    - output_path (str): Path where the filtered NIfTI image will be saved.
    - sigma (float): Standard deviation for the Gaussian kernel.
    """
    print(f"Applying Gaussian filter with sigma={sigma} to '{input_path}'...")
    # Load the NIfTI image
    nifti_image = nib.load(input_path)
    image_data = nifti_image.get_fdata()

    # Apply the Gaussian filter
    filtered_image = ndimage.gaussian_filter(image_data, sigma=sigma)

    # Create a new NIfTI image with the filtered data
    filtered_nifti = nib.Nifti1Image(filtered_image, affine=nifti_image.affine)

    # Save the filtered image to the specified output path
    nib.save(filtered_nifti, output_path)
    print(f"Filtered image saved as '{output_path}'.")
