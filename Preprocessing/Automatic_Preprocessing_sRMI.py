import os
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog, messagebox
import tkinter as tk
from nilearn.image import resample_to_img
from skimage import exposure, filters
from scipy import ndimage
import subprocess
import tempfile

# Setup FSL environment (for brain extraction)
# Ensure FSLDIR is set
os.environ['FSLDIR'] = '/usr/local/fsl'
os.environ['PATH'] = os.environ['FSLDIR'] + '/bin:' + os.environ['PATH']

# Set FSLOUTPUTTYPE environment variable
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'


# Function to perform skull stripping
def skull_stripping(input_image, output_image, frac):
    command = [
        'bet', input_image, output_image,
        '-f', str(frac), '-g', '0'
    ]
    print(f"Running command: {' '.join(command)}")  # Added for debugging purposes
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Brain extraction completed. Output saved as {output_image}")
    else:
        print("Error in brain extraction:", result.stderr)

# Definition of the function to display slices
def load_and_show_sRMI(fig, ax, nifti_path, slice_index, slice_type='axial'):
    """
    Loads a NIfTI image and shows a specific slice in the desired plane.
    """
    # Clear the previous plot
    ax.clear()

    # Load the NIfTI image
    img = nib.load(nifti_path)
    data = img.get_fdata()

    # Check and show the slice in the specified plane
    if slice_type == 'axial':
        if slice_index < 0 or slice_index >= data.shape[2]:
            print(f"Index out of range for axial slice. It must be between 0 and {data.shape[2] - 1}.")
            return
        ax.imshow(data[:, :, slice_index], cmap="gray")
        ax.set_title(f"Axial Slice - Layer {slice_index}")

    elif slice_type == 'sagittal':
        if slice_index < 0 or slice_index >= data.shape[0]:
            print(f"Index out of range for sagittal slice. It must be between 0 and {data.shape[0] - 1}.")
            return
        ax.imshow(data[slice_index, :, :], cmap="gray")
        ax.set_title(f"Sagittal Slice - Layer {slice_index}")

    elif slice_type == 'coronal':
        if slice_index < 0 or slice_index >= data.shape[1]:
            print(f"Layer out of range for coronal slice. It must be between 0 and {data.shape[1] - 1}.")
            return
        ax.imshow(data[:, slice_index, :], cmap="gray")
        ax.set_title(f"Coronal Slice - Layer {slice_index}")

    else:
        print("Slice type not recognized. Use 'axial', 'sagittal', or 'coronal'.")
        return

    ax.axis("off")
    fig.canvas.draw()


# Define the reorientation function
def reoriented_image(reference_image_path, image_to_reorient_path):
    """
    Reorient an image to the orientation of a reference image and return the reoriented image data.
    """
    # Verify if the files exist
    if not (os.path.exists(reference_image_path) and os.path.exists(image_to_reorient_path)):
        raise FileNotFoundError("One or both of the files were not found.")

    try:
        # Load the images
        image_to_reorient = nib.load(image_to_reorient_path)  # Image to be reoriented
        reference_image = nib.load(reference_image_path)  # Reference image
        # Reorient the image
        reoriented_img = resample_to_img(image_to_reorient, reference_image, interpolation='nearest')
        return reoriented_img

    except Exception as e:
        print(f"Error loading the images or performing the reorientation: {e}")
        return None


# Define the normalization function
def normalize_image(image):
    """
    Normalize the intensity of a NIfTI image.
    """
    # Get the image data
    image_data = image.get_fdata()

    # Normalize the image intensity
    normalized_data = exposure.rescale_intensity(image_data, out_range=(0, 1))

    # Create a new NIfTI object with the normalized data
    normalized_img = nib.Nifti1Image(normalized_data, image.affine, image.header)

    return normalized_img


# Define the median filter function
def median_filter_image(image, size=2):
    """
    Apply a median filter to a NIfTI image.
    """
    # Get the image data
    image_data = image.get_fdata()

    # Apply the median filter
    filtered_data = ndimage.median_filter(image_data, size=size)

    # Create a new NIfTI object with the filtered data
    filtered_img = nib.Nifti1Image(filtered_data, image.affine, image.header)

    return filtered_img


# Define the Gaussian filter function
def apply_gaussian_filter(image, sigma=0.1):
    """
    Apply a Gaussian filter to a NIfTI image.
    """
    # Get the image data
    image_data = image.get_fdata()

    # Apply the Gaussian filter
    smoothed_data = ndimage.gaussian_filter(image_data, sigma=sigma)

    # Create a new NIfTI object with the smoothed data
    smoothed_img = nib.Nifti1Image(smoothed_data, image.affine, image.header)

    return smoothed_img


# Define the adaptive histogram equalization function
def adaptive_histogram_equalization(image, clip_limit=0.01):
    """
    Apply adaptive histogram equalization to a NIfTI image.
    """
    # Get the image data
    image_data = image.get_fdata()

    # Apply adaptive histogram equalization
    enhanced_data = exposure.equalize_adapthist(image_data, clip_limit=clip_limit)

    # Create a new NIfTI object with the enhanced contrast data
    enhanced_img = nib.Nifti1Image(enhanced_data, image.affine, image.header)

    return enhanced_img


# Define the edge enhancement function
def enhance_edges(image, weight=1.5):
    """
    Enhance the edges using the Sobel filter.
    """
    # Get the image data
    image_data = image.get_fdata()

    # Apply the Sobel filter to enhance the edges
    edges = filters.sobel(image_data)
    enhanced_data = image_data + weight * edges

    # Create a new NIfTI object with the enhanced edges
    enhanced_img = nib.Nifti1Image(enhanced_data, image.affine, image.header)

    return enhanced_img


# Define GUI-related functions
def open_file_dialog(entry_widget):
    """
    Opens a file dialog for selecting the NIfTI file and sets the entry widget text.
    """
    file_path = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.nii.gz *.nii")])
    entry_widget.delete(0, tk.END)
    entry_widget.insert(0, file_path)
    return file_path


def show_image(fig, ax, nifti_path, slice_index_var, slice_type_var):
    """
    Function to show the image based on user input.
    """
    if not nifti_path:
        messagebox.showerror("File error", "No file selected or invalid file type.")
        return

    try:
        slice_index = int(slice_index_var.get())
        slice_type = slice_type_var.get()
    except ValueError:
        messagebox.showerror("Input error", "Slice index must be an integer.")
        return

    load_and_show_sRMI(fig, ax, nifti_path, slice_index, slice_type)


def execute_reorientation_normalization_filters(raw_img_entry, ref_img_entry, skull_output_img_entry, show_fig, show_ax,
                                                filter_size_var, sigma_var, clip_limit_var, weight_var, frac_var):
    """
    Execute the reorientation, normalization, and filtering process.
    """
    raw_img_path = raw_img_entry.get()
    ref_img_path = ref_img_entry.get()
    skull_output_path = skull_output_img_entry.get()
    filter_size = int(filter_size_var.get())
    sigma = float(sigma_var.get())
    clip_limit = float(clip_limit_var.get())
    weight = float(weight_var.get())
    frac = float(frac_var.get())

    if not all([raw_img_path, ref_img_path, skull_output_path]):
        messagebox.showerror("Input error", "All file paths must be specified.")
        return

    reoriented_img = reoriented_image(ref_img_path, raw_img_path)
    if reoriented_img is not None:
        normalized_img = normalize_image(reoriented_img)
        median_filtered_img = median_filter_image(normalized_img, size=filter_size)
        gaussian_filtered_img = apply_gaussian_filter(median_filtered_img, sigma=sigma)
        adapted_histogram_img = adaptive_histogram_equalization(gaussian_filtered_img, clip_limit=clip_limit)
        edge_enhanced_img = enhance_edges(adapted_histogram_img, weight=weight)

        # Save the enhanced edge image as a temporary file
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as temp_file:
            temp_file_path = temp_file.name
            nib.save(edge_enhanced_img, temp_file_path)

        # Perform skull stripping on the saved enhanced edge image
        skull_stripping(temp_file_path, skull_output_path, frac)

        # Remove the temporary file after skull stripping
        os.remove(temp_file_path)

        # Check if the skull-stripped file was successfully created before proceeding to show it
        if not os.path.exists(skull_output_path):
            messagebox.showerror("File error", f"Skull stripping failed, output not found at {skull_output_path}.")
            return

        # Show the skull-stripped image
        load_and_show_sRMI(show_fig, show_ax, skull_output_path, int(slice_index_var.get()), slice_type_var.get())


# Create the main window
window = tk.Tk()
window.title("Preprocessing sRMI")

# Creating and placing widgets
# Raw image
tk.Label(window, text="Raw Image:").grid(row=0, column=0)
raw_img_entry = tk.Entry(window, width=40)
raw_img_entry.grid(row=0, column=1)
tk.Button(window, text="Browse", command=lambda: open_file_dialog(raw_img_entry)).grid(row=0, column=2)

# Reference image
tk.Label(window, text="Reference Image:").grid(row=1, column=0)
ref_img_entry = tk.Entry(window, width=40)
ref_img_entry.grid(row=1, column=1)
tk.Button(window, text="Browse", command=lambda: open_file_dialog(ref_img_entry)).grid(row=1, column=2)

# Skull stripping output image path
tk.Label(window, text="Output Image Path:").grid(row=2, column=0)
skull_output_img_entry = tk.Entry(window, width=40)
skull_output_img_entry.grid(row=2, column=1)
tk.Button(window, text="Save As", command=lambda: skull_output_img_entry.insert(0, filedialog.asksaveasfilename(
    defaultextension=".nii.gz", filetypes=[("NIfTI files", "*.nii.gz *.nii")]))).grid(row=2, column=2)

# Slice index and type
tk.Label(window, text="Slice Index:").grid(row=3, column=0)
slice_index_var = tk.StringVar(value='100')
slice_index_entry = tk.Entry(window, textvariable=slice_index_var)
slice_index_entry.grid(row=3, column=1)

tk.Label(window, text="Slice Type:").grid(row=4, column=0)
slice_type_var = tk.StringVar(value='axial')
slice_type_menu = tk.OptionMenu(window, slice_type_var, "axial", "sagittal", "coronal")
slice_type_menu.grid(row=4, column=1)

# Median filter size
tk.Label(window, text="Median Filter Size:").grid(row=5, column=0)
filter_size_var = tk.StringVar(value='2')
filter_size_entry = tk.Entry(window, textvariable=filter_size_var)
filter_size_entry.grid(row=5, column=1)

# Gaussian filter sigma
tk.Label(window, text="Gaussian Filter Sigma:").grid(row=6, column=0)
sigma_var = tk.StringVar(value='0.1')
sigma_entry = tk.Entry(window, textvariable=sigma_var)
sigma_entry.grid(row=6, column=1)

# Adaptive histogram equalization clip limit
tk.Label(window, text="AHE Clip Limit:").grid(row=7, column=0)
clip_limit_var = tk.StringVar(value='0.01')
clip_limit_entry = tk.Entry(window, textvariable=clip_limit_var)
clip_limit_entry.grid(row=7, column=1)

# Edge enhancement weight
tk.Label(window, text="Edge Enhancement Weight:").grid(row=8, column=0)
weight_var = tk.StringVar(value='1.5')
weight_entry = tk.Entry(window, textvariable=weight_var)
weight_entry.grid(row=8, column=1)

# Skull stripping fraction
tk.Label(window, text="Skull Stripping Fraction:").grid(row=9, column=0)
frac_var = tk.StringVar(value='0.45')
frac_entry = tk.Entry(window, textvariable=frac_var)
frac_entry.grid(row=9, column=1)

# Button to load and show raw image
tk.Button(window, text="Load Raw Image",
          command=lambda: show_image(raw_fig, raw_ax, raw_img_entry.get(), slice_index_var, slice_type_var)).grid(
    row=10, column=0)

# Button to load and show reference image
tk.Button(window, text="Load Reference Image",
          command=lambda: show_image(ref_fig, ref_ax, ref_img_entry.get(), slice_index_var, slice_type_var)).grid(
    row=10, column=1)

# Execute process button
tk.Button(window, text="Execute Process",
          command=lambda: execute_reorientation_normalization_filters(raw_img_entry, ref_img_entry, skull_output_img_entry,
                                                                      reoriented_fig, reoriented_ax, filter_size_var,
                                                                      sigma_var, clip_limit_var, weight_var,
                                                                      frac_var)).grid(row=10, column=2)

# Matplotlib figures and axes for displaying images horizontally
raw_fig, raw_ax = plt.subplots(figsize=(3, 3))
raw_canvas = FigureCanvasTkAgg(raw_fig, master=window)
raw_canvas.get_tk_widget().grid(row=11, column=0)

ref_fig, ref_ax = plt.subplots(figsize=(3, 3))
ref_canvas = FigureCanvasTkAgg(ref_fig, master=window)
ref_canvas.get_tk_widget().grid(row=11, column=1)

reoriented_fig, reoriented_ax = plt.subplots(figsize=(3, 3))
reoriented_canvas = FigureCanvasTkAgg(reoriented_fig, master=window)
reoriented_canvas.get_tk_widget().grid(row=11, column=2)

# Running the GUI
window.mainloop()