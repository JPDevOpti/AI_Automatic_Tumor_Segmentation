import os
import nibabel as nib
from tkinter import filedialog, messagebox, ttk
import tkinter as tk
from nilearn.image import resample_to_img
from skimage import exposure, filters
from scipy import ndimage
import subprocess
import tempfile

# Setup FSL environment (for brain extraction)
os.environ['FSLDIR'] = '/usr/local/fsl'
os.environ['PATH'] = os.environ['FSLDIR'] + '/bin:' + os.environ['PATH']
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'

def skull_stripping(input_image, output_image, frac):
    command = ['bet', input_image, output_image, '-f', str(frac), '-g', '0']
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Brain extraction completed. Output saved as {output_image}")
    else:
        print("Error in brain extraction:", result.stderr)

def reoriented_image(reference_image_path, image_to_reorient_path):
    image_to_reorient = nib.load(image_to_reorient_path)
    reference_image = nib.load(reference_image_path)
    reoriented_img = resample_to_img(image_to_reorient, reference_image, interpolation='nearest')
    return reoriented_img

def normalize_image(image):
    image_data = image.get_fdata()
    normalized_data = exposure.rescale_intensity(image_data, out_range=(0, 1))
    normalized_img = nib.Nifti1Image(normalized_data, image.affine, image.header)
    return normalized_img

def median_filter_image(image, size=2):
    image_data = image.get_fdata()
    filtered_data = ndimage.median_filter(image_data, size=size)
    filtered_img = nib.Nifti1Image(filtered_data, image.affine, image.header)
    return filtered_img

def apply_gaussian_filter(image, sigma=0.1):
    image_data = image.get_fdata()
    smoothed_data = ndimage.gaussian_filter(image_data, sigma=sigma)
    smoothed_img = nib.Nifti1Image(smoothed_data, image.affine, image.header)
    return smoothed_img

def adaptive_histogram_equalization(image, clip_limit=0.01):
    image_data = image.get_fdata()
    enhanced_data = exposure.equalize_adapthist(image_data, clip_limit=clip_limit)
    enhanced_img = nib.Nifti1Image(enhanced_data, image.affine, image.header)
    return enhanced_img

def enhance_edges(image, weight=1.5):
    image_data = image.get_fdata()
    edges = filters.sobel(image_data)
    enhanced_data = image_data + weight * edges
    enhanced_img = nib.Nifti1Image(enhanced_data, image.affine, image.header)
    return enhanced_img

def open_folder_dialog(entry_widget):
    folder_path = filedialog.askdirectory()
    entry_widget.delete(0, tk.END)
    entry_widget.insert(0, folder_path)
    return folder_path

def open_file_dialog(entry_widget):
    file_path = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.nii.gz *.nii")])
    entry_widget.delete(0, tk.END)
    entry_widget.insert(0, file_path)
    return file_path

def process_all_images(input_folder, reference_image, output_folder, filter_size, sigma, clip_limit, weight, frac, progress_bar):
    files = [f for f in os.listdir(input_folder) if f.endswith(".nii.gz")]
    total_files = len(files)
    for i, filename in enumerate(files):
        raw_img_path = os.path.join(input_folder, filename)
        output_img_path = os.path.join(output_folder, filename)
        reoriented_img = reoriented_image(reference_image, raw_img_path)
        if reoriented_img is not None:
            normalized_img = normalize_image(reoriented_img)
            median_filtered_img = median_filter_image(normalized_img, size=filter_size)
            gaussian_filtered_img = apply_gaussian_filter(median_filtered_img, sigma=sigma)
            adapted_histogram_img = adaptive_histogram_equalization(gaussian_filtered_img, clip_limit=clip_limit)
            edge_enhanced_img = enhance_edges(adapted_histogram_img, weight=weight)
            with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as temp_file:
                temp_file_path = temp_file.name
                nib.save(edge_enhanced_img, temp_file_path)
            skull_stripping(temp_file_path, output_img_path, frac)
            os.remove(temp_file_path)
        progress_bar['value'] = (i + 1) / total_files * 100
        progress_bar.update()

def execute_process(input_folder_entry, ref_img_entry, output_folder_entry, filter_size_var, sigma_var, clip_limit_var, weight_var, frac_var, progress_bar):
    input_folder = input_folder_entry.get()
    ref_img_path = ref_img_entry.get()
    output_folder = output_folder_entry.get()
    filter_size = int(filter_size_var.get())
    sigma = float(sigma_var.get())
    clip_limit = float(clip_limit_var.get())
    weight = float(weight_var.get())
    frac = float(frac_var.get())
    if not all([input_folder, ref_img_path, output_folder]):
        messagebox.showerror("Input error", "All paths must be specified.")
        return
    process_all_images(input_folder, ref_img_path, output_folder, filter_size, sigma, clip_limit, weight, frac, progress_bar)

window = tk.Tk()
window.title("Preprocessing sRMI")

tk.Label(window, text="Input Folder:").grid(row=0, column=0)
input_folder_entry = tk.Entry(window, width=40)
input_folder_entry.grid(row=0, column=1)
tk.Button(window, text="Browse", command=lambda: open_folder_dialog(input_folder_entry)).grid(row=0, column=2)

tk.Label(window, text="Reference Image:").grid(row=1, column=0)
ref_img_entry = tk.Entry(window, width=40)
ref_img_entry.grid(row=1, column=1)
tk.Button(window, text="Browse", command=lambda: open_file_dialog(ref_img_entry)).grid(row=1, column=2)

tk.Label(window, text="Output Folder:").grid(row=2, column=0)
output_folder_entry = tk.Entry(window, width=40)
output_folder_entry.grid(row=2, column=1)
tk.Button(window, text="Browse", command=lambda: open_folder_dialog(output_folder_entry)).grid(row=2, column=2)

tk.Label(window, text="Median Filter Size:").grid(row=3, column=0)
filter_size_var = tk.StringVar(value='2')
filter_size_entry = tk.Entry(window, textvariable=filter_size_var)
filter_size_entry.grid(row=3, column=1)

tk.Label(window, text="Gaussian Filter Sigma:").grid(row=4, column=0)
sigma_var = tk.StringVar(value='0.1')
sigma_entry = tk.Entry(window, textvariable=sigma_var)
sigma_entry.grid(row=4, column=1)

tk.Label(window, text="AHE Clip Limit:").grid(row=5, column=0)
clip_limit_var = tk.StringVar(value='0.01')
clip_limit_entry = tk.Entry(window, textvariable=clip_limit_var)
clip_limit_entry.grid(row=5, column=1)

tk.Label(window, text="Edge Enhancement Weight:").grid(row=6, column=0)
weight_var = tk.StringVar(value='1.5')
weight_entry = tk.Entry(window, textvariable=weight_var)
weight_entry.grid(row=6, column=1)

tk.Label(window, text="Skull Stripping Fraction:").grid(row=7, column=0)
frac_var = tk.StringVar(value='0.45')
frac_entry = tk.Entry(window, textvariable=frac_var)
frac_entry.grid(row=7, column=1)

progress_bar = ttk.Progressbar(window, orient="horizontal", length=400, mode="determinate")
progress_bar.grid(row=8, column=0, columnspan=3, pady=10)

tk.Button(window, text="Execute Process", command=lambda: execute_process(input_folder_entry, ref_img_entry, output_folder_entry, filter_size_var, sigma_var, clip_limit_var, weight_var, frac_var, progress_bar)).grid(row=9, column=0, columnspan=3, sticky=tk.W+tk.E)

window.mainloop()