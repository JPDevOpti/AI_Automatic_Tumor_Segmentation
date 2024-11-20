import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from skimage.metrics import structural_similarity as ssim

# Directory root for preprocessed and original images
preprocessed_dir = '../../dataset/preprocessed/outputs2'
original_dir = '../../dataset/preprocessed/T1GD_not_processed2'

# Load Nifti image format function 
def load_nifti_image(file_path):
    img = nib.load(file_path)
    return img.get_fdata()

# Basic statistics function
def calculate_statistics(image_data):
    mean_intensity = np.mean(image_data)
    std_intensity = np.std(image_data)
    min_intensity = np.min(image_data)
    max_intensity = np.max(image_data)
    
    return mean_intensity, std_intensity, min_intensity, max_intensity

# Kolmogorov-Smirnov test to compare distributions
def ks_test(original_data, preprocessed_data):
    ks_stat, p_value = ks_2samp(original_data.flatten(), preprocessed_data.flatten())
    return ks_stat, p_value

# Smoothing analysis using Fourier transform
def fourier_analysis(original_data, preprocessed_data):
    original_fft = np.fft.fftn(original_data)
    preprocessed_fft = np.fft.fftn(preprocessed_data)
    
    original_magnitude = np.abs(original_fft)
    preprocessed_magnitude = np.abs(preprocessed_fft)
    
    return original_magnitude, preprocessed_magnitude

# Registration analysis using structural similarity
def registration_analysis(original_data, preprocessed_data):
    original_norm = (original_data - np.min(original_data)) / (np.max(original_data) - np.min(original_data))
    preprocessed_norm = (preprocessed_data - np.min(preprocessed_data)) / (np.max(preprocessed_data) - np.min(preprocessed_data))
    
    ssim_index = ssim(original_norm, preprocessed_norm, data_range=preprocessed_norm.max() - preprocessed_norm.min())
    
    return ssim_index

# Initialize lists to store results
stats_preprocessed = []
stats_original = []
ks_results = []
ssim_results = []
fourier_results = []

# Iterate through all the images in the preprocessed image directory
for file_name in os.listdir(preprocessed_dir):
    print(f"Processing image: {file_name}")
    if file_name.endswith('.nii') or file_name.endswith('.nii.gz'):
        preprocessed_path = os.path.join(preprocessed_dir, file_name)
        original_file_name = file_name.replace('.nii_brain_extracted.nii', '.nii.gz')
        original_path = os.path.join(original_dir, original_file_name)
        
        preprocessed_data = load_nifti_image(preprocessed_path)
        original_data = load_nifti_image(original_path)
        
        stats_preprocessed.append(calculate_statistics(preprocessed_data))
        stats_original.append(calculate_statistics(original_data))
        
        ks_results.append(ks_test(original_data, preprocessed_data))
        
        ssim_results.append(registration_analysis(original_data, preprocessed_data))
        
        fourier_results.append(fourier_analysis(original_data, preprocessed_data))

# Calculate averages
avg_stats_preprocessed = np.mean(stats_preprocessed, axis=0)
avg_stats_original = np.mean(stats_original, axis=0)
avg_ks_stat, avg_p_value = np.mean(ks_results, axis=0)
avg_ssim = np.mean(ssim_results)

# Plot average histograms
def plot_average_histogram(data_list, title):
    avg_data = np.mean([d.flatten() for d in data_list], axis=0)
    plt.figure(figsize=(10, 6))
    plt.hist(avg_data, bins=100, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.show()

plot_average_histogram([load_nifti_image(os.path.join(preprocessed_dir, f)) for f in os.listdir(preprocessed_dir) if f.endswith('.nii') or f.endswith('.nii.gz')],
                       'Average Histogram - Preprocessed Images')
plot_average_histogram([load_nifti_image(os.path.join(original_dir, f)) for f in os.listdir(original_dir) if f.endswith('.nii.gz')],
                       'Average Histogram - Original Images')

# Plot average Fourier analysis
avg_original_magnitude = np.mean([r[0] for r in fourier_results], axis=0)
avg_preprocessed_magnitude = np.mean([r[1] for r in fourier_results], axis=0)

plt.figure(figsize=(10, 6))
plt.loglog(np.sort(avg_original_magnitude.flatten())[::-1], label='Original')
plt.loglog(np.sort(avg_preprocessed_magnitude.flatten())[::-1], label='Preprocessed')
plt.title('Average Fourier Transform - Frequency Magnitudes')
plt.legend()
plt.show()

# Print global summary
print("### GLOBAL SUMMARY ###")
print(f"Average Preprocessed statistics (Mean, Std. Dev, Min, Max): {avg_stats_preprocessed}")
print(f"Average Original statistics (Mean, Std. Dev, Min, Max): {avg_stats_original}")
print(f"Average KS test (Statistic, p-value): {avg_ks_stat}, {avg_p_value}")
print(f"Average SSIM index (Registration): {avg_ssim}")