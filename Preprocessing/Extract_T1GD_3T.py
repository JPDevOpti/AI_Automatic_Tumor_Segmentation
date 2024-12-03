import os
import shutil
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox

def select_input_file():
    input_file = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if input_file:
        select_source_folder(input_file)

def select_source_folder(input_file):
    source_folder = filedialog.askdirectory(title="Seleccionar carpeta de origen")
    if source_folder:
        select_destination_folder(input_file, source_folder)

def select_destination_folder(input_file, source_folder):
    destination_folder = filedialog.askdirectory(title="Seleccionar carpeta de destino")
    if destination_folder:
        # Procesar los datos del CSV
        process_data(input_file, source_folder, destination_folder)

def process_data(input_file, source_folder, destination_folder):
    try:
        # Cargar el archivo CSV
        data = pd.read_csv(input_file)

        # Extraer la columna 'ID'
        ids_permitidos = data['ID'].tolist()

        # Eliminar carpetas no permitidas
        eliminar_carpetas(source_folder, ids_permitidos)

        # Crear la carpeta de destino para T1GD
        t1gd_folder = os.path.join(destination_folder, 'T1GD_not_processed')
        if not os.path.exists(t1gd_folder):
            os.makedirs(t1gd_folder)

        # Eliminar archivos no T1GD
        eliminar_archivos_no_t1gd(source_folder)

        # Extraer archivos T1GD
        extract_t1gd_files(source_folder, t1gd_folder)

        # Mostrar mensaje de éxito
        messagebox.showinfo("Éxito", "Eliminación de carpetas y extracción de archivos completadas.")

    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error: {str(e)}")

def eliminar_carpetas(carpeta_principal, ids_permitidos):
    for carpeta in os.listdir(carpeta_principal):
        ruta_carpeta = os.path.join(carpeta_principal, carpeta)
        if os.path.isdir(ruta_carpeta) and carpeta not in ids_permitidos:
            print(f'Eliminando carpeta: {ruta_carpeta}')
            try:
                shutil.rmtree(ruta_carpeta)  # Eliminar la carpeta
            except OSError as e:
                print(f'Error al eliminar la carpeta {ruta_carpeta}: {e}')

def eliminar_archivos_no_t1gd(source_folder):
    # Recorrer todas las subcarpetas en el directorio de origen
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            # Verificar si el archivo NO termina en '_T1GD_unstripped.nii.gz'
            if not file.endswith('_T1GD_unstripped.nii.gz'):
                file_path = os.path.join(root, file)
                # Eliminar el archivo
                os.remove(file_path)
                print(f"Archivo eliminado: {file_path}")

def extract_t1gd_files(source_folder, destination_folder):
    # Recorrer todas las subcarpetas en el directorio de origen
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            # Verificar si el archivo termina en '_T1GD_unstripped.nii.gz'
            if file.endswith('_T1GD_unstripped.nii.gz'):
                file_path = os.path.join(root, file)
                # Copiar el archivo a la carpeta de destino
                shutil.copy(file_path, destination_folder)
                print(f"Archivo copiado: {file}")

if __name__ == "__main__":
    # Crear la ventana principal
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal

    # Iniciar el proceso mostrando solo el diálogo de selección de archivo CSV
    select_input_file()

    # Iniciar el bucle principal
    root.mainloop()
