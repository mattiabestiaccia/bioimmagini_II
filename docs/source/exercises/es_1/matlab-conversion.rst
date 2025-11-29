Confronto MATLAB-Python - Esercizio 1
======================================

Questa sezione documenta la conversione del codice MATLAB originale in Python,
evidenziando le differenze sintattiche e le equivalenze funzionali.

Panoramica Files
----------------

Corrispondenza tra script MATLAB e Python:

+---------------------------+--------------------------------+
| MATLAB (originale)        | Python (convertito)            |
+===========================+================================+
| ``Calcolo_SD.m``          | ``calcolo_sd.py``              |
+---------------------------+--------------------------------+
| ``EsempioCalcoloSD.m``    | ``esempio_calcolo_sd.py``      |
+---------------------------+--------------------------------+
| ``Test_m_SD.m``           | ``test_m_sd.py``               |
+---------------------------+--------------------------------+
| N/A                       | ``utils.py`` (funzioni comuni) |
+---------------------------+--------------------------------+

Differenze Sintattiche Principali
----------------------------------

1. Calcolo SD Map
~~~~~~~~~~~~~~~~~

**MATLAB** (funzione built-in):

.. code-block:: matlab

   % Crea kernel
   kernel = ones(5, 5);

   % Calcola SD map
   SD_map = stdfilt(image, kernel);

**Python** (implementazione custom):

.. code-block:: python

   from scipy import ndimage
   import numpy as np

   def compute_sd_map(image, kernel_size=5):
       """Equivalente di stdfilt MATLAB."""
       footprint = np.ones((kernel_size, kernel_size))
       sd_map = ndimage.generic_filter(
           image,
           np.std,
           footprint=footprint,
           mode='constant',
           cval=0.0
       )
       return sd_map

**Note**:
- MATLAB ha ``stdfilt`` built-in, Python richiede ``scipy.ndimage``
- Python usa ``generic_filter`` con funzione ``np.std``
- Comportamento ai bordi: Python usa ``mode='constant'`` per replicare MATLAB

2. Creazione Immagine Sintetica
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**MATLAB**:

.. code-block:: matlab

   % Dimensioni
   dim = 512;

   % Immagine base
   I = ones(dim, dim) * 50;

   % Pattern con diverse intensità
   I(50:100, 50:100) = 120;
   I(101:180, 101:450) = 200;
   I(200:500, 200:350) = 90;
   I(230:270, 230:270) = 250;
   I(5:400, 450:500) = 150;

   % Aggiungi rumore Gaussiano
   sigma = 5;
   noise = sigma * randn(dim, dim);
   I_noisy = I + noise;

**Python**:

.. code-block:: python

   import numpy as np

   def create_synthetic_image(dim=512, sigma_noise=5.0):
       """Crea immagine sintetica identica a MATLAB."""
       # Immagine base
       image = np.ones((dim, dim)) * 50.0

       # Pattern (nota: Python è 0-indexed)
       image[50:101, 50:101] = 120
       image[101:181, 101:451] = 200
       image[200:501, 200:351] = 90
       image[230:271, 230:271] = 250
       image[5:401, 450:501] = 150

       # Rumore Gaussiano
       noise = np.random.normal(0, sigma_noise, (dim, dim))
       image_noisy = image + noise

       return image, image_noisy

**Differenze chiave**:

- **Indexing**: MATLAB 1-indexed, Python 0-indexed

  - MATLAB: ``I(50:100, 50:100)`` include 51 elementi
  - Python: ``image[50:101, 50:101]`` per ottenere stesso range

- **Rumore**:

  - MATLAB: ``randn(dim, dim)`` genera N(0,1), moltiplicato per sigma
  - Python: ``np.random.normal(0, sigma, size)`` direttamente

3. Istogrammi e Statistiche
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**MATLAB**:

.. code-block:: matlab

   % Istogramma
   [counts, bin_centers] = hist(SD_map(:), 100);

   % Trova massimo
   [~, max_idx] = max(counts);
   sigma_max = bin_centers(max_idx);

   % Statistiche
   sigma_mean = mean(SD_map(:));
   sigma_median = median(SD_map(:));
   sigma_std = std(SD_map(:));

**Python**:

.. code-block:: python

   import numpy as np

   # Istogramma
   counts, bin_edges = np.histogram(sd_map.ravel(), bins=100)
   bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

   # Trova massimo
   max_idx = np.argmax(counts)
   sigma_max = bin_centers[max_idx]

   # Statistiche
   sigma_mean = np.mean(sd_map)
   sigma_median = np.median(sd_map)
   sigma_std = np.std(sd_map)

**Differenze**:

- **Flatten array**: MATLAB usa ``(:)``, Python usa ``.ravel()`` o ``.flatten()``
- **Histogram output**: MATLAB restituisce centri, Python restituisce edges
- **Statistiche**: Sintassi simile, Python non richiede ``.ravel()`` per mean/median

4. Lettura DICOM
~~~~~~~~~~~~~~~~~

**MATLAB**:

.. code-block:: matlab

   % Leggi DICOM
   info = dicominfo('data/phantom.dcm');
   image = dicomread('data/phantom.dcm');

   % Converti a double
   image = double(image);

**Python**:

.. code-block:: python

   import pydicom

   # Leggi DICOM
   dcm = pydicom.dcmread('data/phantom.dcm')
   image = dcm.pixel_array.astype(float)

   # Accesso metadati
   print(f"Patient Name: {dcm.PatientName}")
   print(f"Image Size: {dcm.Rows} x {dcm.Columns}")

**Vantaggi Python**:

- ``pydicom`` combina lettura immagine e metadati in un oggetto unico
- Accesso diretto ai metadati come attributi
- Conversione tipo più esplicita con ``.astype()``

5. Visualizzazione
~~~~~~~~~~~~~~~~~~

**MATLAB**:

.. code-block:: matlab

   % Figure con subplot
   figure;

   subplot(2, 2, 1);
   imshow(I_noisy, []);
   title('Immagine Rumorosa');
   colorbar;

   subplot(2, 2, 2);
   imshow(SD_map, []);
   title('SD Map');
   colormap('hot');
   colorbar;

   subplot(2, 2, 3);
   histogram(I_noisy(:), 256);
   title('Istogramma');

**Python**:

.. code-block:: python

   import matplotlib.pyplot as plt

   fig, axes = plt.subplots(2, 2, figsize=(12, 10))

   # Subplot 1
   im1 = axes[0, 0].imshow(image_noisy, cmap='gray')
   axes[0, 0].set_title('Immagine Rumorosa')
   axes[0, 0].axis('off')
   plt.colorbar(im1, ax=axes[0, 0])

   # Subplot 2
   im2 = axes[0, 1].imshow(sd_map, cmap='hot')
   axes[0, 1].set_title('SD Map')
   axes[0, 1].axis('off')
   plt.colorbar(im2, ax=axes[0, 1])

   # Subplot 3
   axes[1, 0].hist(image_noisy.ravel(), bins=256)
   axes[1, 0].set_title('Istogramma')

   plt.tight_layout()
   plt.show()

**Differenze**:

- MATLAB: ``subplot(m, n, i)`` con indexing lineare
- Python: ``plt.subplots(m, n)`` restituisce array 2D di axes
- MATLAB: ``imshow(I, [])`` auto-scala, Python richiede ``vmin/vmax`` espliciti o normalizzazione
- Python: ``plt.tight_layout()`` per spaziatura automatica

6. Correzione Rayleigh
~~~~~~~~~~~~~~~~~~~~~~~

**MATLAB**:

.. code-block:: matlab

   % Fattore correzione Rayleigh
   rayleigh_factor = sqrt(2 / (4 - pi));

   % Applica correzione
   sigma_corrected = sigma_background * rayleigh_factor;

   fprintf('Sigma background: %.2f\n', sigma_background);
   fprintf('Sigma corrected:  %.2f\n', sigma_corrected);
   fprintf('Fattore:          %.3f\n', rayleigh_factor);

**Python**:

.. code-block:: python

   import numpy as np

   def rayleigh_correction_factor():
       """Fattore correzione Rayleigh per rumore MRI background."""
       return np.sqrt(2.0 / (4.0 - np.pi))

   def apply_rayleigh_correction(sigma_background):
       """Applica correzione Rayleigh."""
       return sigma_background * rayleigh_correction_factor()

   # Uso
   sigma_corrected = apply_rayleigh_correction(sigma_background)

   print(f"Sigma background: {sigma_background:.2f}")
   print(f"Sigma corrected:  {sigma_corrected:.2f}")
   print(f"Fattore:          {rayleigh_correction_factor():.3f}")

**Vantaggi Python**:

- Funzioni riutilizzabili con docstring
- Costanti matematiche: ``np.pi`` invece di ``pi``
- f-strings per formatting più leggibile

7. Test Monte Carlo
~~~~~~~~~~~~~~~~~~~~

**MATLAB**:

.. code-block:: matlab

   % Parametri
   roi_sizes = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200];
   n_iter = 1000;

   % Loop su ROI sizes
   for i = 1:length(roi_sizes)
       roi_size = roi_sizes(i);

       % Loop iterazioni
       for j = 1:n_iter
           % ROI casuale
           y = randi([1, size(I,1) - roi_size]);
           x = randi([1, size(I,2) - roi_size]);

           roi = I(y:y+roi_size-1, x:x+roi_size-1);

           % Calcola SD
           sigma_est = std(roi(:));
           results{i}(j) = sigma_est;
       end
   end

**Python**:

.. code-block:: python

   import numpy as np

   # Parametri
   roi_sizes = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200]
   n_iter = 1000

   # Dizionario risultati
   results = {size: [] for size in roi_sizes}

   # Loop su ROI sizes
   for roi_size in roi_sizes:
       # Loop iterazioni
       for _ in range(n_iter):
           # ROI casuale (0-indexed!)
           max_y = image.shape[0] - roi_size
           max_x = image.shape[1] - roi_size

           y = np.random.randint(0, max_y)
           x = np.random.randint(0, max_x)

           roi = image[y:y+roi_size, x:x+roi_size]

           # Calcola SD
           sigma_est = np.std(roi)
           results[roi_size].append(sigma_est)

**Differenze chiave**:

- MATLAB: 1-indexed, ``randi([1, max])``
- Python: 0-indexed, ``np.random.randint(0, max)`` (esclude max)
- MATLAB: Cell array ``results{i}``
- Python: Dizionario ``results[roi_size]`` (più leggibile)
- MATLAB: ``std(roi(:))`` flattens
- Python: ``np.std(roi)`` funziona su array multidimensionali

Tabella Equivalenze Rapide
---------------------------

Funzioni Comuni
~~~~~~~~~~~~~~~

+--------------------------------+----------------------------------------+
| MATLAB                         | Python (NumPy/SciPy)                   |
+================================+========================================+
| ``ones(m, n)``                 | ``np.ones((m, n))``                    |
+--------------------------------+----------------------------------------+
| ``zeros(m, n)``                | ``np.zeros((m, n))``                   |
+--------------------------------+----------------------------------------+
| ``randn(m, n)``                | ``np.random.randn(m, n)``              |
+--------------------------------+----------------------------------------+
| ``size(A, 1)``                 | ``A.shape[0]``                         |
+--------------------------------+----------------------------------------+
| ``length(v)``                  | ``len(v)`` o ``v.size``                |
+--------------------------------+----------------------------------------+
| ``A(:)``                       | ``A.ravel()`` o ``A.flatten()``        |
+--------------------------------+----------------------------------------+
| ``mean(A(:))``                 | ``np.mean(A)``                         |
+--------------------------------+----------------------------------------+
| ``std(A(:))``                  | ``np.std(A)``                          |
+--------------------------------+----------------------------------------+
| ``median(A(:))``               | ``np.median(A)``                       |
+--------------------------------+----------------------------------------+
| ``max(A(:))``                  | ``np.max(A)`` o ``A.max()``            |
+--------------------------------+----------------------------------------+
| ``[val, idx] = max(A)``        | ``idx = np.argmax(A); val = A[idx]``  |
+--------------------------------+----------------------------------------+
| ``find(A > threshold)``        | ``np.where(A > threshold)``            |
+--------------------------------+----------------------------------------+
| ``A(A > threshold)``           | ``A[A > threshold]``                   |
+--------------------------------+----------------------------------------+
| ``sqrt(x)``                    | ``np.sqrt(x)``                         |
+--------------------------------+----------------------------------------+
| ``pi``                         | ``np.pi``                              |
+--------------------------------+----------------------------------------+

Operazioni su Immagini
~~~~~~~~~~~~~~~~~~~~~~

+--------------------------------+----------------------------------------+
| MATLAB                         | Python                                 |
+================================+========================================+
| ``imshow(I, [])``              | ``plt.imshow(I, cmap='gray')``         |
+--------------------------------+----------------------------------------+
| ``stdfilt(I, ones(5,5))``      | ``compute_sd_map(I, kernel_size=5)``   |
+--------------------------------+----------------------------------------+
| ``imerode(BW, se)``            | ``ndimage.binary_erosion(BW, se)``    |
+--------------------------------+----------------------------------------+
| ``imdilate(BW, se)``           | ``ndimage.binary_dilation(BW, se)``   |
+--------------------------------+----------------------------------------+
| ``bwlabel(BW)``                | ``ndimage.label(BW)``                  |
+--------------------------------+----------------------------------------+
| ``regionprops(BW, 'Area')``    | ``measure.regionprops(BW)``            |
+--------------------------------+----------------------------------------+

I/O Files
~~~~~~~~~

+--------------------------------+----------------------------------------+
| MATLAB                         | Python                                 |
+================================+========================================+
| ``dicomread('file.dcm')``      | ``pydicom.dcmread('file.dcm')``        |
|                                | ``.pixel_array``                       |
+--------------------------------+----------------------------------------+
| ``dicominfo('file.dcm')``      | ``pydicom.dcmread('file.dcm')``        |
|                                | (metadati come attributi)              |
+--------------------------------+----------------------------------------+
| ``save('data.mat', 'var')``    | ``np.save('data.npy', var)``           |
+--------------------------------+----------------------------------------+
| ``load('data.mat')``           | ``var = np.load('data.npy')``          |
+--------------------------------+----------------------------------------+

Miglioramenti della Versione Python
------------------------------------

1. **Modularità**

   - Funzioni riutilizzabili in ``utils.py``
   - Separazione logica tra calcolo e visualizzazione

2. **Documentazione**

   - Docstring NumPy-style per ogni funzione
   - Type hints per parametri (opzionale)

3. **Gestione Errori**

   .. code-block:: python

      def compute_sd_map(image, kernel_size=5):
          if kernel_size % 2 == 0:
              raise ValueError("kernel_size deve essere dispari")
          if kernel_size < 3:
              raise ValueError("kernel_size deve essere >= 3")
          # ... resto della funzione

4. **Testing**

   - Unit test con pytest
   - Validazione output contro risultati MATLAB

5. **Flessibilità**

   .. code-block:: python

      # Parametri configurabili via argparse
      parser = argparse.ArgumentParser()
      parser.add_argument('--kernel-size', type=int, default=5)
      parser.add_argument('--sigma', type=float, default=5.0)
      args = parser.parse_args()

Validazione Conversione
------------------------

Per verificare la correttezza della conversione:

.. code-block:: python

   # 1. Esegui versione MATLAB
   # >> run Calcolo_SD.m
   # >> save('matlab_results.mat', 'sigma_mean', 'sigma_median', 'SD_map')

   # 2. Esegui versione Python
   from src.calcolo_sd import main
   results_py = main(show_plots=False)

   # 3. Confronta risultati
   import scipy.io
   matlab_results = scipy.io.loadmat('matlab_results.mat')

   sigma_mean_matlab = matlab_results['sigma_mean'][0, 0]
   sigma_mean_python = results_py['sigma_mean']

   diff = abs(sigma_mean_matlab - sigma_mean_python)
   assert diff < 1e-6, f"Differenza troppo grande: {diff}"

   print("✓ Conversione validata: risultati identici!")

Risorse Aggiuntive
-------------------

- **NumPy for MATLAB users**: https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
- **PyDICOM Documentation**: https://pydicom.github.io/
- **SciPy ndimage**: https://docs.scipy.org/doc/scipy/reference/ndimage.html
- **Matplotlib Gallery**: https://matplotlib.org/stable/gallery/index.html
