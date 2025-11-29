Teoria - Analisi Rumore MRI
============================

Fondamenti del Rumore in Immagini MRI
--------------------------------------

Nelle immagini di risonanza magnetica (MRI), il rumore ha caratteristiche specifiche
che dipendono dalla modalità di acquisizione e ricostruzione del segnale.

Distribuzione del Rumore
~~~~~~~~~~~~~~~~~~~~~~~~~

Il rumore nelle immagini MRI presenta due comportamenti distinti:

**1. Regioni con segnale (Foreground)**

Nelle regioni anatomiche contenenti tessuto, il rumore segue una **distribuzione Gaussiana**:

.. math::

   I(x,y) = S(x,y) + N(x,y)

dove:
- :math:`I(x,y)` = intensità osservata
- :math:`S(x,y)` = segnale vero
- :math:`N(x,y) \sim \mathcal{N}(0, \sigma^2)` = rumore Gaussiano

**2. Background (Regioni senza segnale)**

Nel background, dove il segnale è assente, le immagini magnitude MRI seguono una
**distribuzione di Rayleigh** anziché Gaussiana.

Standard Deviation Map (SD Map)
--------------------------------

Metodo di Calcolo
~~~~~~~~~~~~~~~~~

La mappa di deviazione standard viene calcolata applicando una finestra scorrevole
sull'immagine e calcolando la SD locale:

.. math::

   SD_{map}(x,y) = \sqrt{\frac{1}{N-1} \sum_{i,j \in W} [I(i,j) - \mu_W]^2}

dove:
- :math:`W` = finestra locale (tipicamente 5×5)
- :math:`N` = numero di pixel nella finestra
- :math:`\mu_W` = media locale nella finestra

**Implementazione Python**:

.. code-block:: python

   from scipy import ndimage

   def compute_sd_map(image, kernel_size=5):
       footprint = np.ones((kernel_size, kernel_size))
       sd_map = ndimage.generic_filter(
           image, np.std, footprint=footprint
       )
       return sd_map

Stima del Rumore
~~~~~~~~~~~~~~~~

Tre metodi principali per stimare :math:`\sigma` dalla SD map:

1. **Media della SD map**: :math:`\hat{\sigma} = \text{mean}(SD_{map})`
2. **Mediana della SD map**: :math:`\hat{\sigma} = \text{median}(SD_{map})`
3. **Massimo dell'istogramma**: :math:`\hat{\sigma} = \arg\max_{\sigma} \text{hist}(SD_{map})`

Il metodo del **massimo dell'istogramma** è generalmente il più robusto poiché
è meno influenzato dai bordi e dalle regioni con variabilità di segnale.

Correzione di Rayleigh
-----------------------

Background Noise Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Nel background delle immagini MRI magnitude, il rumore segue la distribuzione di Rayleigh:

.. math::

   p(A) = \frac{A}{\sigma^2} e^{-A^2/(2\sigma^2)}, \quad A \geq 0

dove :math:`A` è la magnitude e :math:`\sigma` è il parametro di scala.

Fattore di Correzione
~~~~~~~~~~~~~~~~~~~~~~

La deviazione standard misurata nel background :math:`\sigma_{bkg}` deve essere
corretta per ottenere il vero :math:`\sigma` del rumore:

.. math::

   \sigma_{true} = \sigma_{bkg} \cdot \sqrt{\frac{2}{4-\pi}} \approx 1.526 \cdot \sigma_{bkg}

Questo fattore deriva dalle proprietà della distribuzione di Rayleigh:

- Media: :math:`\mu_{Rayleigh} = \sigma \sqrt{\pi/2}`
- Varianza: :math:`\text{Var}_{Rayleigh} = \sigma^2 (4-\pi)/2`

**Implementazione**:

.. code-block:: python

   def rayleigh_correction_factor():
       return np.sqrt(2.0 / (4.0 - np.pi))  # ≈ 1.526

   def apply_rayleigh_correction(sd_background):
       return sd_background * rayleigh_correction_factor()

Test Monte Carlo
----------------

Obiettivo
~~~~~~~~~

Verificare la convergenza della stima di :math:`\sigma` in funzione della dimensione
della Region of Interest (ROI).

Metodologia
~~~~~~~~~~~

1. Creare immagini sintetiche con :math:`\sigma` noto
2. Variare la dimensione della ROI: :math:`d \in [5, 10, 15, ..., 200]` pixel
3. Per ogni dimensione, ripetere :math:`N` volte:

   - Estrarre ROI casuale di dimensione :math:`d \times d`
   - Calcolare :math:`\hat{\sigma} = \text{std}(ROI)`
   - Registrare :math:`\hat{\sigma}` e errore relativo

4. Calcolare statistiche: media, SD, range degli errori per ogni dimensione ROI

Convergenza Attesa
~~~~~~~~~~~~~~~~~~

Per una stima non distorta della varianza, l'errore standard diminuisce come:

.. math::

   SE(\hat{\sigma}) \propto \frac{1}{\sqrt{N}}

dove :math:`N = d^2` è il numero di pixel nella ROI.

Quindi aumentando la dimensione ROI da :math:`d_1` a :math:`d_2`, l'errore si riduce
approssimativamente di un fattore :math:`d_2/d_1`.

Risultati Attesi
~~~~~~~~~~~~~~~~

- **ROI piccole** (5×5 - 10×10): Alta variabilità, stime instabili
- **ROI medie** (20×20 - 50×50): Buon compromesso accuratezza/praticità
- **ROI grandi** (>100×100): Massima accuratezza ma poco pratico su anatomia reale

Applicazione Pratica
---------------------

Su Fantoccio MRI
~~~~~~~~~~~~~~~~

1. Caricare immagine DICOM del fantoccio
2. Identificare regione uniforme (background o tessuto omogeneo)
3. Calcolare SD map con kernel 5×5
4. Estrarre :math:`\sigma` dal background usando correzione Rayleigh
5. Validare con misure ROI manuali

Su Serie Cardiaca
~~~~~~~~~~~~~~~~~

1. Caricare serie multi-slice (es. 18 slice LGE)
2. Per ogni slice:

   - Calcolare SD map
   - Estrarre :math:`\sigma_{bkg}` dal background
   - Applicare correzione Rayleigh

3. Visualizzare variazione :math:`\sigma` attraverso le slice
4. Identificare slice con SNR ottimale

Interpretazione Risultati
--------------------------

Signal-to-Noise Ratio (SNR)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   SNR = \frac{\mu_{signal}}{\sigma_{noise}}

Valori tipici:

- SNR < 5: Immagine molto rumorosa
- SNR 10-20: Qualità diagnostica accettabile
- SNR > 30: Alta qualità

Contrast-to-Noise Ratio (CNR)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   CNR = \frac{|\mu_1 - \mu_2|}{\sigma_{noise}}

Misura la capacità di distinguere due tessuti con intensità diverse.

Riferimenti
-----------

1. Constantinides, C. D., Atalar, E., & McVeigh, E. R. (1997).
   *Signal-to-noise measurements in magnitude images from NMR phased arrays*.
   Magnetic Resonance in Medicine, 38(5), 852-857.

2. Henkelman, R. M. (1985).
   *Measurement of signal intensities in the presence of noise in MR images*.
   Medical Physics, 12(2), 232-233.

3. Gudbjartsson, H., & Patz, S. (1995).
   *The Rician distribution of noisy MRI data*.
   Magnetic Resonance in Medicine, 34(6), 910-914.
