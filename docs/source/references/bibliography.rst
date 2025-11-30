Bibliografia
============

Questa pagina contiene i riferimenti scientifici citati nella documentazione.

Articoli Scientifici
--------------------

.. [Constantinides1997] Constantinides, C. D., Atalar, E., & McVeigh, E. R. (1997).
   Signal-to-noise measurements in magnitude images from NMR phased arrays.
   *Magnetic Resonance in Medicine*, 38(5), 852-857.

   Riferimento per la correzione Rayleigh nelle immagini MRI magnitude.
   Fattore di correzione: :math:`\sqrt{2/(4-\pi)} \approx 1.526`

.. [GonzalezWoods] Gonzalez, R. C., & Woods, R. E. (2018).
   *Digital Image Processing* (4th ed.). Pearson.

   Testo di riferimento per teoria del filtraggio, Wiener filter, e
   tecniche di elaborazione immagini.

Dataset Pubblici
----------------

.. [RIDER] Clark, K., et al. (2013).
   The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public
   Information Repository. *Journal of Digital Imaging*, 26(6), 1045-1057.

   RIDER Phantom PET-CT dataset:
   https://wiki.cancerimagingarchive.net/display/Public/RIDER+Phantom+PET-CT

   Utilizzato nell'Esercitazione 2 per filtraggio 3D su CT.

Standard e Specifiche
---------------------

.. [DICOM] NEMA (National Electrical Manufacturers Association).
   *Digital Imaging and Communications in Medicine (DICOM) Standard*.
   https://www.dicomstandard.org/

   Standard per imaging medico. Documenta tag come:

   * (0028,1052): RescaleIntercept
   * (0028,1053): RescaleSlope
   * (0028,1054): RescaleType

.. [HU] Hounsfield Units.
   https://radiopaedia.org/articles/hounsfield-unit

   Scala di densità per TC:

   * Aria: -1000 HU
   * Acqua: 0 HU
   * Osso: +1000 HU

Libri di Testo
--------------

.. [PrinceLinks] Prince, J. L., & Links, J. M. (2006).
   *Medical Imaging Signals and Systems* (2nd ed.). Pearson.

   Imaging biomedico, teoria segnali, e sistemi di acquisizione.

.. [Bushberg] Bushberg, J. T., et al. (2011).
   *The Essential Physics of Medical Imaging* (3rd ed.). Lippincott Williams & Wilkins.

   Fisica dell'imaging medico: MRI, CT, PET, SPECT.

Software e Librerie
-------------------

.. [NumPy] Harris, C. R., et al. (2020).
   Array programming with NumPy. *Nature*, 585(7825), 357-362.

   https://numpy.org/

.. [SciPy] Virtanen, P., et al. (2020).
   SciPy 1.0: fundamental algorithms for scientific computing in Python.
   *Nature Methods*, 17, 261-272.

   https://scipy.org/

.. [PyDICOM] Mason, D., et al. (2011).
   pydicom: An open source DICOM library. *Medical Physics*, 38(6), 3493-3493.

   https://pydicom.github.io/

.. [Matplotlib] Hunter, J. D. (2007).
   Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90-95.

   https://matplotlib.org/

.. [scikit-image] van der Walt, S., et al. (2014).
   scikit-image: image processing in Python. *PeerJ*, 2, e453.

   https://scikit-image.org/

Risorse Online
--------------

NumPy for MATLAB Users
~~~~~~~~~~~~~~~~~~~~~~~

https://numpy.org/doc/stable/user/numpy-for-matlab-users.html

Guida comparativa MATLAB-NumPy, utile per conversione codice.

DICOM Library
~~~~~~~~~~~~~

https://www.dicomlibrary.com/

Esempi di file DICOM per testing.

The Cancer Imaging Archive
~~~~~~~~~~~~~~~~~~~~~~~~~~~

https://www.cancerimagingarchive.net/

Repository pubblico di imaging medico oncologico.

Come Citare
-----------

Se utilizzi questo materiale didattico, cita:

.. code-block:: bibtex

   @misc{bioimmagini2025,
     title={Bioimmagini Positano: Python Implementation},
     author={Bioimmagini Course Team},
     year={2025},
     howpublished={\url{https://github.com/your-repo/bioimmagini_positano}}
   }

Oppure in formato APA:

   Bioimmagini Course Team. (2025). *Bioimmagini Positano: Python Implementation*.
   https://github.com/your-repo/bioimmagini_positano

Note Legali
-----------

Tutti i dataset medici utilizzati sono di dominio pubblico o utilizzati
secondo licenze permissive per scopi educativi.

Il codice del progetto è rilasciato sotto licenza [specificare licenza].

Per informazioni sui diritti d'autore dei singoli dataset, consulta
la documentazione specifica in ogni cartella ``data/``.

Aggiornamenti
-------------

Questa bibliografia viene aggiornata man mano che vengono aggiunte
nuove esercitazioni e riferimenti.

Ultimo aggiornamento: 2025

Prossimi Passi
--------------

* **Risorse esterne**: :doc:`external-resources`
* **Torna alla documentazione**: :doc:`../index`
