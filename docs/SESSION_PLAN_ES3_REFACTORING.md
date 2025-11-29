# Piano Refactoring Esercitazione 3 - K-means Segmentation

**Data Analisi**: 28 Novembre 2024
**Agente Utilizzato**: python-pro (dal plugin python-development)
**Stato**: Analisi completata, refactoring da applicare

---

## Contesto

L'Esercitazione 3 (`es_3__23_03_2022_clustering`) implementa segmentazione cardiaca MRI usando K-means clustering. Il codice è funzionale ma necessita di modernizzazione secondo Python 3.12+ best practices.

## File da Modificare

```
esercitazioni/esercitazioni_python/es_3__23_03_2022_clustering/
├── src/
│   ├── __init__.py
│   ├── utils.py                 # ⚠️ Priorità ALTA
│   ├── kmeans_segmentation.py   # ⚠️ Priorità ALTA
│   ├── optimize_kmeans.py       # ⚠️ Priorità MEDIA
│   └── plot_time_curves.py      # ⚠️ Priorità BASSA
└── tests/                        # ❌ VUOTO - da creare
```

---

## Top 5 Miglioramenti Prioritizzati

### 1. **TYPE SAFETY COMPLETA** (Priorità: CRITICA)

**Problema**: Type hints incompleti, `np.ndarray` troppo generico

**Soluzione**:
```python
from numpy.typing import NDArray
from typing import Protocol, TypedDict
from pydantic import BaseModel, Field

# Type aliases
ImageStack = NDArray[np.float32]  # (H, W, T)
TriggerTimes = NDArray[np.float32]  # (T,)
BinaryMask = NDArray[np.bool_]  # (H, W)

# Configuration con Pydantic
class PerfusionLoadConfig(BaseModel):
    dicom_dir: Path
    n_frames: int | None = None
    validate_metadata: bool = True
```

**File**: `utils.py`, `kmeans_segmentation.py`

---

### 2. **CUSTOM EXCEPTIONS + ERROR HANDLING** (Priorità: CRITICA)

**Problema**: Try-catch generico, nessun custom exception, print invece di logging

**Soluzione**:
```python
# Exception hierarchy
class CardiacSegmentationError(Exception): pass
class DataLoadError(CardiacSegmentationError): pass
class DicomReadError(DataLoadError): pass
class ValidationError(CardiacSegmentationError): pass
class ShapeMismatchError(ValidationError): pass

# Logging strutturato
import logging
logger = logging.getLogger(__name__)

# Validation helpers
def validate_image_stack(images: ImageStack, *, min_frames: int = 1):
    if images.ndim != 3:
        raise ValidationError(f"Expected 3D array, got {images.ndim}D")
    if np.isnan(images).any():
        raise ValidationError("Image stack contains NaN values")
```

**File**: `utils.py` (nuovo file `exceptions.py`), tutti gli script

---

### 3. **DATACLASSES + ENUMS + CONSTANTS** (Priorità: ALTA)

**Problema**: Magic numbers, string literals, nessun Enum

**Soluzione**:
```python
from dataclasses import dataclass, field
from enum import Enum

# Enums per type safety
class TissueType(Enum):
    BACKGROUND = "background"
    RIGHT_VENTRICLE = "rv"
    LEFT_VENTRICLE = "lv"
    MYOCARDIUM = "myo"

    @property
    def color_rgb(self) -> tuple[float, float, float]:
        return {
            self.BACKGROUND: (0.5, 0.5, 0.5),
            self.RIGHT_VENTRICLE: (0.0, 0.0, 1.0),
            self.LEFT_VENTRICLE: (1.0, 0.0, 0.0),
            self.MYOCARDIUM: (0.0, 1.0, 0.0)
        }[self]

# Constants
class ImagingConstants:
    DEFAULT_PEAK_FRAME: Final[int] = 12
    MIN_TEMPORAL_FRAMES: Final[int] = 10
    N_CARDIAC_CLUSTERS: Final[int] = 4

# Config dataclass
@dataclass(frozen=True)
class KMeansConfig:
    n_clusters: int = ImagingConstants.N_CARDIAC_CLUSTERS
    metric: str = "euclidean"
    random_state: int = 42
```

**File**: Nuovo file `config.py`, `utils.py`, `kmeans_segmentation.py`

---

### 4. **PERFORMANCE OPTIMIZATION** (Priorità: ALTA)

**Problema**: I/O sequenziale DICOM, nessun caching, normalizzazione inefficiente

**Soluzione**:
```python
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, cached_property

# Parallel DICOM loading
def load_perfusion_parallel(dicom_files: list[Path], max_workers: int = 8):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_single_dicom, path, i): i
                   for i, path in enumerate(dicom_files)}
        # ... parallel loading

# Vectorized normalization
def normalize_time_curves_vectorized(curves: NDArray[np.float32]):
    means = curves.mean(axis=1, keepdims=True)
    stds = curves.std(axis=1, keepdims=True)
    np.maximum(stds, 1e-8, out=stds)  # in-place
    normalized = np.empty_like(curves)
    np.subtract(curves, means, out=normalized)
    np.divide(normalized, stds, out=normalized)
    return normalized

# Caching
class SegmentationPipeline:
    @cached_property
    def normalized_curves(self) -> NDArray[np.float32]:
        return normalize_time_curves_vectorized(self.flattened_curves)
```

**File**: `utils.py`, `kmeans_segmentation.py`

---

### 5. **COMPREHENSIVE TEST SUITE** (Priorità: CRITICA)

**Problema**: Directory `tests/` VUOTA, zero test coverage

**Soluzione**:
```python
# tests/conftest.py - Fixtures
@pytest.fixture
def sample_image_stack() -> NDArray[np.float32]:
    # Generate synthetic perfusion data
    ...

# tests/test_utils.py - Unit tests
class TestDiceCoefficient:
    def test_identical_masks(self):
        mask = np.random.rand(100, 100) > 0.5
        dice = dice_coefficient(mask, mask)
        assert dice == pytest.approx(1.0)

    def test_no_overlap(self):
        # ...

# tests/test_kmeans.py - Integration tests
class TestKMeansSegmentation:
    def test_reproducibility(self, sample_image_stack):
        # ...
```

**File**: Nuovo `tests/conftest.py`, `tests/test_utils.py`, `tests/test_kmeans.py`

**Target**: >90% coverage

---

## Piano di Implementazione

### **Fase 1: Setup Infrastruttura** (30 min)
1. Creare branch Git: `git checkout -b refactor/es3-modernization`
2. Creare file strutturali:
   - `src/exceptions.py` (custom exceptions)
   - `src/config.py` (dataclasses, enums, constants)
   - `tests/conftest.py` (pytest fixtures)
   - `pyproject.toml` (configurazione moderna)
3. Installare dipendenze dev: `pip install pytest pytest-cov hypothesis pydantic ruff mypy`

### **Fase 2: Type Safety** (1h)
1. Aggiungere type aliases in `src/types.py`
2. Aggiornare signatures funzioni in `utils.py`
3. Aggiornare signatures in `kmeans_segmentation.py`
4. Validare con `mypy src/`

### **Fase 3: Error Handling** (45 min)
1. Implementare exception hierarchy in `src/exceptions.py`
2. Aggiornare `utils.py` con custom exceptions
3. Aggiungere logging in tutti gli script
4. Testare error paths

### **Fase 4: Dataclasses & Enums** (45 min)
1. Implementare `TissueType`, `DistanceMetric` enums
2. Implementare `KMeansConfig`, `PostProcessConfig` dataclasses
3. Sostituire magic numbers con constants
4. Refactoring chiamate funzioni

### **Fase 5: Performance** (1h)
1. Implementare parallel DICOM loading
2. Ottimizzare normalizzazione (vectorization)
3. Aggiungere caching con `@lru_cache`
4. Benchmark before/after

### **Fase 6: Testing** (2h)
1. Scrivere fixtures in `conftest.py`
2. Unit tests per `utils.py` (>90% coverage)
3. Integration tests per pipeline completa
4. Property-based tests con Hypothesis
5. Performance benchmarks

### **Fase 7: Verifica** (30 min)
1. `pytest tests/ --cov=src --cov-report=html`
2. `mypy src/`
3. `ruff check src/`
4. Eseguire script originali per regression test
5. Review differenze con `git diff main`

---

## Comandi per Domani

### **1. Setup Sessione**
```bash
cd /home/brusc/Projects/bioimmagini_positano
git checkout -b refactor/es3-modernization

# Attiva venv
source esercitazioni/esercitazioni_python/venv/bin/activate

# Installa dev dependencies
pip install pytest pytest-cov pytest-benchmark hypothesis pydantic ruff mypy
```

### **2. Usa Python-Pro Plugin**
```
# In Claude Code chat
Leggi il file docs/SESSION_PLAN_ES3_REFACTORING.md e applica i miglioramenti
secondo il piano. Inizia dalla Fase 1 e procedi sequenzialmente.

Usa l'agente python-pro per assistenza durante il refactoring.
```

### **3. Verifica Progressi**
```bash
# Coverage
pytest tests/ --cov=src --cov-report=term-missing

# Type checking
mypy src/

# Linting
ruff check src/

# Performance benchmark
pytest tests/test_performance.py --benchmark-only
```

### **4. Review Finale**
```bash
# Vedi modifiche
git diff main

# Commit incrementali
git add src/exceptions.py
git commit -m "Add custom exception hierarchy"

git add src/config.py
git commit -m "Add dataclasses, enums, constants"

# Etc...
```

---

## File di Riferimento

Il report completo dell'analisi python-pro è disponibile nella conversazione attuale.

**Percorso conversazione**: `~/.claude/projects/-home-brusc-Projects-bioimmagini-positano/`

**Per recuperare il report domani**:
```bash
# In Claude Code
Mostrami il report completo dell'analisi python-pro sull'esercitazione 3
dalla sessione del 28 Novembre 2024.
```

---

## Note Importanti

⚠️ **BACKUP**: Prima di iniziare, crea backup:
```bash
cp -r esercitazioni/esercitazioni_python/es_3__23_03_2022_clustering \
      esercitazioni/esercitazioni_python/es_3_BACKUP_20241128
```

⚠️ **TESTING**: Esegui gli script originali prima del refactoring per avere baseline:
```bash
cd esercitazioni/esercitazioni_python/es_3__23_03_2022_clustering
python src/kmeans_segmentation.py
python src/optimize_kmeans.py
# Salva output in results/baseline/
```

⚠️ **INCREMENTALE**: Committa dopo ogni fase per poter tornare indietro se necessario.

---

## Checklist Completamento

- [ ] Fase 1: Setup infrastruttura
- [ ] Fase 2: Type safety completa
- [ ] Fase 3: Error handling robusto
- [ ] Fase 4: Dataclasses & Enums
- [ ] Fase 5: Performance optimization
- [ ] Fase 6: Test suite (>90% coverage)
- [ ] Fase 7: Verifica e review
- [ ] Merge in main branch

---

## Tempo Stimato Totale: **6-7 ore**

Suddiviso in sessioni:
- **Sessione 1** (3h): Fasi 1-4 (infrastruttura + architettura)
- **Sessione 2** (2h): Fase 5 (performance)
- **Sessione 3** (2h): Fasi 6-7 (testing + verifica)

---

**Prossimo Step**: Aprire Claude Code domani e dire:
```
Leggi docs/SESSION_PLAN_ES3_REFACTORING.md e iniziamo il refactoring dalla Fase 1
```
