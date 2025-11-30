# Bioimmagini Positano - Project Context

## Project Type
Educational/Research - Medical Image Processing

## Primary Language
Python (converting from MATLAB)

## Project Goal
Systematic rebasing of MATLAB medical imaging exercises to Python while maintaining functional equivalence and following modern best practices.

## Key Directories

- `esercitazioni/esercitazioni_matlab/` - Original MATLAB exercises (READ-ONLY reference)
- `esercitazioni/esercitazioni_python/` - Python conversions (ACTIVE DEVELOPMENT)
- `REBASING_GUIDE.md` - Complete conversion guidelines

## Critical Rules

### File Management
1. **EVERY file** from MATLAB must have Python equivalent
2. **Data files** (DICOM, images, PDFs) → COPY to `data/`
3. **Code files** (.m) → CONVERT to `.py` in `src/`
4. **System files** (__MACOSX, .DS_Store) → IGNORE

### Standard Structure
Every `es_{n°esercitazione}__{date}_{argument}/` must have:
```
├── src/          # Python code
├── data/         # Data and resources
├── docs/         # Documentation (PDFs)
├── results/      # Generated outputs
├── README.md     # Complete documentation
└── .gitignore    # Git ignore rules
```

### Quality Standards
- Type hints on all functions
- NumPy-style docstrings
- Numerical equivalence with MATLAB
- Complete README with theory section

## Completed Work

### ✅ Esercitazione 1 (MRI Noise Analysis)
- 1221 lines of Python code
- All MATLAB scripts converted
- All data files copied
- Complete documentation
- Tested and validated

## Active Tasks

Refer to `REBASING_GUIDE.md` and `TODO.md` for:
- Conversion workflow
- Quality checklist
- MATLAB↔Python equivalences
- Naming conventions

## Python Environment

- Shared venv: `esercitazioni/esercitazioni_python/venv/`
- Python 3.12+
- Key libraries: numpy, scipy, matplotlib, pydicom

## When Starting New Exercise

1. Read `REBASING_GUIDE.md`
2. Analyze MATLAB files
3. Create standard structure
4. Copy data files FIRST
5. Convert code
6. Write README
7. Validate against checklist
