# Convert MATLAB Exercise to Python

Guided workflow for converting a MATLAB exercise to Python following project standards.

## Instructions

You are helping convert a MATLAB medical imaging exercise to Python. Follow this systematic workflow:

### Phase 0: Pre-check (MANDATORY)

**Before doing ANY conversion work, you MUST check existing progress:**

1. **Identify target directory** - Determine the Python exercise path:
   - Pattern: `esercitazioni/esercitazioni_python/es_{N}__*` or `es_{N}_*`
   - Use Glob to find matching directories

2. **If directory exists, analyze progress:**
   - Check for `src/*.py` files (converted code)
   - Check for `data/` directory (migrated data)
   - Check for `README.md` (documentation)
   - Check for `tests/` (validation tests)
   - Check for `pyproject.toml` (package config)

3. **Report status to user:**
   ```
   ## Conversion Status for Exercise N

   | Component | Status | Details |
   |-----------|--------|---------|
   | Directory | ✅/❌ | path if exists |
   | Python files | ✅/⚠️/❌ | X of Y .m files converted |
   | Data files | ✅/⚠️/❌ | X of Y files migrated |
   | Documentation | ✅/❌ | README.md present/missing |
   | Tests | ✅/❌ | test files present/missing |
   | Config | ✅/❌ | pyproject.toml present/missing |
   ```

4. **Based on status:**
   - **Complete (all ✅)**: Inform user "Conversion already complete. Do you want to review or redo any part?"
   - **Partial (some ⚠️)**: Inform user "Partial conversion found. I will complete the remaining work."
   - **Not started (all ❌)**: Inform user "No existing conversion found. Starting fresh."

5. **Wait for user confirmation** before proceeding with any conversion work.

---

### Phase 1: Analysis

1. **Identify the MATLAB exercise** - Ask the user which exercise to convert if not specified
2. **Explore MATLAB source** in `esercitazioni/esercitazioni_matlab/`
3. **Catalog all files**:
   - `.m` files (code to convert)
   - Data files (DICOM, images, MAT files to copy)
   - PDFs (documentation to preserve)
   - Ignore: `__MACOSX/`, `.DS_Store`

### Phase 2: Structure Creation

Create the standard Python exercise structure:

```
esercitazioni/esercitazioni_python/es_{N}__{date}_{topic}/
├── src/           # Python modules
├── data/          # Data files (copied from MATLAB)
├── docs/          # PDFs and reference materials
├── results/       # Generated outputs (gitignored)
├── tests/         # Validation tests
├── README.md      # Complete documentation
├── pyproject.toml # Package configuration
└── .gitignore     # Standard ignores
```

### Phase 3: Data Migration

1. Copy ALL data files to `data/`
2. Copy PDFs to `docs/`
3. Preserve original filenames
4. Update paths in code to use relative `data/` references

### Phase 4: Code Conversion

For each `.m` file:

1. **Analyze** MATLAB code structure and dependencies
2. **Map** MATLAB functions to Python/NumPy equivalents:
   - `imread` → `plt.imread` or `cv2.imread`
   - `dicomread` → `pydicom.dcmread`
   - Matrix indexing: 1-based → 0-based
   - `.*` `.^` `./` → `*` `**` `/` (element-wise by default in NumPy)
3. **Convert** with:
   - Type hints on all functions
   - NumPy-style docstrings
   - Proper error handling
4. **Validate** numerical equivalence

### Phase 5: Documentation

Create `README.md` with:

```markdown
# Exercise N: [Title]

## Obiettivo
[What the exercise teaches]

## Teoria
[Mathematical/theoretical background - IMPORTANT]

## Struttura
[Directory layout]

## Utilizzo
[How to run]

## Risultati Attesi
[Expected outputs with images if applicable]

## Note sulla Conversione
[MATLAB→Python differences]
```

### Phase 6: Validation

Use the `matlab-validator` agent to:
- Compare numerical outputs
- Verify image processing results
- Check edge cases

### Quality Checklist

Before marking complete:

- [ ] All `.m` files have `.py` equivalents
- [ ] All data files copied to `data/`
- [ ] Type hints on every function
- [ ] Docstrings on every public function
- [ ] README.md with theory section
- [ ] Tests pass with numerical equivalence
- [ ] pyproject.toml configured
- [ ] .gitignore includes results/

## Tools to Use

- **Glob/Grep**: Explore MATLAB source
- **Read**: Analyze MATLAB code
- **Write/Edit**: Create Python code
- **matlab-validator agent**: Validate conversion
- **python-pro agent**: Review Python quality
