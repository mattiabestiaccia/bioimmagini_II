# 🚀 Quick Start - Refactoring Esercitazione 3

**Data sessione precedente**: 28 Novembre 2024  
**Plugin usati**: `python-development`, `code-documentation`

---

## Come Riprendere Domani

### Opzione A: Automatica (consigliata)

```bash
cd /home/brusc/Projects/bioimmagini_positano
./setup_refactoring_es3.sh
```

Poi in **Claude Code** scrivi:
```
Leggi docs/SESSION_PLAN_ES3_REFACTORING.md e iniziamo il refactoring dalla Fase 1
```

---

### Opzione B: Manuale

1. **Apri progetto**:
   ```bash
   cd /home/brusc/Projects/bioimmagini_positano
   ```

2. **Leggi piano sessione**:
   ```bash
   cat docs/SESSION_PLAN_ES3_REFACTORING.md
   ```

3. **In Claude Code**, scrivi:
   ```
   Applica i miglioramenti python-pro all'esercitazione 3 secondo il piano
   in docs/SESSION_PLAN_ES3_REFACTORING.md. Inizia dalla Fase 1.
   ```

---

## File Importanti

| File | Descrizione |
|------|-------------|
| `docs/SESSION_PLAN_ES3_REFACTORING.md` | Piano dettagliato con tutte le fasi |
| `docs/PYTHON_PRO_REPORT_ES3.md` | Reference al report python-pro |
| `setup_refactoring_es3.sh` | Script setup automatico |
| `esercitazioni/.../es_3.../src/` | Codice da refactorare |

---

## Checklist Inizio Sessione

- [ ] Letto `SESSION_PLAN_ES3_REFACTORING.md`
- [ ] Eseguito `setup_refactoring_es3.sh`
- [ ] Venv attivato
- [ ] Branch `refactor/es3-modernization` creato
- [ ] Backup creato in `es_3_BACKUP_20241128`
- [ ] Dipendenze dev installate (pytest, mypy, ruff, etc.)

---

## Fasi di Lavoro

1. **Setup Infrastruttura** (30 min) - Crea file base
2. **Type Safety** (1h) - Type hints + Pydantic
3. **Error Handling** (45 min) - Custom exceptions + logging
4. **Dataclasses & Enums** (45 min) - Modern Python patterns
5. **Performance** (1h) - Parallel I/O + caching
6. **Testing** (2h) - Test suite completa >90% coverage
7. **Verifica** (30 min) - Review + test

**Tempo totale**: 6-7 ore

---

## Plugin Disponibili

✅ `code-documentation` - docs-architect, tutorial-engineer  
✅ `python-development` - python-pro, django-pro, fastapi-pro  
✅ `unit-testing` - test-automator, tdd-orchestrator

Usa python-pro per assistenza durante refactoring.

---

## Prompt Suggeriti per Domani

### Per iniziare:
```
Leggi docs/SESSION_PLAN_ES3_REFACTORING.md. Siamo pronti per iniziare
il refactoring dell'esercitazione 3. Partiamo dalla Fase 1.
```

### Durante il lavoro:
```
Usa l'agente python-pro per implementare [Fase X] del piano di refactoring.
Segui le specifiche in SESSION_PLAN_ES3_REFACTORING.md.
```

### Per verificare:
```
Verifica che le modifiche alla Fase X siano conformi al piano.
Esegui test e type checking.
```

---

## Comandi Utili

```bash
# Test coverage
pytest tests/ --cov=src --cov-report=html

# Type checking
mypy src/

# Linting
ruff check src/

# Vedi modifiche
git diff main

# Commit incrementale
git add src/exceptions.py
git commit -m "feat: add custom exception hierarchy"
```

---

## Note Finali

⚠️ **Lavora incrementalmente** - committa dopo ogni fase  
⚠️ **Testa sempre** - esegui script originali come regression test  
⚠️ **Usa python-pro** - non sei solo, usa l'agente per assistenza

**Buon lavoro! 🚀**
