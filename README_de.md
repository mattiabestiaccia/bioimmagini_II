# Bioimmagini Positano - MATLAB-zu-Python-Umwandlungsprojekt

## 🎯 Projektziel

Systematische Umwandlung der MATLAB-Übungen des Kurses "Bioimmagini - Positano" in Python, unter Beibehaltung der funktionalen Äquivalenz und unter Einhaltung moderner Best Practices.

## 📁 Projektstruktur

```
bioimmagini_positano/
├── esercitazioni/
│   ├── esercitazioni_matlab/          # Original-MATLAB-Übungen (Referenz)
│   │   ├── Esercitazione_1_09_03_2022/
│   │   ├── Esercitazione_2_*/
│   │   └── ...
│   └── esercitazioni_python/          # Python-Umwandlungen
│       ├── venv/                      # Gemeinsame virtuelle Umgebung
│       ├── activate.sh                # Schnellaktivierungsskript
│       ├── esercitazione_1/           # ✅ ABGESCHLOSSEN
│       ├── esercitazione_2/           # 🔜 TODO
│       └── ...
├── REBASING_GUIDE.md                  # 📘 Vollständiger Umwandlungsleitfaden
├── .claude/
│   └── project_context.md             # Kontext für KI-Assistenten
└── README.md                          # Diese Datei
```

## 🚀 Schnellstart

### Einrichtung für Cross-Platform (WSL + Windows)

Dieses Repository ist für die Verwendung sowohl unter **WSL** (für Entwicklung mit Cursor/VS Code) als auch unter **Windows** (für Obsidian Vault) konfiguriert:

**WSL-Pfad**:
```bash
/home/brusc/Projects/bioimmagini_positano
```

**Windows-Pfad** (zum Öffnen in Obsidian):
```
\\wsl.localhost\Ubuntu\home\brusc\Projects\bioimmagini_positano
```

**Git-Konfiguration**:
- `.gitattributes` verwaltet automatisch die Zeilenenden (LF)
- `.gitignore` schließt temporäre Obsidian-Dateien aus
- Sie können Dokumente aus beiden Umgebungen ohne Konflikte bearbeiten

### Umgebungseinrichtung

```bash
# Klonen Sie das Repository oder navigieren Sie zum Projekt
cd /path/to/bioimmagini_positano

# Aktivieren Sie die virtuelle Umgebung
cd esercitazioni/esercitazioni_python
source venv/bin/activate

# Überprüfen Sie die Installation
python --version
pip list | grep -E "(numpy|scipy|matplotlib|pydicom)"
```

### Übung 1 ausführen

```bash
cd esercitazione_1/src

# Skript 1: Analyse eines synthetischen Bildes
python calcolo_sd.py

# Skript 2: Analyse eines MRT-Phantoms
python esempio_calcolo_sd.py

# Skript 3: Monte-Carlo-Test
python test_m_sd.py

# Ergebnisse verfügbar unter:
ls ../results/
```

## 📚 Dokumentation

### Für Entwickler/Konvertierer

**Lesen Sie dies, bevor Sie mit einer neuen Umwandlung beginnen**:
- 📘 **[REBASING_GUIDE.md](REBASING_GUIDE.md)** - Vollständiger Leitfaden mit:
  - Standard-Arbeitsablauf
  - Dateiverwaltungsregeln
  - Qualitätscheckliste
  - MATLAB↔Python-Äquivalenzen
  - Namenskonventionen

### Für Studierende/Benutzer

Jede Übung hat eine eigene `README.md` mit:
- Beschreibung der Ziele
- Installationsanweisungen
- Anwendungsbeispiele
- Theorie und Konzepte
- Fehlerbehebung

**Beispiel**: [README Übung 1](esercitazioni/esercitazioni_python/esercitazione_1/README.md)

## ✅ Status der Umwandlungen

### Abgeschlossen

| # | Titel | Datum | Status | Python-Dateien | Dokumentation |
|---|--------|------|--------|-------------|----------------|
| 1 | SD-Berechnung in MRT-Bildern | 09/03/2022 | ✅ | 1221 Zeilen | ✅ Vollständig |

### In Planung

| # | Titel | Datum | Status |
|---|--------|------|--------|
| 2 | TBD | TBD | 🔜 Noch zu katalogisieren |
| 3 | TBD | TBD | 🔜 Noch zu katalogisieren |
| ... | ... | ... | ... |

## 🔧 Technologien

### Python-Stack
- **Python**: 3.12+
- **NumPy**: 2.3+ (Array-Operationen)
- **SciPy**: 1.16+ (wissenschaftliche Verarbeitung)
- **Matplotlib**: 3.10+ (Visualisierung)
- **PyDICOM**: 3.0+ (DICOM-Lesen)
- **Jupyter**: 1.1+ (optionale Notebooks)
- **scikit-image**: 0.25+ (Bildverarbeitung)

### Entwicklungsumgebung
- Gemeinsame virtuelle Umgebung für alle Übungen
- VS Code mit Python-Erweiterungen
- Git für Versionskontrolle

## 📋 Umwandlungsarbeitsablauf

### 1. Analyse
```bash
# MATLAB-Übung erkunden
ls -la esercitazioni/esercitazioni_matlab/Esercitazione_X/

# Dateien katalogisieren
find ... -name "*.m"      # Zu konvertierende Skripte
find ... -name "*.dcm"    # Zu kopierende Daten
find ... -name "*.pdf"    # Zu kopierende Dokumentation
```

### 2. Einrichtung
```bash
# Standardstruktur erstellen
mkdir -p esercitazioni_python/esercitazione_X/{src,data,results,docs}
```

### 3. Umwandlung
- Daten und PDFs kopieren
- MATLAB-Skripte in Python konvertieren
- Hilfsprogramm-Module erstellen
- Dokumentation schreiben

### 4. Validierung
- Alle Skripte testen
- Numerische Äquivalenz überprüfen
- Qualitätscheckliste vervollständigen

**Siehe [REBASING_GUIDE.md](REBASING_GUIDE.md) für vollständige Details**

## 🎓 Übungen - Details

### Übung 1: Berechnung der Standardabweichung in MRT-Bildern

**Didaktische Ziele**:
- Rauschanalyse in MRT-Bildern
- Methoden zur SD-Schätzung (manuelle ROI, SD-Map)
- Rayleigh-Korrektur für Hintergrund
- Statistische Konvergenz (Monte Carlo)

**Python-Skripte**:
1. `calcolo_sd.py` - Analyse eines synthetischen Bildes
2. `esempio_calcolo_sd.py` - Analyse eines MRT-Phantoms
3. `test_m_sd.py` - Monte-Carlo-ROI-Test

**Daten**:
- MRT-Phantom (`phantom.dcm`)
- LGE-Herzserien (18 DICOM)
- Beispielbilder

**[→ Zur vollständigen Dokumentation](esercitazioni/esercitazioni_python/esercitazione_1/README.md)**

## 🤝 Wie man beiträgt

### Eine neue Übung umwandeln

1. **Lesen** Sie [REBASING_GUIDE.md](REBASING_GUIDE.md)
2. **Wählen** Sie eine noch nicht umgewandelte Übung
3. **Folgen** Sie dem Standard-Arbeitsablauf
4. **Validieren** Sie mit der Qualitätscheckliste
5. **Dokumentieren** Sie vollständig
6. **Aktualisieren** Sie diese README

### Qualitätsstandards

Jede Umwandlung muss:
- ✅ ALLE MATLAB-Funktionen replizieren
- ✅ ALLE Datendateien kopieren
- ✅ Eine vollständige README haben
- ✅ Die Qualitätscheckliste bestehen
- ✅ Numerisch äquivalent sein

## 📖 Referenzen

### Kurs
- **Titel**: Bioimmagini
- **Ort**: Positano
- **Dozenten**: [Noch anzugeben]

### Technische Ressourcen
- [NumPy Documentation](https://numpy.org/doc/)
- [SciPy Documentation](https://docs.scipy.org/)
- [PyDICOM Guide](https://pydicom.github.io/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/)

### Bibliographie
Siehe README der einzelnen Übungen für spezifische Referenzen.

## 📄 Lizenz

Lehrmaterial - Nur für Bildungs- und Forschungszwecke.

## 📞 Kontakt

Für Fragen zum Umwandlungsprojekt oder zu den Übungen:
- [Kontakte der Dozenten/Verantwortlichen angeben]

---

**Letzte Änderung**: 2025-11-10
**Projektstatus**: In Bearbeitung (1/X Übungen abgeschlossen)
**Nächste Aktion**: Verbleibende Übungen katalogisieren
