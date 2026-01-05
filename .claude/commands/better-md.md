---
description: Migliora la leggibilità di un documento Markdown per Obsidian
argument-hint: <percorso_file.md>
---

# Better Markdown

Migliora la leggibilità generale di un documento Markdown destinato a Obsidian, correggendo formattazione, paragrafazione e formule matematiche LaTeX.

## Argomenti

- `$ARGUMENTS` - Percorso del file Markdown da migliorare (obbligatorio)

## Istruzioni Operative

### 1. Validazione Input

1. **Verifica che `$ARGUMENTS` contenga un percorso valido**
   - Se vuoto, chiedi all'utente di specificare il file
   - Se il file non esiste, segnala l'errore
   - Accetta percorsi relativi (dalla directory corrente) o assoluti

2. **Leggi il file Markdown** usando il tool Read

### 2. Analisi del Documento

Prima di modificare, analizza:

1. **Struttura generale**:
   - Gerarchie di heading (H1, H2, H3...)
   - Presenza di frontmatter YAML
   - Organizzazione delle sezioni

2. **Problemi di formattazione comuni**:
   - Paragrafi troppo lunghi senza interruzioni
   - Mancanza di linee vuote tra sezioni
   - Liste mal formattate
   - Codice inline non delimitato

3. **Formule matematiche**:
   - Formule inline (tra singoli `$...$`)
   - Formule display/block (tra doppi `$$...$$`)
   - Sintassi LaTeX non valida o incompleta
   - Formule non racchiuse in delimitatori

### 3. Correzioni da Applicare

#### 3.1 Paragrafazione e Leggibilità

- **Spaziatura**: Assicura una linea vuota tra paragrafi, prima/dopo heading, prima/dopo blocchi di codice
- **Heading**: Verifica gerarchia corretta (non saltare livelli)
- **Liste**:
  - Linea vuota prima dell'inizio di una lista
  - Indentazione consistente (2 spazi per sotto-liste in Obsidian)
- **Linee lunghe**: Considera di spezzare frasi molto lunghe per migliorare la leggibilità

#### 3.2 Formule LaTeX per Obsidian

**Sintassi Obsidian:**
- Inline: `$formula$` (senza spazi dopo/prima dei delimitatori)
- Block/Display: `$$formula$$` (su righe separate)

**Correzioni tipiche:**

| Problema | Soluzione |
|----------|-----------|
| `$ x^2 $` (spazi interni) | `$x^2$` |
| Formula senza delimitatori | Aggiungere `$...$` o `$$...$$` |
| `\[...\]` o `\(...\)` | Convertire a `$$...$$` o `$...$` |
| Doppi backslash errati | `\\` solo per newline in ambienti |
| Parentesi non bilanciate | Correggere `\left(` con `\right)` |
| Frazioni malformate | `\frac{num}{den}` |
| Pedici/apici multipli | `x_{ij}` non `x_ij` |
| Simboli comuni | `\alpha`, `\beta`, `\sum`, `\int`, `\partial` |

**Ambienti matematici supportati in Obsidian:**
```latex
$$
\begin{aligned}
  x &= a + b \\
  y &= c + d
\end{aligned}
$$
```

#### 3.3 Elementi Specifici Obsidian

Preserva e correggi se necessario:

- **Callout**: `> [!note]`, `> [!warning]`, `> [!tip]`, ecc.
- **Wikilinks**: `[[NomeNota]]` o `[[NomeNota|Testo visualizzato]]`
- **Tag**: `#tag` (devono iniziare con lettera)
- **Embed**: `![[immagine.png]]` o `![[nota]]`
- **Highlight**: `==testo evidenziato==`
- **Task**: `- [ ]` e `- [x]`
- **Footnotes**: `[^1]` e `[^1]: definizione`
- **Comments**: `%%commento%%`

### 4. Processo di Modifica

1. **Crea backup mentale** - Tieni traccia delle modifiche significative
2. **Applica correzioni** usando il tool Edit per modifiche incrementali
3. **Verifica integrità** - Assicurati che le formule siano valide
4. **Riporta le modifiche** - Elenca brevemente cosa è stato cambiato

### 5. Output Finale

Al termine, mostra:

```
## Modifiche Applicate

### Formattazione
- [lista modifiche paragrafazione/struttura]

### Formule LaTeX
- [lista formule corrette con before/after se significativo]

### Elementi Obsidian
- [eventuali correzioni a callout, link, ecc.]

---
File aggiornato: `percorso/file.md`
```

## Esempi di Correzione Formule

**Prima:**
```markdown
La formula dell'energia è E = mc^2 dove m è la massa.

La derivata parziale df/dx si calcola come segue:

\[ \frac{\partial f}{\partial x} = 2x \]
```

**Dopo:**
```markdown
La formula dell'energia è $E = mc^2$ dove $m$ è la massa.

La derivata parziale $\frac{\partial f}{\partial x}$ si calcola come segue:

$$
\frac{\partial f}{\partial x} = 2x
$$
```

## Note Importanti

- **Non alterare il contenuto semantico** - Solo formattazione e correzione sintassi
- **Preserva lo stile dell'autore** - Non riscrivere frasi, solo correggere formattazione
- **Attenzione ai blocchi di codice** - Non modificare contenuto dentro ``` o `
- **Rispetta il frontmatter** - Non modificare YAML header se presente
- **Formule ambigue** - Se non è chiaro se qualcosa è una formula, chiedi conferma

## Checklist Finale

- [ ] Tutti i heading hanno spaziatura corretta
- [ ] Paragrafi separati da linee vuote
- [ ] Formule inline in `$...$` senza spazi interni
- [ ] Formule block in `$$...$$` su righe separate
- [ ] Liste con indentazione consistente
- [ ] Callout Obsidian ben formattati
- [ ] Nessuna modifica al contenuto semantico
