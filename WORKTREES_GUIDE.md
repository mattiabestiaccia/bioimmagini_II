# Guida Git Worktrees - Lavoro Parallelo

Questa guida spiega come lavorare su più esercitazioni contemporaneamente usando Git Worktrees.

## Cos'è un Worktree?

Un **worktree** è una copia di lavoro aggiuntiva del repository, collegata allo stesso `.git` ma su un branch diverso. Permette di:

- Lavorare su più branch contemporaneamente senza fare `git stash` o `git switch`
- Aprire terminali/editor separati per ogni esercitazione
- Testare modifiche in isolamento

## Struttura Attuale

```
~/Projects/
├── bioimmagini_positano/                    # Repo principale (main)
└── bioimmagini_positano_worktrees/          # Worktrees
    ├── es_4/   → branch: feature/es_4
    ├── es_7/   → branch: feature/es_7
    ├── es_9/   → branch: feature/es_9
    └── es_12/  → branch: feature/es_12
```

---

## Comandi Essenziali

### Vedere tutti i worktrees

```bash
git worktree list
```

### Creare un nuovo worktree

```bash
# Prima crea il branch (dalla repo principale)
cd ~/Projects/bioimmagini_positano
git branch feature/es_5

# Poi crea il worktree
git worktree add ../bioimmagini_positano_worktrees/es_5 feature/es_5
```

### Rimuovere un worktree (dopo merge)

```bash
git worktree remove ../bioimmagini_positano_worktrees/es_4
git branch -d feature/es_4  # opzionale: elimina anche il branch
```

---

## Workflow Passo-Passo

### 1. Inizia a Lavorare su un'Esercitazione

```bash
# Apri un nuovo terminale
cd ~/Projects/bioimmagini_positano_worktrees/es_4

# Avvia Claude Code
claude
```

### 2. Lavora Normalmente

Nel worktree puoi fare tutto quello che faresti normalmente:

```bash
# Modifica file
# Esegui test
python -m pytest

# Controlla lo stato
git status
git diff
```

### 3. Commit delle Modifiche

```bash
git add .
git commit -m "feat(es_4): implement cardiac function analysis"
```

### 4. Push del Branch

```bash
# Prima volta: imposta upstream
git push -u origin feature/es_4

# Volte successive
git push
```

### 5. Crea Pull Request

```bash
gh pr create --title "Exercise 4: Cardiac Function" --body "..."
```

### 6. Dopo il Merge

```bash
# Torna alla repo principale
cd ~/Projects/bioimmagini_positano

# Aggiorna main
git pull

# Rimuovi il worktree
git worktree remove ../bioimmagini_positano_worktrees/es_4

# Elimina il branch locale
git branch -d feature/es_4
```

---

## Lavorare in Parallelo (Più Terminali)

### Terminale 1 - Esercitazione 4
```bash
cd ~/Projects/bioimmagini_positano_worktrees/es_4
claude
# Lavora su funzione cardiaca...
```

### Terminale 2 - Esercitazione 7
```bash
cd ~/Projects/bioimmagini_positano_worktrees/es_7
claude
# Lavora su registrazione...
```

### Terminale 3 - Repo Principale
```bash
cd ~/Projects/bioimmagini_positano
# Controlla stato generale, merge PR, ecc.
```

---

## Sincronizzazione tra Worktrees

I worktrees condividono lo stesso `.git`, quindi:

| Cosa | Condiviso? | Note |
|------|------------|------|
| Commit history | Si | Tutti vedono tutti i commit |
| Branch | Si | `git branch` mostra tutti |
| Stash | Si | Attenzione: stash è globale |
| Working directory | No | Ogni worktree ha i suoi file |
| Index (staging) | No | Ogni worktree ha il suo staging |

### Aggiornare un Worktree con Modifiche da Main

```bash
cd ~/Projects/bioimmagini_positano_worktrees/es_4

# Opzione 1: Rebase (preferito per feature branch)
git fetch origin
git rebase origin/main

# Opzione 2: Merge
git fetch origin
git merge origin/main
```

---

## Troubleshooting

### "fatal: 'feature/es_X' is already checked out"

Un branch può essere attivo in un solo worktree alla volta. Controlla:

```bash
git worktree list
```

### Worktree "locked" o corrotto

```bash
# Sblocca
git worktree unlock ../bioimmagini_positano_worktrees/es_4

# Se corrotto, rimuovi forzatamente
git worktree remove --force ../bioimmagini_positano_worktrees/es_4
```

### Vedere su quale branch sei

```bash
git branch --show-current
# oppure
git status
```

---

## Riepilogo Comandi

| Azione | Comando |
|--------|---------|
| Lista worktrees | `git worktree list` |
| Crea worktree | `git worktree add <path> <branch>` |
| Rimuovi worktree | `git worktree remove <path>` |
| Pulizia worktrees orfani | `git worktree prune` |

---

## Stato Esercitazioni

| Worktree | Branch | Esercitazione | Priorità |
|----------|--------|---------------|----------|
| `es_4/` | feature/es_4 | Funzione Cardiaca | Alta (2 py, 0 test) |
| `es_7/` | feature/es_7 | Registrazione | Alta (4 py, 0 test) |
| `es_9/` | feature/es_9 | Mappe Parametriche | Media (4 py, 0 test) |
| `es_12/` | feature/es_12 | Placeholder | Da creare |
