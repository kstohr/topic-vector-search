# Troubleshooting

Common setup issues and fixes for the workshop environment.

---

## `docker compose up` fails: port already allocated

You see one of:

```
Bind for 0.0.0.0:11434 failed: port is already allocated
Bind for 0.0.0.0:9201  failed: port is already allocated
```

**Cause.** Another process — usually a previously installed native Ollama or
Elasticsearch — is already listening on that port. Docker cannot bind a port
that is already in use.

### Step 1 — identify what is using the port

**Linux / macOS:**

```bash
lsof -iTCP:11434 -sTCP:LISTEN      # 9201 for Elasticsearch
# If lsof is not installed:
ss -lntp | grep 11434
```

**Windows 11 (PowerShell):**

```powershell
Get-NetTCPConnection -LocalPort 11434 -State Listen |
  Select-Object LocalPort, OwningProcess,
    @{Name="ProcessName"; Expression={(Get-Process -Id $_.OwningProcess).ProcessName}}
```

Or, in `cmd.exe`:

```cmd
netstat -ano | findstr :11434
```

### Step 2 — stop the conflicting process

#### Ollama (port 11434)

**Linux:**

```bash
# If installed as a systemd service (most common on Linux):
sudo systemctl stop ollama
sudo systemctl disable ollama          # optional — prevents auto-start at boot

# If started manually:
sudo pkill -f "ollama serve"
```

**macOS:**

```bash
# Most common: Ollama.app from ollama.com runs in the menu bar.
# Easiest — click the Ollama menu-bar icon and choose Quit.
# Equivalent from terminal:
osascript -e 'quit app "Ollama"'

# Or fall back to a process kill:
pkill -f Ollama

# If installed via Homebrew:
brew services stop ollama
```

**Windows 11 (PowerShell as Administrator):**

```powershell
Get-Process ollama -ErrorAction SilentlyContinue | Stop-Process -Force
# If Ollama installed itself as a Windows service:
Stop-Service Ollama -ErrorAction SilentlyContinue
```

#### Elasticsearch (port 9201)

**Linux:**

```bash
# Package install (apt / dnf / rpm):
sudo systemctl stop elasticsearch

# Manual install:
sudo pkill -f elasticsearch
```

**macOS:**

```bash
# Homebrew install (most common on Mac):
brew services stop elasticsearch-full     # full distribution
brew services stop elasticsearch          # OSS distribution

# Manual install / tarball:
pkill -f elasticsearch
```

**Windows 11 (PowerShell as Administrator):**

```powershell
# Default service name when installed via the official MSI:
Stop-Service elasticsearch-service-x64 -ErrorAction SilentlyContinue
# Older installs may use simply 'elasticsearch':
Stop-Service elasticsearch -ErrorAction SilentlyContinue
# Generic fallback:
Get-Process -Name "elasticsearch*" -ErrorAction SilentlyContinue | Stop-Process -Force
```

### Step 3 — verify the port is free, then retry

**Linux / macOS:**

```bash
lsof -iTCP:11434 -sTCP:LISTEN          # should print nothing
```

**Windows 11 (PowerShell):**

```powershell
Get-NetTCPConnection -LocalPort 11434 -State Listen
# should return nothing or 'No matching MSFT_NetTCPConnection objects found'
```

Then bring the stack up:

```bash
docker compose up -d
```

---

### Alternative: keep your native install, skip the Docker service

If your native Ollama (or Elasticsearch) is already working and you would
rather leave it alone, simply skip the conflicting Docker service. The
workshop code does not care whether `localhost:11434` / `localhost:9201` is
served by Docker or by a native install.

```bash
# Bring up only Elasticsearch from compose; keep using your native Ollama:
docker compose up -d elasticsearch

# Make sure your native Ollama has the model the workshop expects:
ollama list | grep qwen2.5:3b || ollama pull qwen2.5:3b
```

## Preprocessing 

## `elasticsearch.BadRequestError 

Error: 

```
PostDocMissingEmbeddingError
``` 

or a BadRequestError such as this: 
```
elasticsearch.BadRequestError: BadRequestError(400, 'document_parsing_exception', "[1:403] failed to parse: The [dense_vector] field [doc_embedding] in doc [document with id '1c9e469e-b3c6-4f2d-bc23-eea9edf596bf'] has a different number of dimensions [0] thn defined in the mapping [384]"... 
```

If you see an error message like this when running 
uv run -m src.preprocessing, it likely means that either you have not completed
the exercise to build the PreProcessingPipeline.generate_embedding() method or
there is a coding error in your message such that embeddings were not generated. 
You may also want to check that you are storing PostDocuments as json.
BadRequest error can indicate a field mapping error. 


## Elasticsearch Issues 

1. First check that documents are stored on Elasticsearch: 

```
# Running the main block on es_index checks the document count.
uv run python -m src.es_index
``` 

## Deleting documents 

If for some reason you need to delete all documents on the index, run: 

```
from src.es_index import get_es_client, count_documents
es_client = get_es_client()
delete_index(es_client)
```

One thing to note: deleting the index also deletes the mapping. `create_index`  recreates it before indexing new documents

## Sample Post Issues 
If you accidentally made a change that corrupted or deleted `src/sample_posts.json` or
`noise_posts.json` You can either git restore the changes to those files. 

```
git restore sample_posts.json
# OR 
git restore sample_noise.json
```
If that is not possible copy the backup versions in `solutions/`: 

`solutions/sample_posts.json.bkp`
`solutions/sample_noise.json.bkp`

## Pytest 

The ResourceTracker warning at the end is a known Python 3.12 bug in the 
multiprocess package — unrelated to the code. Safe to ignore.

## Reset Workspace - ("nuclear option")

If all else fails... Try this. 

Remove generated artifacts and clear Elasticsearch data before re-running the workshop pipeline:

```bash
uv run python reset.py
```

This works on macOS, Linux, and Windows and will:

- Remove generated files from `output/` (JSON/CSV/HTML + `output/bertopic_model/`)
- Delete the Elasticsearch index (if Elasticsearch is reachable) and any
  documents stored on it. 

If you also want to discard uncommitted `.py` edits, use:

```bash
uv run python reset.py --restore-py
```

You will see a confirmation prompt before files are restored.


