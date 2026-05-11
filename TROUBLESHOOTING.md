# Troubleshooting


## Setup Issues

Common setup issues and fixes for the workshop environment.

---

### `docker compose up` fails: port already allocated

You see one of:

```
Bind for 0.0.0.0:11434 failed: port is already allocated
Bind for 0.0.0.0:9201  failed: port is already allocated
```

**Cause.** Another process — usually a previously installed native Ollama or
Elasticsearch — is already listening on that port. Docker cannot bind a port
that is already in use.

#### Step 1 — identify what is using the port

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

#### Step 2 — stop the conflicting process

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

#### Step 3 — verify the port is free, then retry

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

#### Alternative: keep your native install, skip the Docker service

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

## Pytest 

Running tests can help troubleshoot issues and is always a good first step.  The
tests are designed to pass if the implementation of exercise code is correct. 

```
 uv run -m pytest
 ```

If you want to test that non-exercise code is working as expected, you can skip
tests marked "exercise" by running: 
```
uv run -m  pytest -m 'not exercise'
```

Note: The ResourceTracker warning at the end is a known Python 3.12 bug in the 
multiprocess package — unrelated to the code. Safe to ignore.

## Demo App 

### Restarting the demo app
Stop the running app
```
# Control-C in the running terminal, or in a different terminal: 
pkill -f "streamlit run app.py" 
```
Then restart the app: 
```
uv run streamlit run app.py # the app preloads sample posts on startup
```

### Troubleshooting search errors in the Demo App
1. If you try to search in the search bar using either of the semantic search
methods in the left App Controls prior to completing the exercises in 
[notebooks/02_embeddings.ipynb](notebooks/02_embeddings.ipynb) you will see a 
`MissingEmbeddingsError`. This is expected. It means that you have
not preprocessed the sample posts (`sample_posts.json`) and they are stored on
the index without any embeddings. Semantic search depends on embeddings to
compute cosine similarity. 

`Semantic · in-memory` (InMemorySemanticSearch) OR
`Semantic · Elasticsearch`(SemanticSearch) 


3. If you have setup Elasticsearch and you see an error searching with `Semantic · Elasticsearch`(SemanticSearch) option: 
  - Check that Elasticsearch is running. (See: [## Elasticsearch
    Issues](#elasticsearch-issues) below.) If it is not running or you did not set
    it up, this search option will not
work. Use InMemorySemanticSearch, which uses the same underlying scoring but
does not require Elasticsearch instead. 

  - If you see `NoSearchIndexFound` or `EmptySearchIndexError`, that means
    that either the index has not been created (or was deleted by running
    `src/reset.py`) or that no documents are stored on the index. Try the
    following: 
    
    - If you have NOT completed the exercises: 
       - Restart the app. The sample posts in `sample_posts.json` will be loaded
         by default if `output/processed_posts.json` is not available. You should now see a
         `MissingEmbeddingsError` until the exercises are successfully created
         and you have run ` uv run - src.preprocess`

    - If you have completed the exercises: 
        - Try running ` uv run - src.preprocess`. This will rerun the
          preprocessing pipeline and store the posts with embeddings on
          Elasticsearch. The `Semantic - Elasticsearch` option should now work. 

  - If you see or `MissingEmbeddingsError` and you have completed the coding exercises
    in `src/preprocessing.py` something went wrong. Try: 
      - Check that `processed_posts.json` is stored in the `output/` directory. 
        - If not: 
              - try running `uv -m src.preprocess` again and check the log
              output in your terminal. See [## Preprocessing]()
      - Clear the cache. That will refresh data. If will try to load
        `processed_posts.json` and if that does not exist `sample_posts.json)
      - Restart the app (See: [### Restarting the demo app](#restarting-the-demo-app))
  
  - For other issues, try running pytest or see [## Elasticsearch Issues](#elasticsearch-issues) below. 

3. For other issues, verify that the code in src/search.py has not changed in
   some way. You can copy the solutions/search.py for a working seach example.
   This module does not contain exercise code. 

## Preprocessing 

## HuggingFace model loading issues. 

This workshop uses pytorch for model serialization and loading. 

If someone's HuggingFace cache has a model stored as safetensors but the code 
expects pytorch (or vice versa), it can cause load errors.

If you have installed one of the models we are using previously and are
experiencing errors, you may need to remove cached pre-trained models: 

```
rm -rf ~/.cache/huggingface/hub/models--Salesforce--blip-image-captioning-base
```

### `PostDocMissingEmbeddingError` or `elasticsearch.BadRequestError 

Error: 

```
PostDocMissingEmbeddingError
``` 

or a BadRequestError such as this: 
```
elasticsearch.BadRequestError: BadRequestError(400, 'document_parsing_exception', "[1:403] failed to parse: The [dense_vector] field [doc_embedding] in doc [document with id '1c9e469e-b3c6-4f2d-bc23-eea9edf596bf'] has a different number of dimensions [0] thn defined in the mapping [384]"... 
```

A BadRequest error can indicate a field mapping error. 

If you see an error message like this when running 
uv run -m src.preprocessing, it likely means that either you have not completed
the exercise to build the PreProcessingPipeline.generate_embedding() method or
there is a coding error in your implementation of the method such that
embeddings were not generated. 
You may also want to check that you are storing PostDocuments as json and that
the field mappings match the PostDocument model. 


## Elasticsearch Issues 


### ConnectionRefusedError: [Errno 61] Connection refused 

Calls to Elasticsearch will automatically retry 3 times. If you see this error you will
likely that Elasticsearch is unreachable at the configured url:
http://localhost:9201/. The workshop is designed to fallback to using the
locally stored process_posts.json for most tasks. However, this means that the
Elasticsearch in the Demo App will not work. 


Error: 
```
INFO  GET http://localhost:9201/ [status:N/A duration:0.000s]
WARNING  Node <Urllib3HttpNode(http://localhost:9201)> has failed for 4 times in a row, putting on 8 second timeout
INFO  Elasticsearch not available — will use processed_posts.json fallback.
``` 

If you setup Elasticsearch and were expecting it to work, try: 

1. Run  `docker ompose up`
2. Check that another instance of Elasticsearch is not running on the same port.
   This may cause connection errors. 
3. Try running a curl command from your terminal: 

  `curl -sS http://localhost:9201/` 
   
  If Elasticsearch is running, you should get JSON back
  with fields like name, cluster_name, and version.


### Elastic search is connecting but no search results are returned.

1. First check that documents are stored on Elasticsearch: 

```
# Running the main block on es_index checks the document count.
uv run python -m src.es_index
``` 

2. Try running `uv run -m src.preprocess`. This is the source code that stores
   posts in Elasticsearch. Watch the logs for errors and
   warnings. If you see the error below, saving to Elasticsearch, it indicates 
   there was an issue generating embeddings. Embeddings are a required field in the Elasticsearch
   index. Documents that are missing this field will fail to store. See [##
   Preprocessing](#preprocessing)  above for tips on handling this error. 

  ```
    File "/Users/kas/dev/pycon_2026/topic-vector-search/src/preprocess.py", line 228, in save_to_elasticsearch
    raise PostDocMissingEmbeddingError(
  PostDocMissingEmbeddingError: One or more PostDocuments were missing
  doc_embedding vectors. Check your generate_embeddings() implementation.
  ```


## Topic Model Pipeline Issues 

If your model is erroring this could be the result of a few things. 

1. Verify that `output/processed_posts.json` exists and that each post contains
   a document embedding. Run `uv run -m src.preproces` to generate embeddings
   and store the processed documents. 

   ```
    FileNotFoundError: [Errno 2] No such file or directory: 'output/processed_posts.json'

    The above exception was the direct cause of the following exception:

    Traceback (most recent call last):
      File "/Users/kas/dev/pycon_2026/topic-vector-search/src/topic_model.py", line 411, in <module>
        TopicModeler().run()
      File "/Users/kas/dev/pycon_2026/topic-vector-search/src/topic_model.py", line 403, in run
        self.retrieve_post_documents()
      File "/Users/kas/dev/pycon_2026/topic-vector-search/src/topic_model.py", line 116, in retrieve_post_documents
        raise ProccessedPostsNotFoundError("Make sure you have run preprocessing.py") from error
    ProccessedPostsNotFoundError: Make sure you have run preprocessing.py
    ```

2. Ensure that you are running `src/topic_model.py` within the virtual
   environment. 
   
   `uv run -m src.topic_model` 
   
   If `processed_posts.json` file exists, but you get an error like
   the one below, chances are you may not be running the code inside the virtual
   environment. 

3. Check the logs usually the traceback will let you know where in the pipeline
   the model failed. For example, the `NoTopicsFoundError: No topics found. Try
   adjusting BERTopic parameters.' indicates that `HDBSCAN` model was unable to
   discover any clusters given the data and the parameters set on it. See below.


3. Check the parameters to the model. Review the parameters passed to
   `solutions/topic_model.py`
  - Setting  - Setting `HDBHDBSCAN.min_cluster_size` too high, can result in no clusters
   forming 
  - Similarly setting the `BERTopic.min_topic_size` too high, can result in no clusters
   forming 

## AI Labeling Issues

The workshop is setup to fallback to using keywords to label topics, if OLLAMA
or an OPENAI LLM key is not found. This is expected. If you setup OLLAMA and/or
provided an LLM key, and were expecting to see human readable labels, follow the
steps below to troubleshoot. 

1. Check the setup for your LLM backend.

  OLLAMA: 
    - Check that the container is running. Run `docker compose up`  
  OPENAI: 
    - Confirm that you have set your OPENAI_API_KEY credentials in your environment. Read
      the documentation here: https://developers.openai.com/api/docs/quickstart

      ```
      # Export an environment variable on macOS or Linux systems
      export OPENAI_API_KEY="your_api_key_here"
    ```

2. Check the logs. If Ollama is running and/or an OpenAI key is found, the
   `src.ai_labeler.build_llm_representation()` will log a warning with the
   specific issue casing the error. For example, you might see: 

   ```
   WARNING  Ollama reachable but configured model 'qwen2.5:3b' is not available. Falling back to OpenAI API key or KeyBERT-only labels.
   WARNING  No LLM available — topic labels will use KeyBERT keywords only.
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

**IMPORTANT**: You must restart the Demo App after resetting your workspace. The app loads `sample_posts.py` on 
startup by default.  If you do not restart the app, you may see search errors if
you are running Elasticsearch because no posts are stored on the Elasticsearch Index.



