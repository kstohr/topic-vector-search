# Teardown Instructions

Clean up resources used by this workshop. Each section is independent — you can skip steps if you use those tools for other projects.

---

## 1. Remove HuggingFace Cached Embedding Models

This project uses the `all-MiniLM-L6-v2` embedding model (~90 MB).

**Remove only this project's model (safe):**
```bash
rm -rf ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2
```

**Or, to clear the entire HuggingFace cache** (removes ALL cached models from all projects):
```bash
rm -rf ~/.cache/huggingface
```

---

## 2. Remove Ollama Models and Application

### Remove the Ollama model (`qwen2.5:3b`) only:
```bash
ollama rm qwen2.5:3b
```

### Remove all Ollama models:
```bash
ollama list | awk 'NR>1 {print $1}' | xargs -I {} ollama rm {}
```

### Uninstall Ollama completely:

**macOS:**
```bash
# Via Homebrew (if installed that way)
brew uninstall ollama

# Or manually: Applications > Ollama > drag to Trash, then remove:
rm -rf ~/.ollama
```

**Linux:**
```bash
sudo systemctl stop ollama  # if running as service
sudo apt-get remove ollama  # Ubuntu/Debian
# or
sudo dnf remove ollama      # Fedora/RHEL
```

**Windows:**
```
Control Panel > Programs > Programs and Features > Ollama > Uninstall
```

---

## 3. Shutdown Docker Containers and Remove Images

### Stop and remove this project's containers:
```bash
cd /path/to/topic-vector-search
docker compose down
```

### Remove the images** (both Elasticsearch and Ollama):
```bash
docker compose down --rmi all
```

**Or, manually remove only the images used by this project:**
```bash
docker rmi docker.elastic.co/elasticsearch/elasticsearch:8.10.0
docker rmi ollama/ollama:latest
```

### ⚠️ Full cleanup (remove dangling images/volumes from all projects):
```bash
# Remove unused images
docker image prune

# Remove unused volumes
docker volume prune

# Remove all unused images, volumes, and networks (aggressive)
docker system prune -a
```

---

## Summary: One-Step Teardown

If you want to clean up everything at once (safe for this project):

```bash
# Stop and remove Docker containers/images
cd /path/to/topic-vector-search
docker compose down --rmi all

# Remove Ollama model
ollama rm qwen2.5:3b

# Remove HuggingFace embedding model
rm -rf ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2
```

---

## Notes

- **Project venv:** If you want to remove the Python virtual environment, run:
  ```bash
  rm -rf .venv
  ```

- **Project data:** Output files (CSVs, embeddings, models) are in `output/`. To reset:
  ```bash
  rm -rf output/*
  ```

- **Safe to keep:** Your local git history and source code are unaffected by
  these steps.

If you want to remove the entire repo from your local env: 
```bash
cd ../ 
rm -rf topic-vector-search
```
