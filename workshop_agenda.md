# **Workshop Outline (Hands-On)**

**Session type:**   
Tutorial   
Duration 3 hours

## ** Schedule (180 min)**

| # | Section | Time | Notebooks + Exercises |
|---|---------|------|-----------------------|
| 1 | Introduction & Goals | 5 min | Slides + repo tour |
| 2 | Setup | 10 min | [Notebook 0](notebooks/00_setup_check.ipynb) |
| 3 | Lexical Search vs. Semantic Search | 10 min | [Notebook 1](notebooks/01_lexical_vs_semantic_search.ipynb) + [Demo App](http://localhost:8501/) (Don't forget to start the app!) |
| 4 | Understanding Embeddings | 25 min | [Notebook 2](notebooks/02_embeddings.ipynb) + Exercise 2: [generate_embeddings](src/preprocess.py#L130) |
| 5 | Topic Modeling with BERTopic | 25 min | [Notebook 3](notebooks/03_topic_modeling.ipynb) + Exercise 3: [train_topic_model](src/topic_model.py#L108) |
| Break | Coffee break + Survey | 40 min | Survey after break |
| 6 | Evaluating Topic Search Retrieval | 20 min | [Notebook 4](notebooks/04_search_evaluation.ipynb) + Exercise 4: [compute_precision_at_k](src/evaluation.py#L59), [compute_recall_at_k](src/evaluation.py#L83), [compute_random_baseline](src/evaluation.py#L101), [preprocess_text](src/data_models.py#L63) |
| 7 | Including Images in Topic Search Retrieval (bonus) | 25 min | [Notebook 6](notebooks/06_images_bonus.ipynb) + Exercise 5: [_caption_single_post](src/preprocess.py#L98) |
| 8 | Wrap-Up | 10 min + extra | |
| **Total** | | **180 min** | |


## Learning Objectives 

1. Understand the difference between lexical search (keyword/lexical/BM25) and semantic search 
2. Preprocess posts, generate embeddings and implement semantic search. 
3. Build a basic topic model 
4. Retrieve topic-related documents by searching with topic embeddings
5. Define key metrics and evaluate topic search retrieval
6. Learn how to refine the pipeline to improve results. 
