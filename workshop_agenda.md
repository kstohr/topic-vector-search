# **Workshop Outline (Hands-On)**

**Session type:**   
Tutorial   
Duration 3 hours

## ** Schedule (180 min)**

| # | Section | Time | Notebooks + Exercises |
|---|---------|------|-----------------------|
| 1 | Introduction & Goals | 5 min | Slides + repo tour |
| 2 | Setup | 10 min | [Notebook 0](notebooks/00_setup_check.ipynb) |
| 3 | Keyword Search vs. Semantic Search | 10 min | Demo App |
| 4 | Understanding Embeddings | 25 min | [Notebook 1](notebooks/01_embeddings.ipynb) + Exercise 1: [generate_embeddings](src/preprocess.py#L135) |
| 5 | Topic Modeling with BERTopic | 25 min | [Notebook 2](notebooks/02_topic_modeling.ipynb) + Exercise 2: [train_topic_model](src/topic_model.py#L115) |
| Break | Coffee break + Survey | 40 min | Survey after break |
| 6 | Evaluating Topic Search Retrieval | 20 min | [Notebook 3](notebooks/03_search_evaluation.ipynb) + Exercise 3: [compute_precision_at_k](src/evaluation.py#L68), [compute_recall_at_k](src/evaluation.py#L93), [compute_random_baseline](src/evaluation.py#L109), [preprocess_text](src/data_models.py#L69) #  |
| 7 | Including Images in Topic Search Retrieval (bonus) | 25 min | [Notebook 4](notebooks/04_images_bonus.ipynb) + Exercise 4: [_caption_single_post](src/preprocess.py#L115) |
| 8 | Wrap-Up | 10 min + extra | |
| **Total** | | **180 min** | |


## Learning Objectives 

1. Understand the difference between keyword search (lexical/BM25) and semantic search 
2. Preprocess posts, generate embeddings and implement semantic search. 
3. Build a basic topic model 
4. Retrieve topic-related documents by searching with topic embeddings
5. Define key metrics and evaluate topic search retrieval
6. Learn how to refine the pipeline to improve results. 
