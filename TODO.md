TODO: 

 - [ ] Contact participants about setup 
Hello!

Excited to present the workshop Thinking of Topic Modeling as Search tomorrow.
Please take a moment today to download the workshop requirements. 
 
1. Go to https://github.com/kstohr/topic-vector-search
2. Follow the "Installation" instructions on the [README.md](https://github.com/kstohr/topic-vector-search/blob/main/README.md)
3. Run the [Setup Notebook](https://github.com/kstohr/topic-vector-search/blob/main/notebooks/00_setup_check.ipynb) - Checks that all systems are go!~ 
 
 **Do this before the workshop** 

- This project involves both search and language models. Installation
       requires at least ~15GB download and **can be very slow** on conference
       wifi. Especially on older machines. 
- To enable the repo to run cross-platform (Mac Intel, Mac Silicon,
       Windows), we are not running the latest version of python. You may need
       to install python 3.12. Some packages are also pinned to older versions
       for compatibility. While we have tried to test it on different operating
       systems, it is possible you may run into package conflicts. These can
       take time to resolve. 
- We have created a troubleshooting guide to help resolve issues you may
  experience both installing the workshop materials and running code during the
  workshop
  [TROUBLESHOOTING.md](https://github.com/kstohr/topic-vector-search/blob/main/TROUBLESHOOTING.md)
  
 If you have any issues or questions, email both Chris Brousseau
 (chris@surfaceowl.com)  and myself. We are happy to help. 

Workshop Presenter: 
Kas Stohr, kas@99antennas.com

Teaching Assistant: 
Chris Brousseau, chris@surfaceowl.com


P.S. 

See you tomorrow! 
  
 - [x ] Follow up with Paul about TA'ing
 - [x] Clean Code review: PROGRESS
        - [ ] Create classes
	      - [ ] Review modules for and ensure raise errors clearly and quickly. Error messages should guide attendees on how to fix the problem ("Run preprocessing.py and try again.") or ("Check the paramters passed to BERTopic model.")
	      - [ ] Ensure every module has logging
	      - [ ] Ok in preprocessing.py and related tests. We often refer to a PostDocument object as a "post" this is confusing. We should always refer to a Post object as "post" and a PostDocument object as a "postdoc" PROGRESS
	      - [ ] Check for repeated code lines. Is there an existing method we can use instead?
 - [ ] Review comments for stupid AI stuff PROGRESS
 - [x] Update explainer notebooks PROGRESS
       - [x] Add pre-run checks at top of notebooks 
       - [ ] Add outline to 01_lexical_vs_sematnic.py, check outline matches
       markdown (all notebooks)
       - [ ] Add parameter comments to 03_topic_modeling.py
       - [ ] Move postdoc model from code into notebook_02 with structured docs; so
       users are not updating the original methods. 
       - [x] Clarify, simplify exercise prompts in notebooks
          - [x] Add instructions to run pytest for methods in exercises
       - [ ] notebook_03:  need HDDBSCAN image for Step 2 to make concept of balancing
  good matches with clusters make sense...
  (generate_embedding(), success metrics, vision)
  that when user changes code they don't mess with source pipeline
 - [x] Create GLOSSARY.MD PROGRESS
        - [x] check links
 - [x] Create FAQ.md PROGRESS
        - [x] check links
 - [x] Create Troubleshooting.md PROGRESS
 - [ ] Update run of show
 - [ ] Update README.md 
        - [ ] Add restart/
        - [ ] Add tear down
        - [ ] Add TROUBLESHOOTING.md
        - [ ] If don’t have python 3.12, if you don’t have uv … do this.
        - [ ] installing uv 
            - mac 
            - windows (bash command)
            - linux (bash command)
 - [ ] Check requirements and installation restrictions
        - pin packages
 - [ ] github.com/codespaces (free tier)
 - [x] Refine Setup Check Notebook
     - [x] 6GB TO DOWNLOAD - Update docs
     - [x] Update setup notebook to include uv check of python install versions, 
	 - [ ] Add slide on setup
 - [ ] Plan softball Q's for TA's add to run of show
 - [ ] Update slides: 
        - [ ] Add setup slide (open on this slide before talk)
        - [ ] Add wrap up slide ... we did this, this and this. To learn more,
        do this, this and this.
        -  [ ] Add "housekeeping" slide to deck 
            - check attendees have completed setup 
            - review agenda 
            - review repo 
            - explain flow (demo app, notebook, complete exercises, demo app,
            - explain exercises and how to run tests
       notebook, complete exercises... bonus: caption images)
 - [ ] Add side-by-side search comparison toggle to app
 - [ ] Test what would happen if you ran the same package on python 3.13 or 3.14
   in case someone has a newer python interpreter and doesn't change it. 
-  [ ] Add global reset script 
       - check global reset command -  does it reset everything needed?  does it
       rebuild? 
- [ ] Add instructions to run pytest for methods in exercises
  (generate_embedding(), success metrics, vision)
    - [ ] in notebooks 
    - [ ] in code

-  [ ] Add workshop teardown script/notebook 
       - shutdown app 
       - remove Ollama cached model 
       - remove docker containers 
       - remove docker images 
       - remove root dir 
  


----------------



INFO:sentence_transformers.SentenceTransformer:Load pretrained SentenceTransformer: all-MiniLM-L6-v2
2026-05-06 22:20:56.997 Uncaught app execution
Traceback (most recent call last):
  File "/Users/kas/dev/pycon_2026/topic-vector-search/.venv/lib/python3.12/site-packages/streamlit/runtime/scriptrunner/exec_code.py", line 129, in exec_func_with_error_handling
    result = func()
             ^^^^^^
  File "/Users/kas/dev/pycon_2026/topic-vector-search/.venv/lib/python3.12/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 689, in code_to_exec
    exec(code, module.__dict__)  # noqa: S102
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/kas/dev/pycon_2026/topic-vector-search/app.py", line 591, in <module>
    searcher=build_topic_searcher(topic_engine),
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/kas/dev/pycon_2026/topic-vector-search/.venv/lib/python3.12/site-packages/streamlit/runtime/caching/cache_utils.py", line 281, in __call__
    return self._get_or_create_cached_value(args, kwargs, spinner_message)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/kas/dev/pycon_2026/topic-vector-search/.venv/lib/python3.12/site-packages/streamlit/runtime/caching/cache_utils.py", line 326, in _get_or_create_cached_value
    return self._handle_cache_miss(cache, value_key, func_args, func_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/kas/dev/pycon_2026/topic-vector-search/.venv/lib/python3.12/site-packages/streamlit/runtime/caching/cache_utils.py", line 385, in _handle_cache_miss
    computed_value = self._info.func(*func_args, **func_kwargs)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/kas/dev/pycon_2026/topic-vector-search/app.py", line 227, in build_topic_searcher
    return get_topic_searcher(load_doc_index(), engine)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/kas/dev/pycon_2026/topic-vector-search/src/search.py", line 275, in get_topic_searcher
    return _SEARCHER_CLASSES[engine](posts)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/kas/dev/pycon_2026/topic-vector-search/src/search.py", line 196, in __init__
    embeddings = np.array([p["doc_embedding"] for p in posts], dtype=np.float32)
                           ~^^^^^^^^^^^^^^^^^
KeyError: 'doc_embedding'
INFO:sentence_transformers.SentenceTransformer:Use pytorch device_name: mps
INFO:sentence_transformers.SentenceTransformer:Load pretrained SentenceTransformer: all-MiniLM-L6-v2
2026-05-06 22:21:04.266 Uncaught app execution
Traceback (most recent call last):
  File "/Users/kas/dev/pycon_2026/topic-vector-search/.venv/lib/python3.12/site-packages/streamlit/runtime/scriptrunner/exec_code.py", line 129, in exec_func_with_error_handling
    result = func()
             ^^^^^^
  File "/Users/kas/dev/pycon_2026/topic-vector-search/.venv/lib/python3.12/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 689, in code_to_exec
    exec(code, module.__dict__)  # noqa: S102
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/kas/dev/pycon_2026/topic-vector-search/app.py", line 591, in <module>
    searcher=build_topic_searcher(topic_engine),
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/kas/dev/pycon_2026/topic-vector-search/.venv/lib/python3.12/site-packages/streamlit/runtime/caching/cache_utils.py", line 281, in __call__
    return self._get_or_create_cached_value(args, kwargs, spinner_message)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/kas/dev/pycon_2026/topic-vector-search/.venv/lib/python3.12/site-packages/streamlit/runtime/caching/cache_utils.py", line 326, in _get_or_create_cached_value
    return self._handle_cache_miss(cache, value_key, func_args, func_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/kas/dev/pycon_2026/topic-vector-search/.venv/lib/python3.12/site-packages/streamlit/runtime/caching/cache_utils.py", line 385, in _handle_cache_miss
    computed_value = self._info.func(*func_args, **func_kwargs)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/kas/dev/pycon_2026/topic-vector-search/app.py", line 227, in build_topic_searcher
    return get_topic_searcher(load_doc_index(), engine)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/kas/dev/pycon_2026/topic-vector-search/src/search.py", line 275, in get_topic_searcher
    return _SEARCHER_CLASSES[engine](posts)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/kas/dev/pycon_2026/topic-vector-search/src/search.py", line 196, in __init__
    embeddings = np.array([p["doc_embedding"] for p in posts], dtype=np.float32)
                           ~^^^^^^^^^^^^^^^^^
KeyError: 'doc_embedding'
