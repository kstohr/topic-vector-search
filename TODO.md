TODO: 

 - [ ] Email Sarah about contacting participants
 - [x] Copy existing main to PyBay branch
 - [x] Fix bug the localized embedding toggle doesn't work on results in topic view
 - [x] Fix keyword search in memory to split on words (tokens) and run search with OR for all keywords.
 - [x] Commit working streamlit
 - [ ] Implement precision/recall metric
	 - [x] In workshop
	 - [ ] In progress code
	 - [ ] In notebook
 - [x] Write cleaner code; each method does one task well and only one task. Create classes
	 - [ ] Review modules for and ensure raise errors clearly and quickly. Error messages should guide attendees on how to fix the problem ("Run preprocessing.py and try again.") or ("Check the paramters passed to BERTopic model.")
	 - [ ] Ensure every module has logging
	 - [ ] Ok in preprocessing.py and related tests. We often refer to a PostDocument object as a "post" this is confusing. We should always refer to a Post object as "post" and a PostDocument object as a "postdoc" PROGRESS
	 - [ ] Check for repeated code lines. Is there an existing method we can use instead?
 - [ ] Fix references to centroids PROGRESS
        - [x] Fix in workshop 
        - [ ] Fix in slides
 - [ ] Add noise to posts (use sample posts, sample model)
	 - [ ] Create a script that uses tfidf to get common words for top topics
	 (cats, trains)
     - [ ] Add numbers 99 address, 99 Red Balloons, etc. 
     - [ ] Add PII (should we really cluster on a person's name if it's just an
     - [ ] Mexican Woman Presidential Bid - update slide to show improving prompt.
       @mention, or a name in say a copyright notice?)
 - [x] Add noise to agenda and [run of show] (https://docs.google.com/spreadsheets/d/1Yo4mmH4ojbLEFRLpCZeK9xkWJ2fLJxvNpKJPdO9My04/edit?gid=444788039#gid=444788039)
 - [x] Identify placeholder code for workshop - refactor to label and isolate methods. Ensure that tests exist for each method.
	 - [x] uv run pytest -m "not exercise"
 - [ ] Review comments for stupid AI stuff PROGRESS
 - [x] Create solutions files (preprocessing, evaluation)
	 - [x ] Will add exercise to data_model.py, topic_model.py, inline.
 - [ ] Create explainer notebooks 
        - [ ] Keyword vs Semantic Search (explain cosine similarity scoring)
        - [ ] What is an embedding exactly? 
        - [ ] Training a topic model 
            - Exercise instructions: 
                - small examples, try this ... does this happen? 
                    - changing the min-num of docs per topic (break)
                    - changing the VectorCount min-df (will dumb down labels or
                    break)
                    - changing the representation LLM prompt ('It's opposite
                    day!" )
        - [ ] Evaluating search retrieval 
        - [ ] Dealing with Noise 
        - [ ] Searching for images.. (bonus)
 - [ ] Create GLOSSARY.MD
 - [ ] Create FAQ.md
 - [ ] Create Troubleshooting.md
 - [ ] Update run of show
 - [ ] Add KNN /cosine similarity slide to talk, notebook
 - [ ] Add restart/tear down instructions to README.md (progress) / TROUBLESHOOTING.md PROGRESS
 - [x] Check requirements and installation restrictions
 - [ ] Refine Setup Instructions 
     - 6GB TO DOWNLOAD - Update docs
     - [ ] Update setup notebook to include instructions on switching python
       versions, 
     - [ ] installing uv 
            - mac 
            - windows (bash command)
            - linux (bash command)
	 - [ ] If don’t have python 3.12, if you don’t have uv … do this.
	 - [ ] Add slide on setup
 - [ ] Plan softball Q's for TA's add to run of show
 - [ ] Add wrap up slide ... we did this, this and this. 
        - If you want to learn more, do this, this and this.

