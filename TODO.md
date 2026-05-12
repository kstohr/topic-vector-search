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
 - [ ] github.com/codespaces (free tier)
 - [ ] Add side-by-side search comparison toggle to app
 - [ ] Update run of show
 - [ ] Update slides: 
        - [ ] Add setup slide (open on this slide before talk)
        - [ ] Add wrap up slide ... we did this, this and this. To learn more,
        do this, this and this.
        - [ ] Add "housekeeping" slide to deck 
            - check attendees have completed setup 
            - review agenda 
            - review repo 
            - explain flow (demo app, notebook, complete exercises, demo app,
            - explain exercises and how to run tests
       notebook, complete exercises... bonus: caption images)
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
 - [ ] Plan softball Q's for TA's add to run of show
-  [ ] Add workshop teardown script/notebook 
       - shutdown app 
       - remove Ollama cached model 
       - remove docker containers 
       - remove docker images 
       - remove root dir 
   - [x] Clean Code review: PROGRESS
        - [ ] Create classes
	      - [ ] Review modules for and ensure raise errors clearly and quickly. Error messages should guide attendees on how to fix the problem ("Run preprocessing.py and try again.") or ("Check the paramters passed to BERTopic model.")
	      - [ ] Ensure every module has logging
	      - [ ] Ok in preprocessing.py and related tests. We often refer to a PostDocument object as a "post" this is confusing. We should always refer to a Post object as "post" and a PostDocument object as a "postdoc" PROGRESS
	      - [ ] Check for repeated code lines. Is there an existing method we can use instead?
 - [ ] Review comments for stupid AI stuff PROGRESS
 - [ ] Test what would happen if you ran the same package on python 3.13 or 3.14
   in case someone has a newer python interpreter and doesn't change it. 
