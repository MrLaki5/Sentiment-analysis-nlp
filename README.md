## Sentiment-analysis-nlp

### How to run:
* Install python 3.6.1 (tensorflow not supporting 3.7 currently)
* Install needed modules with: pip install -r src/requirements.txt
* Change dir from root to /src
* Run project with: python main.py

### Results:

#### Lexicons only:
* Not using preprocessed lexicons:

Negation\Levenshtein's distance | 0 | 1 | 2 | 3 | 4 | 5 | 6
--- | --- | --- | --- | --- | --- | --- | ---
False | <img src="results/lex_only_2_classes/npp_f/Figure_1.png" width="150"/> | <img src="results/lex_only_2_classes/npp_f/Figure_2.png" width="150"/> | <img src="results/lex_only_2_classes/npp_f/Figure_3.png" width="150"/> | <img src="results/lex_only_2_classes/npp_f/Figure_4.png" width="150"/> | <img src="results/lex_only_2_classes/npp_f/Figure_5.png" width="150"/> | <img src="results/lex_only_2_classes/npp_f/Figure_6.png" width="150"/> | <img src="results/lex_only_2_classes/npp_f/Figure_7.png" width="150"/>
True | <img src="results/lex_only_2_classes/npp_t/Figure_1.png" width="150"/> | <img src="results/lex_only_2_classes/npp_t/Figure_2.png" width="150"/> | <img src="results/lex_only_2_classes/npp_t/Figure_3.png" width="150"/> | <img src="results/lex_only_2_classes/npp_t/Figure_4.png" width="150"/> | <img src="results/lex_only_2_classes/npp_t/Figure_5.png" width="150"/> | <img src="results/lex_only_2_classes/npp_t/Figure_6.png" width="150"/> | <img src="results/lex_only_2_classes/npp_t/Figure_7.png" width="150"/>

* Using preprocessed lexicons:

Negation\Levenshtein's distance | 0 | 1 | 2 | 3 | 4 | 5 | 6
--- | --- | --- | --- | --- | --- | --- | ---
False | <img src="results/lex_only_2_classes/pp_f/Figure_1.png" width="150"/> | <img src="results/lex_only_2_classes/pp_f/Figure_2.png" width="150"/> | <img src="results/lex_only_2_classes/pp_f/Figure_3.png" width="150"/> | <img src="results/lex_only_2_classes/pp_f/Figure_4.png" width="150"/> | <img src="results/lex_only_2_classes/pp_f/Figure_5.png" width="150"/> | <img src="results/lex_only_2_classes/pp_f/Figure_6.png" width="150"/> | <img src="results/lex_only_2_classes/pp_f/Figure_7.png" width="150"/>
True | <img src="results/lex_only_2_classes/pp_f/Figure_1.png" width="150"/> | <img src="results/lex_only_2_classes/pp_f/Figure_2.png" width="150"/> | <img src="results/lex_only_2_classes/pp_f/Figure_3.png" width="150"/> | <img src="results/lex_only_2_classes/pp_f/Figure_4.png" width="150"/> | <img src="results/lex_only_2_classes/pp_f/Figure_5.png" width="150"/> | <img src="results/lex_only_2_classes/pp_f/Figure_6.png" width="150"/> | <img src="results/lex_only_2_classes/pp_f/Figure_7.png" width="150"/>
