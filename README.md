## Sentiment-analysis-nlp

### How to run:
* Install python 3.6.1 (tensorflow not supporting 3.7 currently)
* Install needed modules with: pip install -r src/requirements.txt
* Change dir from root to /src
* Run project with: python main.py

### Results:

#### Lexicons only, 2 classes:
* **Not using preprocessed lexicons:**

  Negation\Levenshtein's distance | 0 | 1 | 2 | 3 | 4 | 5 | 6
  :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---:
  False | <img src="results/lex_only_2_classes/npp_f/Figure_1.png" width="150"/> 59.2% | <img src="results/lex_only_2_classes/npp_f/Figure_2.png" width="150"/> 58.5% | <img src="results/lex_only_2_classes/npp_f/Figure_3.png" width="150"/> 56.6% | <img src="results/lex_only_2_classes/npp_f/Figure_4.png" width="150"/> 56.7% | <img src="results/lex_only_2_classes/npp_f/Figure_5.png" width="150"/> 56.2% | <img src="results/lex_only_2_classes/npp_f/Figure_6.png" width="150"/> 56.2% | <img src="results/lex_only_2_classes/npp_f/Figure_7.png" width="150"/> 55.6%
  True | <img src="results/lex_only_2_classes/npp_t/Figure_1.png" width="150"/> 59.5% | <img src="results/lex_only_2_classes/npp_t/Figure_2.png" width="150"/> 58.5% | <img src="results/lex_only_2_classes/npp_t/Figure_3.png" width="150"/> 56.7% | <img src="results/lex_only_2_classes/npp_t/Figure_4.png" width="150"/> 57.6% | <img src="results/lex_only_2_classes/npp_t/Figure_5.png" width="150"/> 55.8% | <img src="results/lex_only_2_classes/npp_t/Figure_6.png" width="150"/> 56.1% | <img src="results/lex_only_2_classes/npp_t/Figure_7.png" width="150"/> 55.5%

* **Using preprocessed lexicons:**

  Negation\Levenshtein's distance | 0 | 1 | 2 | 3 | 4 | 5 | 6
  :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---:
  False | <img src="results/lex_only_2_classes/pp_f/Figure_1.png" width="150"/> 58.8% | <img src="results/lex_only_2_classes/pp_f/Figure_2.png" width="150"/> 60.5% | <img src="results/lex_only_2_classes/pp_f/Figure_3.png" width="150"/> 58.2% | <img src="results/lex_only_2_classes/pp_f/Figure_4.png" width="150"/> 58% | <img src="results/lex_only_2_classes/pp_f/Figure_5.png" width="150"/> 58.9% | <img src="results/lex_only_2_classes/pp_f/Figure_6.png" width="150"/> 58.5% | <img src="results/lex_only_2_classes/pp_f/Figure_7.png" width="150"/> 58.4%
  True | <img src="results/lex_only_2_classes/pp_t/Figure_1.png" width="150"/> 60% | <img src="results/lex_only_2_classes/pp_t/Figure_2.png" width="150"/> 60.7% | <img src="results/lex_only_2_classes/pp_t/Figure_3.png" width="150"/> 59.1% | <img src="results/lex_only_2_classes/pp_t/Figure_4.png" width="150"/> 59.1% | <img src="results/lex_only_2_classes/pp_t/Figure_5.png" width="150"/> 59.3% | <img src="results/lex_only_2_classes/pp_t/Figure_6.png" width="150"/> 59.3% | <img src="results/lex_only_2_classes/pp_t/Figure_7.png" width="150"/> 59.4%


#### Lexicons only, 3 classes:
* **Not using preprocessed lexicons:**

  Negation\Levenshtein's distance | 0 | 1 | 2 | 3 | 4 | 5 | 6
  :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---:
  False | <img src="results/lex_only_3_classes/npp_f/Figure_1.png" width="150"/> 39.9% | <img src="results/lex_only_3_classes/npp_f/Figure_2.png" width="150"/> 39% | <img src="results/lex_only_3_classes/npp_f/Figure_3.png" width="150"/> 37.4% | <img src="results/lex_only_3_classes/npp_f/Figure_4.png" width="150"/> 37.1% | <img src="results/lex_only_3_classes/npp_f/Figure_5.png" width="150"/> 37.2% | <img src="results/lex_only_3_classes/npp_f/Figure_6.png" width="150"/> 37.4% | <img src="results/lex_only_3_classes/npp_f/Figure_7.png" width="150"/> 37.3%
  True | <img src="results/lex_only_3_classes/npp_t/Figure_1.png" width="150"/>  40.3% | <img src="results/lex_only_3_classes/npp_t/Figure_2.png" width="150"/> 39.3% | <img src="results/lex_only_3_classes/npp_t/Figure_3.png" width="150"/> 38.3% | <img src="results/lex_only_3_classes/npp_t/Figure_4.png" width="150"/> 38.2% | <img src="results/lex_only_3_classes/npp_t/Figure_5.png" width="150"/> 38.4% | <img src="results/lex_only_3_classes/npp_t/Figure_6.png" width="150"/> 37.8% | <img src="results/lex_only_3_classes/npp_t/Figure_7.png" width="150"/> 37.8%

* **Using preprocessed lexicons:**

  Negation\Levenshtein's distance | 0 | 1 | 2 | 3 | 4 | 5 | 6
  :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---:
  False | <img src="results/lex_only_3_classes/pp_f/Figure_1.png" width="150"/> 38.7% | <img src="results/lex_only_3_classes/pp_f/Figure_2.png" width="150"/> 39% | <img src="results/lex_only_3_classes/pp_f/Figure_3.png" width="150"/> 38.4% | <img src="results/lex_only_3_classes/pp_f/Figure_4.png" width="150"/> 39.3% | <img src="results/lex_only_3_classes/pp_f/Figure_5.png" width="150"/> 38.4% | <img src="results/lex_only_3_classes/pp_f/Figure_6.png" width="150"/> 38.2% | <img src="results/lex_only_3_classes/pp_f/Figure_7.png" width="150"/> 38.1%
  True | <img src="results/lex_only_3_classes/pp_t/Figure_1.png" width="150"/> 39.4% | <img src="results/lex_only_3_classes/pp_t/Figure_2.png" width="150"/> 39.2% | <img src="results/lex_only_3_classes/pp_t/Figure_3.png" width="150"/> 39.4% | <img src="results/lex_only_3_classes/pp_t/Figure_4.png" width="150"/> 39.5% | <img src="results/lex_only_3_classes/pp_t/Figure_5.png" width="150"/> 38.6% | <img src="results/lex_only_3_classes/pp_t/Figure_6.png" width="150"/> 39% | <img src="results/lex_only_3_classes/pp_t/Figure_7.png" width="150"/> 38.8%

#### Neural network with Lexicons, 2 classes:

* **Adeline with no bias**
    * Cross validation, with 5 splits
    * Stratification
    * Number of epochs: 100
    * Accuracy: 60.9%

      Split num | 0 | 1 | 2 | 3 | 4
      :---: | :---: | :---: | :---: | :---: | :---:
      | | 58.9% | 61.6% | 61.9% | 61% | 61%

* **Adeline with bias**
    * Cross validation, with 5 splits
    * Stratification
    * Number of epochs: 100
    * Accuracy: 60.7%

      Split num | 0 | 1 | 2 | 3 | 4
      :---: | :---: | :---: | :---: | :---: | :---:
      | | 59.2% | 61% | 61.9% | 61% | 60.4%

* **1 layer perceptron**
    * Cross validation, with 5 splits
    * Stratification
    * Number of epochs: 100
    * Accuracy: 56.01%
  
      Split num | 0 | 1 | 2 | 3 | 4
      :---: | :---: | :---: | :---: | :---: | :---:
      | | <img src="results/1_layer_perceptron_2_classes/Fold_1.png" width="150"/> 55.3% | <img src="results/1_layer_perceptron_2_classes/Fold_2.png" width="150"/> 53.6% | <img src="results/1_layer_perceptron_2_classes/Fold_3.png" width="150"/> 53.8% | <img src="results/1_layer_perceptron_2_classes/Fold_4.png" width="150"/> 58.9% | <img src="results/1_layer_perceptron_2_classes/Fold_5.png" width="150"/> 58.3%
      
#### Neural network with Lexicons, 3 classes:
      
* **1 layer perceptron**
    * Cross validation, with 5 splits
    * Stratification
    * Number of epochs: 100
    * Accuracy: 39.5%
  
      Split num | 0 | 1 | 2 | 3 | 4
      :---: | :---: | :---: | :---: | :---: | :---:
      | | <img src="results/1_layer_perceptron_3_classes/Fold_1.png" width="150"/> 36.3% | <img src="results/1_layer_perceptron_3_classes/Fold_2.png" width="150"/> 38.5% | <img src="results/1_layer_perceptron_3_classes/Fold_3.png" width="150"/> 41.5% | <img src="results/1_layer_perceptron_3_classes/Fold_4.png" width="150"/> 40.1% | <img src="results/1_layer_perceptron_3_classes/Fold_5.png" width="150"/> 41.3%
