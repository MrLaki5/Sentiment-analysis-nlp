## Sentiment-analysis-nlp

### About:
* Project done for faculty course [natural language processing](https://rti.etf.bg.ac.rs/rti/ms1opj/). Subject of project was to do sentiment analysis of movie comments in Serbian. Sentiment analysis was done in few ways: with prepared lexicons only, with ML only and with mix of neural networks and prepared lexicons.

### How to run:
* Install python 3.6.1
* Install needed modules with: pip install -r src/requirements.txt
* Change dir from root to /src
* Run project with: python main.py

### References:
* Stemmer used in project, [link](http://vukbatanovic.github.io/SCStemmers/)
* Dataset with movie comments used in project, [link](https://vukbatanovic.github.io/SerbMR/)
* German lexicon used in project, [link](https://www.kaggle.com/rtatman/german-sentiment-analysis-toolkit)
    * R. Remus, U. Quasthoff & G. Heyer: SentiWS - a Publicly Available German-language Resource for Sentiment Analysis. In: Proceedings of the 7th International Language Ressources and Evaluation (LREC'10), 2010
This version of the data set was last updated in March 2012.
* English lexicon used in project, [link]()

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
      
* **Multiple layer perceptron**
    * Epoch number 100, with option to stop if overfitt starts 
    * Hyperparameters choosen for best accuracy with grid search
    * Cross validation
    * Stratification
    * Class of models: reduce_first PCA
    * Input layer: 1839 neurons
    * First hidden layer: 10 neurons
    * No second hidden layer
    * Accuracy on test set: 73.8%
    </br>
      <img src="results/mlp/mlp_2_classes.png" width="200"/>
      
#### Neural network with Lexicons, 3 classes:
* **1 layer perceptron**
    * Cross validation, with 5 splits
    * Stratification
    * Number of epochs: 100
    * Accuracy: 39.5%
  
      Split num | 0 | 1 | 2 | 3 | 4
      :---: | :---: | :---: | :---: | :---: | :---:
      | | <img src="results/1_layer_perceptron_3_classes/Fold_1.png" width="150"/> 36.3% | <img src="results/1_layer_perceptron_3_classes/Fold_2.png" width="150"/> 38.5% | <img src="results/1_layer_perceptron_3_classes/Fold_3.png" width="150"/> 41.5% | <img src="results/1_layer_perceptron_3_classes/Fold_4.png" width="150"/> 40.1% | <img src="results/1_layer_perceptron_3_classes/Fold_5.png" width="150"/> 41.3%
* **Multiple layer perceptron**
    * Epoch number 100, with option to stop if overfitt starts 
    * Hyperparameters choosen for best accuracy with grid search
    * Cross validation
    * Stratification
    * Class of models: reduce_last TruncatedSVD
    * Input layer: 100 neurons
    * First hidden layer: 20 neurons
    * Second hidden layer: 10 neurons, sigmoid activation
    * Accuracy on test set: 45.9%
    </br>
      <img src="results/mlp/mlp_3_classes.png" width="200"/>
