# An approach to solve RUSSE 2018 Word Sense Induction and Disambiguation Shared Task

My solutions for the [shared task on word sense induction and disambiguation for the Russian language](http://russe.nlpub.org/2018/wsi).

### Overview of my approach
Для выполнения задания использовал предобученный word2vec embedding и pymystem для лемматизации и pos-tagging.
Мой подход отличается от подхода baseline тем что я сосредоточился на графовых кластеризациях построенных 
из TOP-200 похожих слов для каждого целевого слова (ego-network). Полносвязанный граф фильтровался от ребер посредством граничных значений
 и далее применялся алгоритм кластеризации. На каждый кластер взвешенным средним получал свой cluster vector. В методе disambiguation отфильтровывал контекст от незначимых слов и по контекстному вектору выбирал самый похожий кластер и выдавал его идентификатор. Далее по train dataset я подобрал веса значимости k кластеров.

ps: идентификаторы кластеров 0..k изначально формируются от самого похожего cluster vector к целевому слову (по убывающей).

Метрику ARI до 0.2 не довел, остановился на 0.1769 вероятно подход не самый лучший выбрал, но интересно было порешать задачу.

The task
-------------------

Your goal is to **design a system which takes as an input a pair of (word, context) and outputs the sense identifier**, e.g. "1" or "2". This is important to note that it does not matter which sense identifiers you use (numbers in the "gold_sense_id" and "predict_sense_id" columns)! It is not needed that they match sense identifiers of the gold standard! For instance, if in the "gold_sense_id" column you use identifiers {a,b,c} and in the "predict_sense_id" you use identifiers {1,2,3}, but the labelling of the data match so that each context labeled with "1" is always labeled with "a", each context labeled with "2" is always labeled with "b", etc. you will get the top score. Matching of the gold and predict sense inventories is not a requirement as we use [clustering based evaluation](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html), namely we rely on the [Adjusted Rand Index](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html). Therefore, your cluster sense labels should not necessarily correspond to the labels from the gold standard.

Thus, the successful submissions will group all contexts referring to the same word sense (by assigning the same ```predict_sense_id```). To achieve this goal, you can you models which induce sense inventory from a large corpus of all words in the corpus, e.g. Adagram or try to cluster directly the contexts of one word, e.g. using the k-Means algorithm. Besides, you can use an existing sense inventory from a dictionary, e.g. RuWordNet, to build your modes (which again do not match exactly the gold dataset, but this is not a problem).  
Below we provide more details on differences between two tracks.


#### Example
You are given a word, e.g. ```"замок"``` and a bunch of text fragments (aka "contexts") where this word occurs, e.g. ```"замок владимира мономаха в любече"``` and  ```"передвижению засова ключом в замке"```. You need to cluster these contexts in the (unknown in advance) number of clusters which correspond to various senses of the word. In this example, you want to have two groups with the contexts of the "lock" and the "castle" senses of the word ```"замок"```. For each of the three test datasets, you need to download the **text.csv** file, fill the ``predicted_sense_id`` in this file.


Using the evaluation script
--------------------------

To evaluate a sample baseline based on the Adagram sense embeddings model, provided by the organizers, run:
```
python3 evaluate.py data/main/wiki-wiki/train.baseline-adagram.csv
```

You should get the ARI scores per word and also the final value per dataset (the last value, here 0.392449):

```
word   ari       count
бор    0.591175  56
замок  0.495386  138
лук    0.637076  110
суда   0.005465  135
       0.392449  439
```

The output of your system should have the same format as the sample baseline file provided in this repository ```data/main/wiki-wiki/train.baseline-adagram.csv```. When the test data will be available, you'll need to run your system against a test file in the similar format and submit to a CSV file with the result to the organizers.


Description of the datasets
--------

The participants of the shared task need to work with three datasets of varying sense inventories and types of texts. All the datasets are located in the directory ```data/main```. One dataset is located in one directory. The name of the directory is ```<inventory>-<corpus>```. For instance ```bts-rnc```, which represents datasets based on the word sense inventory BTS (Bolshoi Tolkovii Slovar') and the RNC corpus. Here is the list of the datasets:

1. **active-dict** located in ```data/main/active-dict```: The senses of this dataset correspond to the senses of the Active Dictionary of the Russian Language a.k.a. the 'Dictionary of Apresyan'. Contexts are extracted from examples and illustrations sections from the same dictionary.

In addition, in the directory ```data/additional```, we provide three extra datasets, which can be used as additional training data from (Lopukhin and Lopukhina, 2016). These datasets are based on various sense inventories (active dictionary, BTS) and various corpora (RNC, RuTenTen). Note that we will not release any test datasets that correspond to these datasets (yet they still may be useful e.g. for multi-task learning).  

The table below summarizes the datasets:

|Dataset|Type|Inventory|Corpus|Split|Num. of words|Num. of senses|Avg. num. of senses|Num. of contexts|
|-----|-----|---------|-----|------|:---------:|:----------:|:----------:|:----------:|
|active-dict|main|Active Dict.|Active Dict.|train|85|312|3.7|2073
|active-rutenten|additional|Active Dict.|ruTenTen|train|21|71|3.4|3671

Format of the dataset files
----------

Train and test datasets are stored in .csv files (the name of the folder corresponds to the name of the dataset), each file has a header:

```
context_id    word    gold_sense_id    predict_sense_id    positions    context
```

**Type of the file and dialect**: CSV (TSV): tab separated,  with a header, no quote chars for fields, one row is one line in the file (no multi-line rows are supported).

**Encoding**: utf-8

**Target**: ```predict_sense_id``` is the prediction of a system (this field is not filled by default)

**Sample**:

```
context_id    word    gold_sense_id    predict_sense_id    positions    context

1    граф    2        0-3,132-137    Граф -- это структура данных. Кроме этого, в дискретной математике теория графов ...

...    ...    ...    ...    ...    ...
```

Structure of this repository
---------------------------

- ```solutions.py``` -- my solution for the task.
- ```data``` -- directory with the train datasets and the corresponding baselines based on the Adagram
- ```evaluate.py``` -- evaluation script used to compute the official performance metric (ARI).
- ```baseline_trivial.py``` -- a script for generation of various trivial baselines, e.g. random labels.
- ```baseline_adagram.py``` -- the script used to generate the Adagram baselines
- ```requirements.txt``` -- the list of all dependencies of the ```evaluate.py``` and ```baseline_trivial.py``` scripts (note that the ```baseline_adagram.py``` has more dependencies specified inside this file)

