# Code accompanying "Surveying the space of descriptions of a composite system with machine learning" (2024) 
by Kieran A Murphy, Yujing Zhang, and Dani S Bassett [[arxiv](https://arxiv.org/abs/)]

![Figure from the manuscript showing the space of descriptions of a 5-spin system.](/images/spin_system.png)

The analyses of the manuscript can be run through `train.py` and the accompanying command line arguments.  For instance, to find the 6 bits of information from a 4x4 Sudoku board with minimal O-information, run the following:

> `python train.py --dataset_name sudoku --quantity_name oinfo --number_training_steps 20000 --information_in_start 6 --information_in_end 6 --information_in_coefficient_start 1 --information_in_coefficient_end 10`


If you want to use the Wikipedia word frequency dataset that we used, download `results/enwiki-2023-04-13.txt` from [https://github.com/IlyaSemenov/wikipedia-word-frequency](https://github.com/IlyaSemenov/wikipedia-word-frequency) and put `enwiki-2023-04-13.txt` into this directory.  You can also download another dataset (such as from [the Google Web Trillion Word Corpus on Kaggle](https://www.kaggle.com/datasets/rtatman/english-word-frequency)) and pass its filename through the `--ngram_dataset_filename` flag.

