## A PyTorch implementation of skipgram W2V with negative sampling.

The ```utils.py``` file stores all the different functions that will be used
for I/O and other procedures on the corpus.

The ```SkipW2V.py``` file implements the W2V-skipgram architecture with negative sampling.

The ```main.py``` file is used for training the algorithm.

Example commands:

Training:

```
python ./main.py -c ../data/1bwc50000.txt -w 2 -min 0 -ll 50000 -tsize 10000 -nex 5 -opt sgd -e 15
```

Testing with the word "man":

```
python ./main.py --train_test test -words man
```

#### Articles used:

General papers & notes:

- [T. Mikolov et al. (2013)](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
- [X. Rong (2016)](https://arxiv.org/abs/1411.2738)
- [Stanford C.S. 224n Notes](https://web.stanford.edu/class/cs224n/archive/WWW_1617/lecture_notes/cs224n-2017-notes1.pdf)

Misc.:

- [C original implementation](https://github.com/tmikolov/word2vec/blob/master/word2vec.c)
- [On weights initialization](https://www.quora.com/How-are-vectors-initialized-in-word2vec-algorithm)

#### To do:

Optimize passes over the data.

- Implement subsampling when reading the corpora
- Discard words that do not meet the ```min_count```.
- Implement random batching of data
- Implement an independent testing suite?
