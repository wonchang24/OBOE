# Look Before You Leap: Confirming Edge Signs in Random Walk with Restart for Personalized Node Ranking in Signed Networks
This repository provides a reference implementation of *OBOE* as described in the following paper:
> Look Before You Leap: Confirming Edge Signs in Random Walk with Restart for Personalized Node Ranking in Signed Networks<br>
> Wonchang Lee, Yeon-Chang Lee, Dongwon Lee and Sang-Wook Kim<br>
> 44th Int'l ACM SIGIR Conf. on Research and Development in Information Retrieval (ACM SIGIR 2021)<br>

### Authors
- Wonchang Lee (wonchang24@hanyang.ac.kr)
- Yeon-Chang Lee (lyc0324@hanyang.ac.kr)
- Dongwon Lee (dongwon@psu.edu)
- Sang-Wook Kim (wook@hanyang.ac.kr)

### Input
The input files should be saved in `datasets/` folder. The structure of the input file is the following:

```node_id1 node_id2 sign```

Node ids start from *0* to *N-1* (*N* is the number of nodes in the graph).

### Output
The output files are saved in `results/` folder. 
It includes parameter and accuracies in top-k and bottom-k tasks of *OBOE*.

### Arguments

```
--dataset                 Dataset name. (default: "wiki")
--func                    Select a function of (extract, predict, run). (default: "run")
--p_thres                 Positive threshold (beta_+). (default: 1.0)
--n_thres                 Negative threshold (beta_-). (default: 0.6)
--c                       Restart probability. (default: 0.4)
--m_iter                  Number of maximum iterations. (default: 50) 
```

### Procedure
1. Extract features of train dataset.
2. Predict FExtra scores between two nodes using features.
3. Run OBOE using FExtra scores.

### Basic Usage
```
python ./src/main.py --dataset wiki --func extract
python ./src/main.py --dataset wiki --func predict
python ./src/main.py --dataset wiki --func run --p_thres 1.0 --n_thres 0.6 --c 0.4 --m_iter 50
```

### Requirements
The code has been tested running under Python 3.9.4. The required packages are as follows:

- ```numpy == 1.20.1```
- ```pandas == 1.2.4```
- ```scikit-learn == 0.24.2```
- ```scipy == 1.6.2```
- ```tqdm == 4.59.0```

### Cite
We encourage you to cite our paper if you have used the code in your work. You can use the following BibTex citation:
```
@inproceedings{lee21sigir,
  author   = {Wonchang Lee and Yeon{-}Chang Lee and Dongwon Lee and Sang{-}Wook Kim},
  title     = {Look Before You Leap: Confirming Edge Signs in Random Walk with Restart for Personalized Node Ranking in Signed Networks},
  booktitle = {International ACM SIGIR Conference on Research and Development in Information Retrieval (ACM SIGIR 2021)},      
  year      = {2021}
}
```
