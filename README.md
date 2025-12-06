# Instance Decision-entropy Inhibition-driven Prior-guiding Weakly-supervised Clustering for Squamous Carcinoma Whole-slide Grading

## 🧔 Authors
- Pan Huang, _Member_, _IEEE_, Mingrui Ma, Yunchun Sun, Sukun Tian, _Member_, _IEEE_, Qin Jin, _Senior Member_, _IEEE_

## :fire: News
- [2025/12/28] Our manuscript will be submitted to _IEEE Transactions on Medical Imaging (IF 9.8)_.



## :rocket: Pipeline

Here's an overview of our **Instance Decision-entropy Inhibition-driven Prior-guiding Weakly-supervised Clustering (IDI-PWC)** method:

<img src="https://github.com/Baron-Huang/TicMIL/blob/main/Main_fig/Main_Frame_for_TicMIL.jpg" style="width:80%; height:80%;">



## :mag: TODO
<font color="red">**We are currently organizing all the code. Stay tuned!**</font>
- [x] training code
- [x] Evaluation code
- [x] Model code
- [ ] Pretrained weights
- [ ] Datasets





## 🛠️ Getting Started

To get started with **IDI-PWC**, follow the installation instructions below.

1.  Clone the repo

```sh
git clone https://github.com/Baron-Huang/TicMIL
```

2. Install dependencies
   
```sh
pip install -r requirements.txt
```

3. Training on Swin Transformer-S Backbone
```sh
sh TicMIL_main.sh
Modify: --abla_type sota --run_mode train --random_seed ${seed}
```

4. Evaluation
```sh
sh TicMIL_main.sh
Modify: --abla_type sota --run_mode test --random_seed ${seed}
```

5. Extract features for plots
```sh
sh TicMIL_main.sh
Modify: --abla_type sota --run_mode test --random_seed ${seed} --feat_extract
```

6. Interpretability plots
```sh
sh TicMIL_main.sh
Modify: --abla_type sota --run_mode test --random_seed ${seed} --bag_weight
```

## :postbox: Contact
If you have any questions, please contact [Dr. Pan Huang](https://scholar.google.com/citations?user=V_7bX4QAAAAJ&hl=zh-CN) (`panhuang@polyu.edu.hk`).
