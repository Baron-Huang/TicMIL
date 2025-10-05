# Tumor Priori Knowledge-Guided Instance Clustering-Driving Multi-instance Learning for Squamous Cell Carcinoma Whole-slide Grading

## 🧔 Authors
- Pan Huang, _Member_, _IEEE_, Mingrui Ma, Yunchun Sun, Sukun Tian, _Member_, _IEEE_, Ziyue Xu, _Senior Member_, _IEEE_, Qin Jin, _Senior Member_, _IEEE_

## :fire: News

- [2025/10/02] Our manuscript was currently "In Peer Review" in _IEEE Transactions on Circuits and Systems for Video Technology (IF 11.2)_.
- [2025/06/27] Our manuscript was submitted to _IEEE Transactions on Circuits and Systems for Video Technology (IF 11.2)_.



## :rocket: Pipeline

Here's an overview of our **Tumor Priori Knowledge-Guided Instance Clustering-Driving Multi-instance Learning (TicMIL)** method:

<img src="https://github.com/Baron-Huang/TicMIL/blob/main/Main_fig/Main_Frame_for_TicMIL.jpg" style="width:80%; height:80%;">



## :mag: TODO
<font color="red">**We are currently organizing all the code. Stay tuned!**</font>
- [x] training code
- [x] Evaluation code
- [x] Model code
- [ ] Pretrained weights
- [ ] Datasets





## 🛠️ Getting Started

To get started with TicMIL, follow the installation instructions below.

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
If you have any questions, please contact [Dr. Pan Huang](https://scholar.google.com/citations?user=V_7bX4QAAAAJ&hl=zh-CN) (`mrhuangpan@163.com or pan.huang@polyu.edu.hk`).
