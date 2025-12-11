# Instance Decision-entropy Inhibition-optimized Prior-guiding Active Explanation Clustering for High-effect Pathology-grade Indicator Learning
## 🧔 Authors
- Pan Huang, _Member_, _IEEE_, Mingrui Ma, Yifang Ping, Sukun Tian, Qin Jin, _Senior Member_, _IEEE_

## :fire: News
- [2025/12/29] Our manuscript will be submitted to _IEEE Transactions on Medical Imaging (IF 9.8)_.



## :rocket: Pipeline

Here's an overview of our **Instance Decision-entropy Inhibition-optimized Prior-guiding Active Explanation Clustering (IDI-PAEC)** method:

<img src="https://github.com/Baron-Huang/TicMIL/blob/main/Main_fig/Main_Frame_for_TicMIL.jpg" style="width:80%; height:80%;">



## :mag: TODO
<font color="red">**We are currently organizing all the code. Stay tuned!**</font>
- [x] training code
- [x] Evaluation code
- [x] Model code
- [ ] Pretrained weights
- [ ] Datasets





## 🛠️ Getting Started

To get started with **IDI-PAEC**, follow the installation instructions below.

1.  Clone the repo

```sh
git clone https://github.com/Baron-Huang/IDI-PWC
```

2. Install dependencies
   
```sh
pip install -r requirements.txt
```

3. Training on Swin Transformer-S Backbone
```sh
sh IDI_PAEC_main.sh
Modify: --abla_type sota --run_mode train --random_seed ${seed}
```

4. Evaluation
```sh
sh IDI_PAEC_main.sh
Modify: --abla_type sota --run_mode test --random_seed ${seed}
```

5. Extract features for plots
```sh
sh IDI_PAEC_main.sh
Modify: --abla_type sota --run_mode test --random_seed ${seed} --feat_extract
```

6. Interpretability plots
```sh
sh IDI_PAEC_main.sh
Modify: --abla_type sota --run_mode test --random_seed ${seed} --bag_weight
```

## :postbox: Contact
If you have any questions, please contact [Dr. Pan Huang](https://scholar.google.com/citations?user=V_7bX4QAAAAJ&hl=zh-CN) (`panhuang@polyu.edu.hk`).
