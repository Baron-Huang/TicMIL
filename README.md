# Tumor Prior-Guided Instance Clustering Driving-Multi-instance Learning for Squamous Cell Carcinoma Whole-slide Grading
Pan Huang, Member, IEEE, Mingrui Ma, Yunchun Sun, Sukun Tian, Member, IEEE, Ziyue Xu, Senior Member, IEEE, and Jing Qin, Senior Member, IEEE

Our manuscript is in the peer review, and we will completely share the dataset and code after the peer review.

# Introduction
Accurate pathological grading is vital for the diagnosis, treatment, and prognosis of squamous cell carcinoma (SCC), which is expected to focus on the tumor region. However, existing relational multi-instance learning (MIL) approaches overly learn nontumor instances, resulting in inaccurate SCC pathology grading patterns. Inspired by this critical issue, this study proposes an end-to-end tumor prior-guided instance clustering driving multi-instance learning, i.e., TicMIL, with three fold ideas: first, we develop an end-to-end instance inhibition parallel learning algorithm that is able to build the precise feature representations for the SCC pathology grading task; second, we incorporate the tumor priori guiding instance clustering into the network for addressing the fuzzy labeling of cluster centers, as well as enhancing the model’s representation ability to the tumor instances; third, we model the imbalance between tumor and nontumor instances into the relational MIL, thus decreasing the decision entropy value of model for enlarging the learning margin to obtain high grading accraucy. We employ extensive experiments on two SCC datasets, i.e., AMU-CSCC and AMU-LSCC, which show that the TicMIL outperforms other MIL SOTAs. It is able to represent the SCC pathology grading patterns with high accuracy, proving its potential for clinical utility.


