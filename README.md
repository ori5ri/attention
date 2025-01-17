# Attention FPN: Reinforcing Semantic Information for Object Detection

## 🧾 Project Overview

Recently, many of the object detectors have been widely used Feature Pyramid Network (FPN) to improve features with semantic information of high levels, because of its simplicity and efficient design using top-down feature pyramid with lateral connections. Many studies have been conducted to find connections between features to supplement FPN for improvements of semantic information.

We propose AttentionFPN to efficiently fuse multi-scale features for information improvements without trying to find cross-scale connections manually. We design feature hierarchy using self-attention module to obtain attention map which is relationship between multi-scale features. Through this attention map, we reinforce the semantic information of features.

Experimental results with AttentionFPN show improvements using a range of other backbones, detectors and dataset compared to FPN.

---

## 💣 Problem Statement

Feature pyramid network (FPN) is a standard feature pyramid architecture. It is based on top-down feature hierarchy with lateral connections, adding two adjacent feature maps. This enables that strong semantic information of features from high-level is propagated to features of lower-levels.

FPN simply fuse feature maps of two layers from backbone without distinction. To improve more semantic information, various studies have been conducted how to fuse multi-scale features such as PANet, NAS-FPN, Auto-FPN, MLFPN, BiFPN and so on.

The key of designing feature pyramid ar- chitecture is determining connections of multi-level features and fuse feature maps in an optimal method. In this work, we develop a new feature pyramid network by introducing attention module, no need to consider connections between features.

---

## 💡 Novelty

1. This research propose AttentionFPN, a new connectionless and information-rich feature pyramid architecture using relations between multi-scale features.

2. It is a general approach that can be applied to various backbones, and can be complementary to various object detectors.

---

## ⚙ Method

1. Architecture

   ![architecture_v2](https://github.com/ori5ri/attention/assets/77871365/94c94b3b-f86d-4dc9-af29-22808e0f64e9)

   The overall architecture of AttentionFPN is shown in the figure. The Attention Module at the specific level n, denoted as $D_{n}$, receives the input features from backbone, and returns the feature map, denoted as $P_{n}$. It can be expressed as $$P_{n} = D_{n}(M_{2}, M{3}, ..., M_{k})$$

2. Attention Module

   ![architecture_p5](https://github.com/ori5ri/attention/assets/77871365/cc860f3c-22ca-489a-87fa-c73367995266)


   Attention Module of $D_{5}$ is shown in the figure. All features serves as inputs of Attention Block to extract the global context. ${R_{n2}, R_{n3}, R_{n4}, R_{n5}}$ is extracted by aggregating original features with attention map which is the output of the Attention Block. The output of the Attention Module is generated by summation of ${R_{n2}, R_{n3}, R_{n4}, R_{n5}}$.

3. Attention Block

   | ![attention_module_CM](https://github.com/ori5ri/attention/assets/77871365/2803707a-b7bc-4caa-867f-30d793f54cdf) | ![attention_module_T](https://github.com/ori5ri/attention/assets/77871365/333ebc5d-8e4d-4acd-abcd-351162193abf) |
   | --------------------------------------------------- | -------------------------------------------------- |

   We propose attention block that improves GCNet block to obtain relationships considering entire feature maps. Attention Block is consisted with two processes, context modeling and transfromation.

   Context modeling of the proposed attention block extracts the global context that represents the features of different resolution. The features have different information according to the levels.

   Transformation performs after context modeling. To obtain attention map, the global context goes through $1\times1$ convolution layer, Norm, ReLU.

---

## 🌎 Environment

1.  install mmdetection

2.  append

```
attention/mmdetection/mmdet/models/necks/attention_module/
```

and

```
attention/mmdetection/mmdet/models/necks/attention.py
```

to

```
attention/mmdetection/mmdet/models/necks/
```

---

## Reference

https://github.com/open-mmlab/mmdetection

---

## 👼 Collaborator

```
⭐️ Minju Ro
- Chungang University
- https://github.com/ori5r2
- https://github.com/ori5ri

⭐️ Sehwan Joo
- Chungang University
- https://github.com/SeHwanJoo

⭐️ Team GitHub
- https://github.com/vllab-under

```
