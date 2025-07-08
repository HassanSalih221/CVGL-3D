# SkyLink: Unifying Street-Satellite Geo-Localization via UAV-Mediated 3D Scene Alignment
Official implementation of 2025 ACM'MM UAV Challenging paper (https://codalab.lisn.upsaclay.fr/competitions/22073), TeamName: XMUSmart.

## News
- **2025.07.07: Comming Soon! Codes will be released upon the paper's publicationd.**

## Description 📜
Cross-view geo-localization aims at establishing location correspondences between different viewpoints. We focus on robust feature retrieval under viewpoint variation and propose the novel SkyLink method. Meanwhile, we integrate the 3D scene information constructed from multi-scale UAV images as a bridge between street and satellite viewpoints, and perform feature alignment through self-supervised and cross-view contrastive learning.

## Framework 🖇️
<td style="text-align: center"><img src="./figures/overview.jpg" alt="Framework" width="850"></td>

## Requirements
### Installation
Create a conda environment and install dependencies:
```bash
conda create -n cvgl python=3.10
conda activate cvgl

# Install the according versions of torch and torchvision
conda install pytorch torchvision cudatoolkit
```

## Contact
If you have any question about this project, please feel free to contact hyzhang@stu.xmu.edu.cn.
