# Recent Research by Our Visual Perception Group (视觉感知小组的最新研究成果，[个人主页](https://cs.seu.edu.cn/hongsongwang/main.htm)) ([Paper Portal of Top Conferences in Computer Vision and Machine Learning](https://hongsong-wang.github.io/CV_Paper_Portal/), [计算机视觉与机器学习顶会论文摘要](https://hongsong-wang.github.io/CV_Paper_Portal/))

## Human Motion Understanding and Generation

### 1. Human Action Understanding
#### [USDRL: Unified Skeleton-Based Dense Representation Learning with Multi-Grained Feature Decorrelation](https://arxiv.org/html/2412.09220v2), [Code](https://github.com/wengwanjiang/USDRL), [AAAI2025](https://ojs.aaai.org/index.php/AAAI/article/view/32899)

We propose a simple yet effective method named Unified Skeleton-based Dense Representation Learning (USDRL) that learns dense representations through multi-grained feature decorrelation, demonstrating the feasibility of feature decorrelation in skeleton-based dense representations learning.

#### [Dual Conditioned Motion Diffusion for Pose-Based Video Anomaly Detection](https://arxiv.org/html/2412.17210v2), [Code](https://github.com/guijiejie/DCMD-main), [AAAI2025](https://ojs.aaai.org/index.php/AAAI/article/view/32829)

We introduce a novel framework that seamlessly integrates reconstruction-based and prediction-based methods for video anomaly detection, leveraging the strengths of both approaches. We propose a Dual Conditioned Motion Diffusion (DCMD), which incorporates both conditioned motion and conditioned embedding in a diffusion-based model.

### [Zero-Shot Skeleton-based Action Recognition with Dual Visual-Text Alignment](https://arxiv.org/abs/2409.14336)

We propose an Dual Visual-Text Alignment (DVTA), a novel zero-shot approach for skeleton-based action recognition. The method enhances generalization to unseen classes by jointly optimizing two modules: Direct Alignment (DA) and Augmented Alignment (AA).

### [Frequency-Guided Diffusion Model with Perturbation Training for Skeleton-Based Video Anomaly Detection](https://arxiv.org/abs/2412.03044), [Code](https://github.com/Xiaofeng-Tan/FGDMAD-Code)

We introduce a perturbation-based training paradigm for diffusion models to improve robustness against unseen normal motions in open-set scenarios. We introduce a frequency-guided denoising process to separate the global and local motion information into low-frequency and high-frequency components, prioritizing global reconstruction for effective anomaly detection.

### [Training-Free Zero-Shot Temporal Action Detection with Vision-Language Models](https://arxiv.org/abs/2501.13795), [Code](https://github.com/Chaolei98/FreeZAD), [Project](https://chaolei98.github.io/FreeZAD/)

To the best of our knowledge, we are the first to investigate the problem of training-free ZSTAD. We propose FreeZAD, a training-free approach for ZSTAD, which effectively leverages the generalization capabilities of ViL models to detect unseen activities. We introduce a simple yet effective TTA method that extends FreeZAD and enhances its performance by enabling adaptation to a video sequence without supervision.

### [Region-aware Image-based Human Action Retrieval with Transformers](https://arxiv.org/html/2407.09924v2), [CVIU2024](https://www.sciencedirect.com/science/article/abs/pii/S1077314224002832)

We empirically study the neglected task of image-based human action retrieval, and establish new benchmarks and important baselines to promote research in this field. We introduce an efficient Region-aware Image-based human Action Retrieval with Transformers (RIART), which leverages both person-related and contextual object cues, and employs a fusion transformer module for human action retrieval.

For further works about skeleton-based uman action understanding, please visit the [skeleton_action_awesome](https://hongsong-wang.github.io/skeleton_action_awesome/), [skeleton_action_awesome](https://github.com/hongsong-wang/skeleton_action_awesome/).

### 2. Human Motion Generation

## [ReAlign: Bilingual Text-to-Motion Generation via Step-Aware Reward-Guided Alignment](https://arxiv.org/abs/2505.04974), [Project](https://wengwanjiang.github.io/ReAlign-page/)

Our first contribution lies in the introduction of a pioneering bilingual text-to-motion dataset, BiHumanML3D, accompanied by a corresponding bilingual text-to-motion method, Bilingual Motion Diffusion (BiMD). To address the scarcity of bilingual text-motion datasets, we extend the widely used text-to-motion dataset, HumanML3D, by introducing its bilingual version, BiHumanML3D. Specifically, a multi-stage translation pipeline based on large language models and manual correction is designed to ensure high-quality annotations and accurate semantic translations.

### [SoPo: Text-to-Motion Generation Using Semi-Online Preference Optimization](https://arxiv.org/abs/2412.05095), [Code](https://github.com/Xiaofeng-Tan/SoPO), [Project](https://sopo-motion.github.io/)

Our first contribution is the explicit revelation of the limitations of both online and offline DPO. Online DPO is constrained by biased sampling, resulting in high-preference scores that limit the preference gap between preferred and unpreferred motions. Meanwhile, offline DPO suffers from overfitting due to limited labeled preference data, especially for unpreferred motions, leading to poor generalization. We propose a novel and effective SoPo method to address these limitations. SoPo trains models on “semi-online” data pairs that incorporate high-quality preferred motions from offline datasets alongside diverse unpreferred motions generated dynamically.

### [PAMD: Plausibility-Aware Motion Diffusion Model for Long Dance Generation](https://www.arxiv.org/abs/2505.20056), [Code](https://github.com/mucunzhuzhu/PAMD), [Project](https://mucunzhuzhu.github.io/PAMD-page/)

We introduce Plausibility-Aware Motion Diffusion (PAMD), a diffusion-based framework for music-to-dance generation. PAMD generates dances that are both musically aligned and physically realistic. To ensure realistic dance generation, we design Plausible Motion Constraint (PMC), which uses NDFs to model plausible human poses on a continuous manifold, the first application of NDFs in music-to-dance generation.

### [Flexible Music-Conditioned Dance Generation with Style Description Prompts](https://arxiv.org/abs/2406.07871)

We propose flexible music-conditioned Dance Generation with Style Description Prompts (DGSDP) which deliberately incorporates style description prompts to enhance dance generation with styles. We introduce Music-Conditioned Style-Aware Diffusion (MCSAD) which primarily comprises a Transformer-based dance generation network and a novel style modulation module. 

# Below is the Chinese version (以下为对应的中文版本)

## 人体动作理解与生成

### 1. 人体行为理解

#### [USDRL：基于骨架的统一密集表征学习与多粒度特征去相关](https://arxiv.org/html/2412.09220v2), [代码](https://github.com/wengwanjiang/USDRL), [AAAI2025](https://ojs.aaai.org/index.php/AAAI/article/view/32899)

我们提出了一种简单而有效的方法，称为统一骨架密集表征学习（USDRL），通过多粒度的特征去相关方式学习密集表征，验证了在骨架基础上进行密集表征学习中，特征去相关方法的可行性。

#### [基于人体姿态的视频异常检测的双条件运动扩散模型](https://arxiv.org/html/2412.17210v2), [代码](https://github.com/guijiejie/DCMD-main), [AAAI2025](https://ojs.aaai.org/index.php/AAAI/article/view/32829)

我们提出了一个新颖的框架，将基于重建和基于预测的视频异常检测方法无缝结合，充分发挥二者优势。我们引入了双条件运动扩散模型（DCMD），将条件运动与条件嵌入共同引入扩散模型中进行建模。

### [具有动作-文本双重对齐的零样本骨架动作识别](https://arxiv.org/abs/2409.14336)

我们提出了一种新颖的零样本骨架动作识别方法：双重动作-文本对齐（DVTA）。该方法通过联合优化直接对齐（DA）和增强对齐（AA）两个模块，从而提升对未见类别的泛化能力。

### [面向骨架视频异常检测的带扰动训练的频域引导扩散模型](https://arxiv.org/abs/2412.03044), [代码](https://github.com/Xiaofeng-Tan/FGDMAD-Code)

我们提出了一种用于扩散模型的扰动训练范式，以增强在开放集场景下对未见正常动作的鲁棒性。该方法引入频域引导的去噪过程，将全局与局部运动信息分别作为低频与高频成分进行建模，从而优先重建全局动作以提升异常检测效果。

### [基于视觉语言模型的零样本免训练时间动作检测](https://arxiv.org/abs/2501.13795), [代码](https://github.com/Chaolei98/FreeZAD), [项目主页](https://chaolei98.github.io/FreeZAD/)

我们首次探讨无需训练的零样本时间动作检测（ZSTAD问题。我们提出一种无需训练的ZSTAD方法，有效利用ViL模型的泛化能力来检测未见动作。此外，我们提出了一种简单有效的测试时自适应策略，在无监督条件下增强该方法在视频序列上的适应能力。

### [基于区域感知Transformer的人体动作图像检索](https://arxiv.org/html/2407.09924v2), [CVIU2024](https://www.sciencedirect.com/science/article/abs/pii/S1077314224002832)

我们系统研究了图像级人体动作检索这一长期被忽视的任务，建立了新的基准和重要基线，以推动该方向研究。我们提出了基于区域感知Transformer的人体动作图像检索方法，结合人物区域与上下文物体线索，并采用融合Transformer模块以提升检索性能。

如需了解更多关于基于骨架的人体动作理解研究，请访问[skeleton_action_awesome](https://hongsong-wang.github.io/skeleton_action_awesome/), [skeleton_action_awesome](https://github.com/hongsong-wang/skeleton_action_awesome/).

### 2. 人体动作生成

## [中英双语文本驱动的人体动作生成](https://arxiv.org/abs/2505.04974), [项目主页](https://wengwanjiang.github.io/ReAlign-page/)

我们的第一个贡献是提出了首个双语文本到动作数据集 BiHumanML3D，并配套提出双语动作扩散模型 BiMD。为了解决双语数据稀缺的问题，我们扩展了广泛使用的 HumanML3D 数据集，构建了其双语版本 BiHumanML3D。具体来说，我们设计了一种结合大语言模型翻译与人工校对的多阶段流程，以确保高质量的语义对齐。

### [基于半在线偏好优化的文本驱动动作生成](https://arxiv.org/abs/2412.05095), [代码](https://github.com/Xiaofeng-Tan/SoPO), [项目主页](https://sopo-motion.github.io/)

我们首先明确指出在线偏好优化受限于偏倚采样，导致高偏好得分之间差距过小；而离线偏好优化由于负样本匮乏，易过拟合，泛化能力差。为此我们提出SoPo方法，通过半在线偏好训练方式，将高质量的离线正样本与动态生成的多样负样本相结合，从而提升模型的生成能力与偏好区分能力。

### [面向长时舞蹈生成的可信度感知动作扩散模型](https://www.arxiv.org/abs/2505.20056), [代码](https://github.com/mucunzhuzhu/PAMD), [项目主页](https://mucunzhuzhu.github.io/PAMD-page/)

我们提出了PAMD，一种面向音乐驱动舞蹈生成的扩散模型，能够生成既符合音乐节奏又符合人体物理可行性的舞蹈动作。为提升生成舞蹈的可信度，我们设计了可信动作约束（PMC），首次在音乐驱动动作生成中引入了NDF（神经距离场）以建模人类动作的连续可行性流形。

### [基于风格提示的灵活音乐驱动舞蹈生成](https://arxiv.org/abs/2406.07871)

我们提出了一种支持风格提示的灵活音乐驱动舞蹈生成方法，通过引入风格描述提示词增强动作的风格表现。提出的MCSAD框架包含一个基于Transformer的舞蹈生成网络和风格调制模块。

<div style="background-color: #f6f8fa; padding: 1em; border-radius: 6px; font-family: monospace; white-space: pre-wrap;">
<code>
This webpage is protected by copyright laws. Without the written permission of the owner of this webpage, no individual or organization shall use the content of this webpage in any form. If there is a need to reprint the content of this webpage for non-commercial purposes such as learning, research, or personal sharing, the source must be clearly indicated as "Content sourced from [https://github.com/hongsong-wang/Perception-Lab/]". The content must be kept intact, and no alteration or distortion of the original text is allowed. The owner of this webpage reserves the right to pursue legal liability for any unauthorized use of the content of this webpage. If you find these works useful, please cite the above works.
</code>
</div>
