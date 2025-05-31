# Recent Research by Our Visual Perception Group

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

For further works about skeleton-based uman action understanding, please visit this page [skeleton_action_awesome](https://hongsong-wang.github.io/skeleton_action_awesome/).

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

### 2. 人体动作生成
