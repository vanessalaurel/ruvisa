# Skin Analyzer — Expanded Materials and Method

The skin lesion analyzer in Ruvisa is implemented as a multi-stage computer-vision pipeline that transforms a single face photograph into a structured **7-dimensional concern vector** describing the user's current skin condition. The pipeline combines four sequential models: **face parsing**, **lesion detection**, **severity classification**, and **wrinkle segmentation**. These stages are not used independently; instead, they are coupled so that low-level visual evidence (lesions and wrinkles) is progressively converted into region-level summaries, concern-level scores, and a final whole-face representation suitable for personalization and recommendation.

---

## 1. Overall Pipeline

The complete pipeline proceeds in four stages:

1. **Face parsing (BiSeNet):** segment the face and derive geometric facial regions.
2. **Lesion detection (YOLOv8m):** localize lesion candidates on the input face image.
3. **Severity classification (ResNet-18):** classify each detected lesion crop into one of four severity levels.
4. **Wrinkle segmentation (UNet):** estimate wrinkle masks and wrinkle severity by region.

The outputs of these stages are fused into:

- per-lesion records: class, confidence, severity, and face region,
- region-level lesion summaries,
- wrinkle coverage and wrinkle severity summaries,
- a **7-dimensional concern vector** in the shared concern space used by the downstream recommendation module.

The final concern dimensions are:

$$
\mathbf{u} = [\text{acne},\ \text{comedonal\_acne},\ \text{pigmentation},\ \text{acne\_scars\_texture},\ \text{pores},\ \text{redness},\ \text{wrinkles}]
$$

These model families were selected for practical reasons rather than novelty alone. BiSeNet is designed to preserve both spatial detail and contextual information at real-time speed [11], which is suitable for facial masking and region construction. YOLOv8m is used because the Ultralytics detector is lightweight, fast, and strong on small-object localization in a single-stage pipeline [13]. ResNet-18 provides a stable and comparatively lightweight residual classifier for crop-level severity prediction [14]. U-Net is used for wrinkle segmentation because its encoder-decoder structure with skip connections is well suited to thin, spatially localized structures [15]. Thus, the skin analyzer is not only a detector, but also a feature-construction module that converts raw image evidence into a recommendation-ready state vector.

---

## 2. Datasets and Preprocessing

### 2.1 Acne detection dataset

Lesion localization is trained using the **Roboflow Acne Detection dataset (v2)** [35]. The original dataset provides bounding-box annotations over six lesion classes:

- Acne
- Nodule
- Blackhead
- Whitehead
- Acne_scars
- Flat_wart

During preprocessing, semantically equivalent Roboflow classes such as **Pustule**, **Papular**, and **Pimples-acne** are remapped into the unified **Acne** class to reduce label fragmentation and improve detector consistency. The dataset follows the standard Roboflow split [35]:

- **80% training**
- **10% validation**
- **10% testing**

All images are resized to **1280 x 1280** pixels during training and evaluation in order to preserve small lesion detail.

### 2.2 Severity classification dataset

Lesion severity is trained using the **ACNE04** dataset, which contains lesion crops annotated into four ordinal severity levels [16]:

- `level0` = none
- `level1` = mild
- `level2` = moderate
- `level3` = severe

The dataset also contains lesion-type annotations (non-inflammatory, inflammatory, and cystic), but the severity model in this work uses only the four-level severity labels. The predefined index files (`NNEW_trainval_*.txt`, `NNEW_test_*.txt`) define five cross-validation folds. Within the train/validation portion, the data is further split into **90% training** and **10% validation** using stratified sampling to preserve severity-class balance. The four acne-severity levels in ACNE04 are grounded in the Hayashi grading criterion, which maps inflammatory lesion counts to mild, moderate, severe, and very severe categories [16], [17].

For downstream scoring, the four severity labels are mapped into a normalized scalar:

$$
\text{severity\_score}_{\text{lesion}} \in \{0,\; \tfrac{1}{3},\; \tfrac{2}{3},\; 1\}
$$

corresponding to:

$$
\{ \text{level0},\ \text{level1},\ \text{level2},\ \text{level3} \}
\mapsto
\{ 0.0,\ 0.333,\ 0.667,\ 1.0 \}
$$

This normalized scale is **defined in this work** rather than copied from ACNE04 directly. A linear mapping from the ordinal labels to $\{0,\tfrac{1}{3},\tfrac{2}{3},1\}$ preserves the rank order of severity while converting the labels into a bounded continuous quantity in $[0,1]$. This makes later averaging and concern-level aggregation mathematically straightforward without implying a more precise clinical interval scale than the data actually provides.

### 2.3 Wrinkle segmentation dataset

The wrinkle analysis branch uses the **Flickr-Faces-HQ (FFHQ)** dataset, a collection of 70,000 high-quality face images at **1024 x 1024** resolution [18]. From this corpus, **1,000 images** with manually annotated binary wrinkle masks are used for evaluation and fine-tuning support.

The wrinkle model follows the two-stage weak-supervision strategy proposed by Moon et al. [19]:

1. **Texture-map pretraining**
2. **Wrinkle-mask fine-tuning**

In Stage 1, the model is pretrained to predict texture-enhanced weak labels that approximate wrinkle locations without requiring dense manual masks. In the original Moon et al. pipeline, masked texture maps are used as weak supervision to emphasize fine facial line structure [19]. In the present implementation, this idea is instantiated using a simple high-pass texture map:

$$
T(x,y) = \left| I_{\text{gray}}(x,y) - (G_{\sigma} * I_{\text{gray}})(x,y) \right|
$$

where $I_{\text{gray}}$ is the grayscale face image and $G_{\sigma}$ is a Gaussian blur with $\sigma = 3$. This equation is used because wrinkles are high-frequency structures: subtracting a blurred low-frequency version of the face image leaves edges, folds, and fine texture cues that are more informative for wrinkle localization than raw intensity alone. Therefore, while the exact equation here is an implementation choice in Ruvisa, its motivation is consistent with the texture-map-based weak supervision strategy of Moon et al. [19].

In Stage 2, the pretrained model is fine-tuned on manually annotated wrinkle masks so that it learns to distinguish genuine wrinkles from other texture artifacts such as pores, moles, shadows, or illumination noise.

The pretrained weights `stage2_unet.pth` from Moon et al. are used directly in the deployed Ruvisa pipeline [19].

### 2.4 Training-time preprocessing and augmentation

#### Lesion detection

YOLOv8's built-in augmentation pipeline is applied during training:

- mosaic composition (`p = 1.0`, disabled in the final 10 epochs),
- horizontal flip (`p = 0.5`),
- HSV jitter (`H = 0.015`, `S = 0.7`, `V = 0.4`),
- random scale (`±50%`),
- translation (`±10%`),
- random erasing (`p = 0.4`),
- RandAugment.

These augmentations are used to improve robustness to viewpoint, scale, and lighting variability in consumer face photographs.

#### Severity classification

Severity-classification training uses the following transforms:

- `Resize(256)`
- `RandomResizedCrop(224, scale = 0.8–1.0)`
- `RandomHorizontalFlip`
- `ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2)`
- `GaussianBlur(p = 0.2)`

All crops are normalized using ImageNet statistics:

$$
\mu = [0.485,\ 0.456,\ 0.406], \qquad
\sigma = [0.229,\ 0.224,\ 0.225]
$$

---

## 3. Model Architecture

### 3.1 Stage 1 — Face parsing (BiSeNet)

The first stage uses **BiSeNet** with a ResNet-18 backbone for semantic face parsing [11]. The parser predicts **19 facial classes** from the CelebAMask-HQ labeling scheme [12]. In the Ruvisa pipeline, the output is simplified into a face-only mask by retaining:

- **skin class** (`class 1`)
- **nose class** (`class 10`)

The rest of the image is masked out. This face mask is then geometrically partitioned into four coarse acne-analysis regions:

- **Forehead:** top 32% of the face bounding box
- **Left cheek:** 32–78% vertical band, left half
- **Right cheek:** 32–78% vertical band, right half
- **Chin:** bottom 22% of the face bounding box

These regions are used for lesion-to-region assignment and regional burden summaries. This stage is needed because later recommendation uses concern-level facial summaries rather than raw segmentation masks.

### 3.2 Stage 2 — Lesion detection (YOLOv8m)

Lesion localization is performed using **YOLOv8-Medium**, a single-stage, anchor-free object detector composed of:

- a **CSPDarknet backbone**,
- a **PAFPN neck**,
- three **decoupled detection heads** operating at different feature scales.

The model is initialized from **COCO-pretrained weights** and fine-tuned on the acne dataset at **1280 x 1280** resolution to better preserve small lesion targets. YOLOv8 is used here because the detector combines efficient multi-scale feature extraction with real-time inference, making it a reasonable choice for a deployed skincare application that must localize small lesions without introducing a heavy two-stage detection pipeline [13].

At inference, the detector outputs:

- bounding box coordinates,
- confidence score,
- lesion class label.

Each detection is then passed to the severity classifier.

### 3.3 Stage 3 — Severity classification (ResNet-18)

Lesion severity is estimated using **ResNet-18** pretrained on ImageNet [14]. The final classification head is replaced with a 4-way linear classifier:

$$
\text{fc}: \mathbb{R}^{512} \rightarrow \mathbb{R}^{4}
$$

The input to the classifier is the cropped lesion patch resized to **224 x 224**. The model predicts one of the four ordinal severity levels (`level0`-`level3`), which is then converted to the normalized severity score described in Section 2.2.

This stage adds semantic intensity information to each lesion proposal and is critical because the downstream system uses lesion severity rather than lesion count alone. ResNet-18 was chosen because residual learning provides strong image-classification performance with relatively low computational overhead, which is appropriate for repeated crop-level inference in a multi-stage pipeline.

### 3.4 Stage 4 — Wrinkle segmentation (UNet)

Wrinkle analysis is handled by a separate **UNet** segmentation branch with **4-channel input**:

- RGB image channels
- 1 texture-map channel

The model performs binary wrinkle segmentation at **1024 x 1024** resolution. The encoder follows a standard contraction path:

$$
64 \rightarrow 128 \rightarrow 256 \rightarrow 512 \rightarrow 512
$$

using repeated `Conv(3x3) -> BatchNorm -> ReLU` blocks. The decoder uses bilinear upsampling and skip connections from the encoder.

The texture map given to the model is:

$$
T(x,y)=\left| I_{\text{gray}}(x,y) - (G_{\sigma} * I_{\text{gray}})(x,y) \right|
$$

which guides the network toward fine wrinkle-like structures. U-Net is suitable here because skip connections help preserve thin and spatially localized details during upsampling, which is important for wrinkle segmentation [15].

---

## 4. Inference-Time Region Construction and Scoring

### 4.1 Lesion-to-region assignment

After detection, each lesion is assigned to a facial region using the **centroid** of its bounding box. If the bounding-box center lies inside a given region mask, the lesion is assigned to that region. Formally, for a bounding box:

$$
b = (x_1, y_1, x_2, y_2)
$$

the centroid is:

$$
(c_x, c_y) = \left( \frac{x_1 + x_2}{2},\; \frac{y_1 + y_2}{2} \right)
$$

The lesion is assigned to the first region mask for which the centroid lies inside the valid face region. This provides a simple and computationally efficient region assignment rule. The centroid equation itself is standard Euclidean geometry rather than a learned or cited model-specific formula.

### 4.2 Per-lesion severity score

Each detected lesion receives:

- a categorical severity label (`level0` to `level3`),
- a normalized scalar severity score:

$$
s_i \in \{0,\; \tfrac{1}{3},\; \tfrac{2}{3},\; 1\}
$$

These per-lesion scores are then aggregated spatially and semantically.

### 4.3 Regional ROI calculation

For each acne-analysis region $r$ (forehead, left cheek, right cheek, chin), the system computes:

- the number of lesions in that region,
- the regional ROI score.

In this implementation, **ROI** refers to a **regional burden indicator**, not to physical lesion area. It is calculated as the mean lesion severity score within the region:

$$
\text{ROI}_r = \frac{1}{n_r} \sum_{i=1}^{n_r} s_i
$$

where:

- $n_r$ = number of lesions assigned to region $r$
- $s_i$ = normalized severity score of lesion $i$

If no lesions are assigned to a region, then:

$$
\text{ROI}_r = 0
$$

Thus, ROI summarizes how severe the lesions are **on average** in each region. This formula is **defined in this work** as a reporting-oriented regional burden score. A mean is used instead of a raw sum so that regions with many mild lesions and regions with fewer severe lesions can be compared on a common severity scale without simply rewarding larger lesion counts.

### 4.4 Wrinkle region scoring

Wrinkles are not treated as discrete object detections. Instead, the wrinkle branch computes a **binary wrinkle mask** over the face and measures wrinkle coverage per region.

For each wrinkle region $r$, the wrinkle coverage percentage is:

$$
\text{wrinkle\_pct}_r = 100 \times \frac{\text{wrinkle pixels in region } r}{\text{face pixels in region } r}
$$

The implemented wrinkle regions are defined as fixed fractions of the face bounding box:

- **Forehead:** top 32%
- **Under-eye:** 32–48% vertically, central 50% horizontally
- **Crow's feet:** 32–48% vertically, outer left and right 25%
- **Nasolabial:** 48–70% vertically, central 60%
- **Perioral:** bottom 30%

Wrinkle coverage is converted into ordinal wrinkle severity using the thresholds:

$$
\text{none} < 0.8\%, \qquad
\text{mild} < 2.0\%, \qquad
\text{moderate} < 4.0\%, \qquad
\text{severe} \ge 4.0\%
$$

The wrinkle concern value used downstream is not based on the ordinal label, but on direct percentage scaling:

$$
\text{wrinkle\_concern} = \min\left(1.0,\; \frac{\text{wrinkle\_pct}}{5.0}\right)
$$

The wrinkle coverage formula is a standard pixel-area ratio derived from the segmentation mask. By contrast, the ordinal thresholds and the downstream scaling

$$
\text{wrinkle\_concern} = \min\left(1.0,\; \frac{\text{wrinkle\_pct}}{5.0}\right)
$$

are **engineering definitions introduced in this work**. They were chosen to convert a small percentage-valued wrinkle measurement into a bounded concern value that is numerically compatible with the lesion-derived concern dimensions used later by the recommender. The cap at 1.0 prevents high wrinkle coverage from dominating the full vector.

### 4.5 Concern-vector construction

After lesion detection and wrinkle estimation, the system builds the final user concern vector. Detected lesion classes are mapped into the shared concern space as follows:

| Detection class | Concern dimension |
| --- | --- |
| acne, papule, pustule, nodule, cyst | acne |
| blackhead, whitehead, comedone | comedonal_acne |
| dark_spot, pigmentation | pigmentation |
| acne_scars, scar | acne_scars_texture |
| redness | redness |
| wrinkle branch | wrinkles |

For each non-wrinkle concern $c$, the concern value combines lesion count and mean lesion severity:

$$
\text{count\_score}_c = \min\left(1.0,\; \frac{N_c}{10}\right)
$$

$$
\text{avg\_sev}_c = \frac{1}{N_c}\sum_{i=1}^{N_c}s_i
$$

$$
u_c = \text{count\_score}_c \times \left(0.5 + 0.5 \times \text{avg\_sev}_c\right)
$$

where $N_c$ is the number of lesion detections mapped to concern $c$. If a concern has no mapped detections, then $u_c = 0$.

This concern-construction rule is also **defined in this work**. It is designed to combine two intuitions: prevalence matters, so the score should increase with lesion count, but intensity also matters, so the score should be modulated by average severity. The count term is capped at 10 to reduce sensitivity to unusually dense detections, and the multiplicative factor $(0.5 + 0.5 \times \text{avg\_sev}_c)$ keeps the final concern value in a bounded and interpretable range.

For the wrinkle dimension:

$$
u_{\text{wrinkles}} = \text{wrinkle\_concern}
$$

This produces the final 7-dimensional user vector:

$$
\mathbf{u} = [u_{\text{acne}}, u_{\text{comedonal\_acne}}, u_{\text{pigmentation}}, u_{\text{acne\_scars\_texture}}, u_{\text{pores}}, u_{\text{redness}}, u_{\text{wrinkles}}]
$$

### 4.6 Whole-face score

The frontend also displays an overall skin score derived from the concern vector. Since higher concern values indicate worse skin burden, the system converts the mean concern level into a health-style score where higher is better:

$$
\overline{u} = \frac{1}{7}\sum_{c=1}^{7}u_c
$$

$$
\text{overall\_score} = \max\left(20,\; \min\left(98,\; 100 - 80 \times \overline{u}\right)\right)
$$

If the concern vector is empty or all zeros, the displayed score defaults to **75**. The lower and upper bounds (20 and 98) prevent visually extreme outputs and make the score more stable for user-facing presentation.

This score is primarily an interpretability and UI metric; the downstream recommender uses the full concern vector rather than this scalar summary. Accordingly, the whole-face formula is **introduced in this work** as a bounded user-facing summary score rather than a clinically standardized dermatology metric.

---

## 5. Training Configuration

### Table 1. Lesion detection (YOLOv8m)

| Parameter | Value |
| --- | --- |
| Base model | YOLOv8m (COCO-pretrained) |
| Input resolution | 1280 x 1280 |
| Optimizer | AdamW |
| Initial learning rate | 0.01 |
| Final learning rate | 0.01 (cosine decay) |
| Momentum | 0.937 |
| Weight decay | $5 \times 10^{-4}$ |
| Batch size | 8 |
| Epochs | 100 |
| Early stopping patience | 20 |
| Warmup | 3 epochs, bias LR = 0.1 |
| Mixed precision (AMP) | Yes |

### Table 2. Severity classification (ResNet-18)

| Parameter | Value |
| --- | --- |
| Base model | ResNet18 (ImageNet-pretrained) |
| Input resolution | 224 x 224 |
| Optimizer | AdamW |
| Learning rate | $1 \times 10^{-4}$ |
| Scheduler | CosineAnnealingLR ($T_{\max}=40$) |
| Weight decay | $1 \times 10^{-4}$ |
| Batch size | 64 |
| Epochs | 40 |
| Early stopping patience | 8 |
| Validation split | 10% |

Gradient clipping is applied before each optimizer step using:

$$
\mathbf{g} \leftarrow \mathbf{g}\cdot \min\left(1,\frac{\tau}{\|\mathbf{g}\|_2}\right),
\qquad \tau = 5.0
$$

This stabilizes fine-tuning by bounding the total gradient norm. The clipping rule follows the standard gradient-norm clipping strategy used to prevent exploding updates [20].

---

## 6. Loss Functions

### 6.1 Lesion detection loss

YOLOv8 uses a composite detection loss:

$$
\mathcal{L}_{\text{detect}} =
\lambda_{\text{box}}\mathcal{L}_{\text{CIoU}} +
\lambda_{\text{cls}}\mathcal{L}_{\text{BCE}} +
\lambda_{\text{dfl}}\mathcal{L}_{\text{DFL}}
$$

with:

$$
\lambda_{\text{box}} = 7.5,\qquad
\lambda_{\text{cls}} = 0.5,\qquad
\lambda_{\text{dfl}} = 1.5
$$

Here:

- **CIoU loss** handles bounding-box regression while considering overlap, center distance, and aspect ratio.
- **BCE loss** handles classification.
- **Distribution Focal Loss (DFL)** refines box localization by learning distributions over boundary positions instead of a single regression point.

This formulation follows the Ultralytics YOLOv8 implementation [13]. CIoU is cited from Zheng et al., who proposed it to improve box regression by jointly considering overlap, center distance, and aspect ratio [21]. DFL is cited from Li et al., who introduced Distribution Focal Loss to model box coordinates as discrete distributions rather than single-point regression targets [22]. These losses are appropriate here because lesion localization involves many small and visually ambiguous targets, so both geometric consistency and fine-grained boundary prediction are important.

### 6.2 Severity classification loss

The severity classifier is trained using standard cross-entropy loss:

$$
\mathcal{L}_{\text{sev}} = -\sum_{k=0}^{3} y_k \log \hat{y}_k
$$

where $y_k$ is the one-hot target and $\hat{y}_k = \text{softmax}(z)_k$ is the predicted class probability for severity level $k$. Cross-entropy is used because severity prediction is modeled as a four-class classification problem with mutually exclusive labels.

### 6.3 Face parsing loss

The face-parsing model uses **Online Hard Example Mining Cross-Entropy (OHEM-CE)** over three output heads:

$$
\mathcal{L}_{\text{parse}} =
\mathcal{L}_{\text{OHEM}}^{\text{main}} +
\mathcal{L}_{\text{OHEM}}^{\text{aux16}} +
\mathcal{L}_{\text{OHEM}}^{\text{aux32}}
$$

OHEM retains only the hardest pixels (loss above threshold 0.7) during backpropagation, thereby focusing optimization on difficult boundaries and confusing regions. The hard-example-mining idea follows Shrivastava et al. [23], and its use here is motivated by the fact that facial boundaries, hairline edges, and small accessory regions are harder to segment than large easy skin regions.

---

## 7. Figure

### Figure 1. Visual targets handled by the skin analyzer

The skin analyzer processes seven major visual targets across its detection and wrinkle-analysis branches:

1. Acne  
2. Nodule  
3. Blackhead  
4. Whitehead  
5. Acne scars  
6. Flat wart  
7. Wrinkles  

The first six are handled by the lesion-detection branch, while wrinkles are handled by the segmentation branch. Together, these targets are mapped into the downstream concern representation used by Ruvisa.

---

## 8. Key Method Citations

[11] C. Yu, J. Wang, C. Peng, C. Gao, G. Yu, and N. Sang, “BiSeNet: Bilateral Segmentation Network for Real-Time Semantic Segmentation,” in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2018, pp. 325–341.

[12] C.-H. Lee, Z. Liu, L. Wu, and P. Luo, “MaskGAN: Towards Diverse and Interactive Facial Image Manipulation,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2020.

[13] G. Jocher, A. Chaurasia, and J. Qiu. (2023). *Ultralytics YOLOv8* [Online]. Available: <https://github.com/ultralytics/ultralytics>

[14] K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Las Vegas, NV, USA, 2016, pp. 770–778.

[15] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” in *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, 2015, pp. 234–241.

[16] X. Wu, L. Wen, G. Lin, Z. Ni, J. Suo, and Q. Tian, “Joint Acne Image Grading and Counting via Label Distribution Learning,” in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, 2019, pp. 10642–10651.

[17] N. Hayashi, H. Akamatsu, M. Kawashima, et al., “Establishment of grading criteria for acne severity,” *J. Dermatol.*, vol. 35, no. 5, pp. 255–260, May 2008.

[18] T. Karras, S. Laine, and T. Aila, “A Style-Based Generator Architecture for Generative Adversarial Networks,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2019, pp. 4401–4410.

[19] J. Moon, H. Chung, and I. Jang, “Facial Wrinkle Segmentation for Cosmetic Dermatology: Pretraining with Texture Map-Based Weak Supervision,” in *Pattern Recognition*, ICPR 2024, 2025, pp. 319–334.

[20] R. Pascanu, T. Mikolov, and Y. Bengio, “On the difficulty of training recurrent neural networks,” in *Proc. 30th Int. Conf. Mach. Learn. (ICML)*, 2013, pp. 1310–1318.

[21] Z. Zheng, P. Wang, W. Liu, J. Li, R. Ye, and D. Ren, “Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression,” in *Proc. AAAI Conf. Artif. Intell.*, vol. 34, no. 7, 2020, pp. 12993–13000.

[22] X. Li, W. Wang, L. Wu, S. Chen, X. Hu, J. Li, J. Tang, and J. Yang, “Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection,” in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 33, 2020, pp. 21002–21012.

[23] A. Shrivastava, A. Gupta, and R. Girshick, “Training Region-Based Object Detectors with Online Hard Example Mining,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Las Vegas, NV, USA, 2016, pp. 761–769.
