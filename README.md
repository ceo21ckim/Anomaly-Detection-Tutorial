
- modified: 2023-05-02

# Anomaly-Detection

# Outline
1. [Packages](#Packages)
2. [Tutorial](#Tutorial)
3. [Images](#Image)
4. [Time Series](#Time-Series)
5. [Video](#Video)
6. [Survey](#Survey)


### Packages

- PyOD (Python Outlier Detection) : [URL](https://github.com/yzhao062/pyod) | [Document](https://pyod.readthedocs.io/en/latest/pyod.html)
- TDOS (Time-series Outlier Detection Systems) : [URL](https://github.com/datamllab/tods) | [Document](https://tods-doc.github.io/)
- PyGOD (Python Graph Outlier Detection) : [URL](https://github.com/pygod-team/pygod) | [Document](https://docs.pygod.org/en/latest/)
- alibi-detect : [URL](https://github.com/SeldonIO/alibi-detect) | [Document](https://docs.seldon.io/projects/alibi-detect/en/stable/)
- PyNomaly: [URL](https://github.com/vc1492a/PyNomaly)

## Tutorial

### Machine Learning

- [Isolated Forest](https://github.com/ceo21ckim/Anomaly-Detection-Tutorial/blob/main/Isolated%20Forest/Isolation%20Forest.ipynb)
- [One Class SVM](https://github.com/ceo21ckim/Anomaly-Detection-Tutorial/blob/main/One%20Class%20SVM/One-Class-SVM.ipynb)
- [Local Outlier Factor](https://github.com/ceo21ckim/Anomaly-Detection-Tutorial/blob/main/Local%20Outlier%20Factor/Local%20Outlier%20Factor.ipynb)


### Deep Learning

- [Variational AutoEncoder](https://github.com/ceo21ckim/Anomaly-Detection-Tutorial/tree/main/Variational%20AutoEncoder)
- [LSTM-AE](https://github.com/ceo21ckim/Anomaly-Detection-Tutorial/tree/main/LSTM-AE)
- [Anomaly Transformer](https://github.com/ceo21ckim/Anomaly-Detection-Tutorial/tree/main/Anomaly%20Transformer)


# Domain

## Image
- [MemSeg: A semi-supervised method for image surface defect detection using differences and commonalities](https://arxiv.org/pdf/2205.00908.pdf) (EAAI'23)
- [Multi-Scale Patch-Based Representation Learning for Image Anomaly Detection and Segmentation](https://openaccess.thecvf.com/content/WACV2022/papers/Tsai_Multi-Scale_Patch-Based_Representation_Learning_for_Image_Anomaly_Detection_and_Segmentation_WACV_2022_paper.pdf) (WACV'22)
- [Anomaly Detection via Reverse Distillation from One-Class Embedding](https://openaccess.thecvf.com/content/CVPR2022/papers/Deng_Anomaly_Detection_via_Reverse_Distillation_From_One-Class_Embedding_CVPR_2022_paper.pdf) (CVPR'22)
- [Multiresolution knowledge distillation for anomaly detection](https://openaccess.thecvf.com/content/CVPR2021/papers/Salehi_Multiresolution_Knowledge_Distillation_for_Anomaly_Detection_CVPR_2021_paper.pdf) (CVPR'21)
- [PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization](https://hal-cea.archives-ouvertes.fr/cea-03251821v1/file/Pixdim_paper.pdf) (ICPR'21)
- [DRAEM-A discriminatively trained reconstruction embedding for surface anomaly detection](https://openaccess.thecvf.com/content/ICCV2021/papers/Zavrtanik_DRAEM_-_A_Discriminatively_Trained_Reconstruction_Embedding_for_Surface_Anomaly_ICCV_2021_paper.pdf) (ICCV'21)
- [Uninformed Students: Student–Teacher Anomaly Detection with Discriminative Latent Embeddings](https://openaccess.thecvf.com/content_CVPR_2020/papers/Bergmann_Uninformed_Students_Student-Teacher_Anomaly_Detection_With_Discriminative_Latent_Embeddings_CVPR_2020_paper.pdf) (CVPR'20)
- [Patch SVDD: Patch-level SVDD for Anomaly Detection and Segmentation](https://openaccess.thecvf.com/content/ACCV2020/papers/Yi_Patch_SVDD_Patch-level_SVDD_for_Anomaly_Detection_and_Segmentation_ACCV_2020_paper.pdf) (ACCV'20)
- [Modeling the Distribution of Normal Data in Pre-Trained Deep Features for Anomaly Detection](https://arxiv.org/pdf/2005.14140.pdf) (ICPR'20)
- [Sub-Image Anomaly Detection with Deep Pyramid Correspondences](https://arxiv.org/pdf/2005.02357.pdf) (arXiv'20)
- [Memorizing Normality to Detect Anomaly: Memory-Augmented Deep Autoencoder for Unsupervised Anomaly Detection](https://openaccess.thecvf.com/content_ICCV_2019/papers/Gong_Memorizing_Normality_to_Detect_Anomaly_Memory-Augmented_Deep_Autoencoder_for_Unsupervised_ICCV_2019_paper.pdf) (ICCV'19)
- [Adversarially learned one-class classifier for novelty detection](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sabokrou_Adversarially_Learned_One-Class_CVPR_2018_paper.pdf) (CVPR'18)
- [Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery](https://arxiv.org/pdf/1703.05921.pdf) (IPMI'17)
- [Deep One-Class Classification](http://proceedings.mlr.press/v80/ruff18a/ruff18a.pdf) (ICML'18)
- [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf) (ICLR'14)


## Time Series

- [Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy](https://openreview.net/pdf?id=LzQQ89U1qm_) (ICLR'22)
- [Inpainting Transformer for Anomaly Detection](https://arxiv.org/pdf/2104.13897.pdf) (ICIAP'22)
- [AnoViT: Unsupervised anomaly detection and localization with vision transformer-based encoder-decoder](https://arxiv.org/pdf/2203.10808.pdf) (IEEE Acess'22)
- [Multivariate time se- ries anomaly detection and interpretation using hierarchical inter-metric and temporal embedding](https://dl.acm.org/doi/pdf/10.1145/3447548.3467075) (KDD'21)
- [VT-ADL: A vision transformer network for image anomaly detection and localization](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9576231) (ISIE'21)
- [USAD: UnSupervised Anomaly Detection on Multivariate Time Series](https://dl.acm.org/doi/pdf/10.1145/3394486.3403392) (KDD'20)
- [Timeseries anomaly detection using temporal hierarchical one-class network](https://proceedings.neurips.cc/paper_files/paper/2020/file/97e401a02082021fd24957f852e0e475-Paper.pdf) (NIPS'20)
- [Integrative tensor-based anomaly detection system for reducing false positives of satellite systems](https://dl.acm.org/doi/pdf/10.1145/3340531.3412716) (CIKM'20)
- [Detecting anomalies in space using multivariate convolutional lstm with mixtures of probabilistic pca](https://dl.acm.org/doi/pdf/10.1145/3292500.3330776) (KDD'19)
- [Robust anomaly detection for multivariate time series through stochastic recurrent neural network](https://dl.acm.org/doi/pdf/10.1145/3292500.3330672) (KDD'19)
- [Outlier Detection for Time Series with Recurrent Autoencoder Ensembles](https://www.ijcai.org/proceedings/2019/0378.pdf) (IJCAI'19)
- [Time-Series Anomaly Detection Service at Microsoft](https://dl.acm.org/doi/pdf/10.1145/3292500.3330680) (KDD'19)
- [Deep autoencoding gaussian mixture model for unsupervised anomaly detection](https://openreview.net/pdf?id=BJJLHbb0-) (ICLR'18)
- [A Multimodal Anomaly Detector for Robot-Assisted Feeding Using an LSTM-Based Variational Autoencoder](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8279425) (RA-L'18)
- [LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection](https://arxiv.org/pdf/1607.00148.pdf) (ICML'16)


## Video

- [Real-world Anomaly Detection in Surveillance Videos](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.pdf) (CVPR'18)


## Survey

- [A Survey on Unsupervised Visual Industrial Anomaly Detection Algorithms](https://arxiv.org/pdf/2204.11161.pdf) (2022.08)
- [A Comprehensive Survey on Graph Anomaly Detection with Deep Learning](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9565320) (TKDE'21)
- [Deep Learning for Anomaly Detection: A Survey](https://arxiv.org/pdf/1901.03407.pdf) (2019.01)
- [Anomaly Detection: A Survey](https://dl.acm.org/doi/pdf/10.1145/1541880.1541882) (2009.06)
