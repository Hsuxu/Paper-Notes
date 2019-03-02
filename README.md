CV-DL Papers Reading List(Irregular updating)
===
Recommanded resources in Computer Vision and Deep Learning including advanced paper and issue-solutions in experiments. Mark read status as  :heavy_check_mark:


## Contents
- [CV-DL Papers Reading List(Irregular updating)](#cv-dl-papers-reading-listirregular-updating)
  - [Contents](#contents)
  - [Image Recognition](#image-recognition)
    - [Backbone Network](#backbone-network)
    - [Light Weight Network](#light-weight-network)
  - [Image Segmentation](#image-segmentation)
    - [Semantic Segmentation](#semantic-segmentation)
      - [Real Time Segmentation](#real-time-segmentation)
    - [Instance Segmentaion](#instance-segmentaion)
    - [Panoptic Segmentation](#panoptic-segmentation)
  - [Object Detection](#object-detection)
  - [Image Caption](#image-caption)
  - [Generative Adversarial Networks](#generative-adversarial-networks)
  - [Attention Mechanism](#attention-mechanism)
  - [Natural Language Processing Related](#natural-language-processing-related)
  - [Medical Image Analysis](#medical-image-analysis)
  - [Other Applications](#other-applications)
    - [Super Resolution](#super-resolution)
  - [Pose Estimation](#pose-estimation)
  - [Training Tricks](#training-tricks)
    - [Data Augmentation](#data-augmentation)
    - [Normalization](#normalization)
    - [Initialization](#initialization)
  - [AutoML](#automl)
  - [Dataset and Contest](#dataset-and-contest)
    - [Dataset](#dataset)
    - [Contest](#contest)
        
---
## Image Recognition

### Backbone Network
- [x] **[AlexNet]** ImageNet classification with deep convolutional neural networks(2012 *NeurIPS  2012*) [[Paper]](http://www.image-net.org/challenges/LSVRC/2012/supervision.pdf) [[PyTorch]](https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py)
- [x] **[ZFNet]** Visualizing and Understanding Convolutional Networks (2013.11 *ECCV 2014*) [[Paper]](https://arxiv.org/abs/1311.2901) [[TensorFlow]](https://github.com/InFoCusp/tf_cnnvis)
- [x] **[NIN]** Network In Network (2013.12 *ICLR 2014*) [[Paper]](https://arxiv.org/abs/1312.4400) [[TensorFlow(TFLearn)]](https://github.com/tflearn/tflearn/blob/master/examples/images/network_in_network.py)
- [x] **[SPPNet]** Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition (2014.6 *ECCV 2014*) [[Paper]](https://arxiv.org/abs/1406.4729)
- [x] **[VGG]** Very Deep Convolutional Networks for Large-Scale Image Recognition (2014.09)[[Paper]](https://arxiv.org/abs/1409.1556)[[Slide]](http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf) [[PyTorch]](https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py)
- [x] **[Inception V1]** Going Deeper with Convolutions (2014.09) [[Paper]](https://arxiv.org/abs/1409.4842) [[PyTorch]](https://github.com/kuangliu/pytorch-cifar/blob/master/models/googlenet.py)
- [x] **[Inception V2]** Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift (2015.02) [[Paper]](https://arxiv.org/abs/1502.03167) 
- [x] **[Inception V3]** Rethinking the Inception Architecture for Computer Vision (2015.12 *CVPR 2016*) [[Paper]](https://arxiv.org/abs/1512.00567) [[PyTorch]](https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py)
- [x] **[ResNet]** Deep Residual Learning for Image Recognition (2015.12 *CVPR 2016*) [[Paper]](https://arxiv.org/abs/1512.03385) [[PyTorch]](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)
- [x] **[ResNet V2]** Identity Mappings in Deep Residual Networks (2016.03 *ECCV 2016*) [[Paper]](https://arxiv.org/abs/1603.05027) [[Torch]](https://github.com/KaimingHe/resnet-1k-layers) [[MXNet]](https://github.com/tornadomeet/ResNet)
- [x] **[Inception v4]** Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning (2016.02) [[Paper]](https://arxiv.org/abs/1602.07261) [[TensorFlow]](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v4.py) [[PyTorch]](https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/inceptionv4.py)
- [x] **[Wide ResNet]** Wide Residual Networks (2016.05) [[Paper]](https://arxiv.org/abs/1605.07146) [[PyTorch]](https://github.com/szagoruyko/wide-residual-networks)
- [x] **[DenseNet]** Densely Connected Convolutional Networks (2016.08 *CVPR 2017*) [[Paper]](https://arxiv.org/abs/1608.06993) [[PyTorch]](https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py)
- [x] **[Xception]** Xception: Deep Learning with Depthwise Separable Convolutions (2016.10) [[Paper]](https://arxiv.org/abs/1610.02357) [[PyTorch]](https://github.com/Cadene/pretrained-models.pytorch)
- [x] **[ResNeXt]** Aggregated Residual Transformations for Deep Neural Networks (2016.11) [[Paper]](https://arxiv.org/abs/1611.05431) [[PyTorch]](https://github.com/Hsuxu/ResNeXt)
- [ ] **[PolyNet]** PolyNet: A Pursuit of Structural Diversity in Very Deep Networks (2016.11) [[Paper]](https://arxiv.org/abs/1611.05725) [[PyTorch]](https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/polynet.py)
- [x] **[DRN]** Dilated Residual Networks (2017.05 *CVPR 2017*) [[Paper]](https://arxiv.org/abs/1705.09914) [[PyTorch]](https://github.com/fyu/drn/blob/master/classify.py)
- [x] **[DPN]** Dual Path Networks (2017.07) [[Paper]](https://arxiv.org/abs/1707.01629) [[PyTorch]](https://github.com/rwightman/pytorch-dpn-pretrained) [[MXNet]](https://github.com/cypw/DPNs)
- [x] **[NASNet]** Learning Transferable Architectures for Scalable Image Recognition (2017.07) [[Paper]](https://arxiv.org/abs/1707.07012) [[TensorFlow]](https://github.com/tensorflow/models/tree/master/research/slim/nets/nasnet)
- [x] **[SENet]** Squeeze-and-Excitation Networks (2017.09) [[Paper]](https://arxiv.org/abs/1709.01507) [[Caffe]](https://github.com/hujie-frank/SENet) [[PyTorch]](https://github.com/moskomule/senet.pytorch)
- [x] **[Capsules]** Dynamic Routing Between Capsules (2017.10) [[Paper]](https://arxiv.org/abs/1710.09829) [[PyTorch]](https://github.com/gram-ai/capsule-networks) 
- [x] **[Non-Local]** Non-local Neural Networks (2017.11) [[Paper]](https://arxiv.org/abs/1711.07971) [[PyTorch]](https://github.com/AlexHex7/Non-local_pytorch) 
- [ ] **[PNASNet]** Progressive Neural Architecture Search (2017.12) [[Paper]](https://arxiv.org/abs/1712.00559) [[PyTorch](https://github.com/chenxi116/PNASNet.pytorch) [[TensorFlow]](https://github.com/tensorflow/models/blob/696b69a498b43f8e6a1ecb24bb82f7b9db87c570/research/slim/nets/nasnet/pnasnet.py)

### Light Weight Network
- [x] **[Squeeze Net]** SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size (2016.02) [[Paper]](https://arxiv.org/abs/1602.07360) [[PyTorch]](https://github.com/pytorch/vision/blob/master/torchvision/models/squeezenet.py)
- [x] **[MobileNets]** MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications (2017.04) [[Paper]](https://arxiv.org/abs/1704.04861) [[TensorFlow]](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py) [[PyTorch]](https://github.com/marvis/pytorch-mobilenet)
- [x] **[ShuffleNet V1]** ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices (2017.07) [[Paper]](https://arxiv.org/abs/1707.01083) [[PyTorch]](https://github.com/jaxony/ShuffleNet)
- [ ] **[IGCV1]** Interleaved Group Convolutions for Deep Neural Networks (20017.07 *ICCV 2017)* [[Paper]](https://arxiv.org/abs/1707.02725) [[MXNet]](https://github.com/hellozting/InterleavedGroupConvolutions)
- [ ] **[MobileNet V2]** MobileNetV2: Inverted Residuals and Linear Bottlenecks (2018.01 *CVPR 2018*) [[Paper]](https://arxiv.org/abs/1801.04381) [[PyTorch]](https://github.com/tonylins/pytorch-mobilenet-v2)
- [x] **[ENAS]** Efficient Neural Architecture Search via Parameter Sharing (2018.02 *ICML 2018*) [[Paper]](https://arxiv.org/abs/1802.03268) [[TensorFlow]](https://github.com/melodyguan/enas) [[PyTorch]](https://github.com/carpedm20/ENAS-pytorch)
- [x] **[SqueezeNext]** SqueezeNext: Hardware-Aware Neural Network Design (2018.03 *CVPR 2018*) [[Paper]](https://arxiv.org/abs/1803.10615) [[Caffe]](https://github.com/amirgholami/SqueezeNext) [[PyTorch]](https://github.com/luuuyi/SqueezeNext.PyTorch)
- [ ] **[IGCV2]** IGCV2: Interleaved Structured Sparse Convolutional Neural Networks (2018.04 *CVPR 2018*) [[Paper]](https://arxiv.org/abs/1804.06202) [[MXNet]](https://github.com/homles11/IGCV3) 
- [ ] **[IGCV3]** IGCV3: Interleaved Low-Rank Group Convolutions for Efficient Deep Neural Networks (2018.06 *BMVC 2018*) [[Paper]](https://arxiv.org/abs/1806.00178) [[MXNet]](https://github.com/homles11/IGCV3) 
- [x] **[ShuffleNet V2]** ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design (2018.07 *CVPR 2018*) [[Paper]](https://arxiv.org/abs/1807.11164) [[PyTorch]](https://github.com/miaow1988/ShuffleNet_V2_pytorch_caffe) [[PyTorch]](https://github.com/ericsun99/Shufflenet-v2-Pytorch)
- [ ] **[ESPNetv2]** ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network (2018.11 ) [[Paper]](https://arxiv.org/abs/1811.11431) [[PyTorch]](https://github.com/sacmehta/ESPNetv2) 

---
## Image Segmentation

### Semantic Segmentation
- [x] **[FCN1]** Fully Convolutional Networks for Semantic Segmentation (2014.11,*CVPR 2015*) [[Paper1]](http://www.arxiv.org/abs/1411.4038) [[Paper1]](http://www.arxiv.org/abs/1605.06211) [[PyTorch]](https://github.com/wkentaro/pytorch-fcn) 
- [x] **[DeepLab V1]** Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs (2014.12, *ICLR 2015*) [[Paper]](https://arxiv.org/abs/1412.7062) [[Caffe]](https://github.com/cdmh/deeplab-public) 
- [x] **[CRF-RNN]** Conditional Random Fields as Recurrent Neural Networks (2015.02, *ICCV 2015*) [[Paper]](https://arxiv.org/abs/1502.03240) [[PyTorch]](https://github.com/torrvision/crfasrnn
- [x] **[U-Net]** U-Net: Convolutional Networks for Biomedical Image Segmentation (2015.05, *MICCAI 2015*) [[Paper]](https://arxiv.org/abs/1505.04597) [[PyTorch]](https://github.com/milesial/Pytorch-UNet) [[PyTorch_Hsu]](https://github.com/Hsuxu/carvana-pytorch-uNet)
- [ ] **[ParseNet]** ParseNet: Looking Wider to See Better (2015.06, *ICLR 2016*) [[Paper]](https://arxiv.org/abs/1506.04579) [[Caffe]](https://github.com/debidatta/caffe-parsenet) 
- [ ] **[DAG]** DAG-Recurrent Neural Networks For Scene Labeling (2015.09, *CVPR 2016*) [[Paper]](https://arxiv.org/abs/1509.00552) [[PyTorch]](https://github.com/sallymmx/DAG-RNN) 
- [x] **[DPN]** Semantic Image Segmentation via Deep Parsing Network (2015.09 *ICCV 2015*) [[Paper]](https://arxiv.org/abs/1509.02634) [[Project]](https://liuziwei7.github.io/projects/DPN.html)
- [x] **[SegNet]** SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation (2015.11, *PAMI 2016*) [[Paper]](https://arxiv.org/abs/1511.00561) [[Caffe]](https://github.com/alexgkendall/caffe-segnet) [[TensoFlow]](https://github.com/tkuanlun350/Tensorflow-SegNet) [[PyTorch]](https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/models/segnet.py)
- [x] **[Attention to Scale]** Attention to Scale: Scale-aware Semantic Image Segmentation Liang-Chieh (2015.11 *CVPR 2016*) [[Paper]](http://arxiv.org/abs/1511.03339) [[Caffe]](https://bitbucket.org/aquariusjay/deeplab-public-ver2) [[Project]](http://lijiancheng0614.github.io/2016/11/03/2016_11_03_Attention_to_Scale/)
- [x] **[Dilated Conv]** Multi-Scale Context Aggregation by Dilated Convolutions (2015.11 *ICLR 2016*) [[Paper]](https://arxiv.org/abs/1511.07122) [[Caffe]](https://github.com/fyu/dilation)
- [ ] **[SEC]** Seed, Expand and Constrain: Three Principles for Weakly-Supervised Image Segmentation (2016.03 *ECCV 2016*) [[Paper]](http://arxiv.org/abs/1603.06098) [[Caffe]](https://github.com/kolesman/SEC)
- [ ] **[LRR]** Laplacian Pyramid Reconstruction and Refinement for Semantic Segmentation (2016.05 *ECCV 2016*) [[Paper]](https://arxiv.org/abs/1605.02264) [[Matlab]](https://github.com/golnazghiasi/LRR)
- [x] **[DeepLab V2]** DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs (2016.06 *TPAMI 2018*) [[Paper]](https://arxiv.org/abs/1606.00915) [[Caffe]](https://bitbucket.org/aquariusjay/deeplab-public-ver2)
- [x] **[DPN-MRF]** Deep Learning Markov Random Field for Semantic Segmentation (2016.06 *TPAMI 2017*) [[Paper2]](https://arxiv.org/abs/1606.07230)
- [x] **[RefineNet]** RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation (2016.11 *CVPR 2017*) [[Paper]](https://arxiv.org/abs/1611.06612) [[MatConvNet]](https://github.com/guosheng/refinenet) [[PyTorch]](https://github.com/thomasjpfan/pytorch_refinenet)
- [x] **[IFCN]** Improving Fully Convolution Network for Semantic Segmentation (2016.11) [[Paper]](https://arxiv.org/abs/1611.08986)
- [ ] **[FRRN]** Full-Resolution Residual Networks for Semantic Segmentation in Street Scenes (2016.11 *CVPR 2017*) [[Paper]](https://arxiv.org/abs/1611.08323) [[Theano]](https://github.com/TobyPDE/FRRN)
- [x] **[Tiramisu]** The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation (2016.11 *CVPRW 2017*) [[Paper]](https://arxiv.org/abs/1611.09326) [[Theano]](https://github.com/SimJeg/FC-DenseNet) [[PyTorch]](https://github.com/bfortuner/pytorch_tiramisu)
- [x] **[PSPNet]** Pyramid Scene Parsing Network (2016.12 *CVPR 2017*) [[Paper]](https://arxiv.org/abs/1612.01105) [[Caffe]](https://github.com/hszhao/PSPNet) [[PyTorch1]](https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/models/pspnet.py) [PyTorch2]](https://github.com/Lextal/pspnet-pytorch)
- [x] **[DUC]** Understanding Convolution for Semantic Segmentation (2017.02 *WACV 2018*) [[Paper]](https://arxiv.org/abs/1702.08502) [[mxnet]](https://github.com/TuSimple/TuSimple-DUC)
- [ ] **[PixelNet]** PixelNet: Representation of the pixels, by the pixels, and for the pixels (2017.02 ) [[Paper]](https://arxiv.org/abs/1702.06506) [[Caffe]](https://github.com/aayushbansal/PixelNet) [[Project]](http://www.cs.cmu.edu/~aayushb/pixelNet/)
- [x] **[GCN]** Large Kernel Matters -- [x] Improve Semantic Segmentation by Global Convolutional Network (2017.03 ) [[Paper]](https://arxiv.org/abs/1703.02719) [[PyTorch1]](https://github.com/MEDAL-IITB/Lung-Segmentation) [[PyTorch2]](https://github.com/SConsul/Global_Convolutional_Network)
- [ ] **[PixelTCN]** Pixel Deconvolutional Networks (2017.05) [[Paper]](https://arxiv.org/abs/1705.06820) [[TensorFlow]](https://github.com/HongyangGao/PixelTCN)
- [x] **[LovaszSoftmax]** The Lovász-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks (2017.05 *CVPR 2018*) [[Paper]](https://arxiv.org/abs/1705.08790) [[PyTorch&TensorFlow]](https://github.com/bermanmaxim/LovaszSoftmax)
- [x] **[DRN]** Dilated Residual Networks (2017.05 *CVPR 2017*) [[Paper]](https://arxiv.org/abs/1705.09914) [[PyTorch]](https://github.com/fyu/drn/blob/master/classify.py)
- [ ] **[G-FRNet]** Gated feedback refinement network for dense image labeling (*CVPR 2017*) [[Paper]](https://www.cs.umanitoba.ca/~ywang/papers/cvpr17.pdf) [[Caffe]](https://github.com/mrochan/gfrnet)
- [x] **[Tversky loss]** Tversky loss function for image segmentation using 3D fully convolutional deep networks (2017.06) [[Paper]](https://arxiv.org/abs/1706.05721)
- [x] **[Generalised Dice]** Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations (2017.07 ) [[Paper]](https://arxiv.org/abs/1707.03237)
- [x] **[Segmentation-aware]** Segmentation-Aware Convolutional Networks Using Local Attention Masks (2017.08 *ICCV 2017*) [[Paper]](https://arxiv.org/abs/1708.04607) [[Caffe]](https://github.com/aharley/segaware)
- [ ] **[SDN]** Stacked Deconvolutional Network for Semantic Segmentation (2017.08 ) [[Paper]](https://arxiv.org/abs/1708.04943)
- [x] **[Seg Everything]** Learning to Segment Every Thing (2017.11 *CVPR 2018*) [[Paper]](https://arxiv.org/abs/1711.10370) [[Caffe2]](https://github.com/ronghanghu/seg_every_thing)
- [x] **[DeepLab V3+]** Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation (2018.02) [[Paper]](https://arxiv.org/abs/1802.02611) [[TensorFlow]](https://github.com/rishizek/tensorflow-deeplab-v3-plus) [[TensorFlow(official)]](https://github.com/tensorflow/models/tree/master/research/deeplab) [[PyTorch]](https://github.com/jfzhang95/pytorch-deeplab-xception)
- [x] **[R2U-Net]** Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net) for Medical Image Segmentation (2018.02) [[Paper]](https://arxiv.org/abs/1802.06955) [[PyTorch]](https://github.com/LeeJunHyun/Image_Segmentation)
- [ ] **[AdaptSegNet]** Learning to Adapt Structured Output Space for Semantic Segmentation (2018.02 *CVPR 2018*) [[Paper]](https://arxiv.org/abs/1802.10349) [[PyTorch ]](https://github.com/wasidennis/AdaptSegNet)
- [x] **[Attention U-Net]** Attention U-Net: Learning Where to Look for the Pancreas (2018.04 ) [[Paper]](https://arxiv.org/abs/1804.03999) [[PyTorch]](https://github.com/ozan-oktay/Attention-Gated-Networks)
- [x] **[Vortex Pooling]** Vortex Pooling: Improving Context Representation in Semantic Segmentation(2018.04) [[Paper]](https://arxiv.org/Paper/1804.06242)
- [ ] **[DFN]** Learning a Discriminative Feature Network for Semantic Segmentation (2018.04 *CVPR 2018*) [[Paper]](https://arxiv.org/abs/1804.09337) [[TensorFlow]](https://github.com/whitesockcat/Discriminative-Feature-Network)
- [x] **[PAG]** Pixel-wise Attentional Gating for Parsimonious Pixel Labeling (2018.05 *WACV 2019*) [[Paper]](https://arxiv.org/abs/1805.01556)
- [x] **[FPANet]** Pyramid Attention Network for Semantic Segmentation(2018.05) [[Paper]](https://arxiv.org/Paper/1805.10180) [[PyTorch]](https://github.com/JaveyWang/Pyramid-Attention-Networks-pytorch)
- [x] **[Probabilistic U-Net]** A Probabilistic U-Net for Segmentation of Ambiguous Images (2018.06 *NeurIPS  2018*) [[TensorFlow]](https://github.com/SimonKohl/probabilistic_unet)
- [ ] **[G-FRNet]** Gated Feedback Refinement Network for Coarse-to-Fine Dense Semantic Image Labeling (2018.06) [[Paper]](https://arxiv.org/abs/1806.11266) 
- [x] **[OCNet]** OCNet: Object Context Network for Scene Parsing (2018.09 ) [[Paper]](https://arxiv.org/abs/1809.00916) [[PyTorch]](https://github.com/PkuRainBow/OCNet.pytorch)
- [x] **[DANet]** Dual Attention Network for Scene Segmentation (2018.09 *AAAI 2019*) [[Paper]](https://arxiv.org/abs/1809.02983) [[PyTorch]](https://github.com/junfu1115/DANet)
- [ ] **[DPC]** Searching for Efficient Multi-Scale Architectures for Dense Image Prediction (2018.09) [[Paper]](https://arxiv.org/abs/1809.04184) [[TensorFlow]](https://github.com/tensorflow/models/tree/master/research/deeplab)
- [ ] **[LadderNet]** LadderNet: Multi-path networks based on U-Net for medical image segmentation [[Paper]](https://arxiv.org/abs/1810.07810) [[PyTorch]](https://github.com/juntang-zhuang/LadderNet)
- [x] **[Pixel Augmentation]** Pixel Level Data Augmentation for Semantic Image Segmentation using Generative Adversarial Networks (2018.11) [[Paper]](https://arxiv.org/abs/1811.00174)
- [ ] **[ESPNetv2]** ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network (2018.11 ) [[Paper]](https://arxiv.org/abs/1811.11431) [[PyTorch]](https://github.com/sacmehta/ESPNetv2)
- [x] **[CCNet]** CCNet: Criss-Cross Attention for Semantic Segmentation (2018.11) [Paper]](https://arxiv.org/abs/1811.11721) [[PyTorch]](https://github.com/speedinghzl/CCNet)
- [x] **[DenseASPP]** for Semantic Segmentation in Street Scenes (*CVPR 2018*) [[Paper]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_DenseASPP_for_Semantic_CVPR_2018_paper.pdf)
- [x] **[DRN]** Dense Relation Network: Learning Consistent and Context-Aware Representation for Semantic Image Segmentation (2018 *ICIP 2018*) [[Paper]](https://ieeexplore.ieee.org/document/8451830/) [[MXNet1]](https://github.com/zhuangyqin/DRN) [MXNet2]](https://github.com/tonysy/DRN-MXNet)
- [ ] **[PSANet]** PSANet: Point-wise Spatial Attention Network for Scene Parsing (2018 *ECCV 2018*) [[Paper]](https://hszhao.github.io/papers/eccv18_psanet.pdf) [[Project]](https://hszhao.github.io/projects/psanet/) [[Caffe]](https://github.com/hszhao/PSANet)

#### Real Time Segmentation
- [x] **[ENet]** ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation (2016.06 ) [[Paper]](https://arxiv.org/abs/1606.02147) [[Caffe]](https://github.com/TimoSaemann/ENet)
- [x] **[ICNet]** ICNet for Real-Time Semantic Segmentation on High-Resolution Images (2017.04 *ECCV 2018*) [[Paper]](https://arxiv.org/abs/1704.08545) [[Caffe]](https://github.com/hszhao/ICNet) [[PyTorch]](https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/models/icnet.py)
- [x] **[ESPNet]** ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation (2018.03 *ECCV 2018*) [[Paper]](https://arxiv.org/abs/1803.06815) [[PyTorch]](https://github.com/sacmehta/ESPNet)
- [x] **[LW-RefineNet]** Light-Weight RefineNet for Real-Time Semantic Segmentation (2018.10 *BMVC 2018*) [[Paper]](https://arxiv.org/abs/1810.03272) [[PyTorch]](https://github.com/DrSleep/light-weight-refinenet) [[PyTorch]](https://github.com/DeepMotionAIResearch/DenseASPP)

### Instance Segmentaion
- [x] **[MaskLab]** MaskLab: Instance Segmentation by Refining Object Detection with Semantic and Direction Features (2017.12 *CVPR 2018*) [[Paper]](https://arxiv.org/abs/1712.04837) [[Project]](https://vitalab.github.io/deep-learning/2018/07/26/masklab.html)
- [x] **[PANet]** Path Aggregation Network for Instance Segmentation (2018.03 *CVPR 2018*) [[Paper]](https://arxiv.org/abs/1803.01534) [[PyTorch]](https://github.com/ShuLiu1993/PANet)
- [ ] **[PRMs]** Weakly Supervised Instance Segmentation Using Class Peak Response (2018.04 *CVPR 2018*) [[Paper]](https://arxiv.org/abs/1804.00880) [[PyTorch]](https://github.com/ZhouYanzhao/PRM)

### Panoptic Segmentation
- [x] **[PS]** Panoptic Segmentation (2018.01 ) [[Paper]](https://arxiv.org/abs/1801.00868) 
- [ ] **[Semi-Supervised PS]** Weakly- and Semi-Supervised Panoptic Segmentation (2018.08 *ECCV 2018* ) [[Paper]](https://arxiv.org/abs/1808.03575) [[Matlab]](https://github.com/qizhuli/Weakly-Supervised-Panoptic-Segmentation)
- [ ] **[JSIS-Net]** Panoptic Segmentation with a Joint Semantic and Instance Segmentation Network (2018.09) [[Paper]](https://arxiv.org/abs/1809.02110) 
- [ ] **[TASCNet]** Learning to Fuse Things and Stuff (2018.12) [[Paper]](https://arxiv.org/abs/1812.01192)
- [x] Interactive Full Image Segmentation (2018.12) [[Paper]](https://arxiv.org/abs/1812.01888)
- [x] **[AUNet]** Attention-guided Unified Network for Panoptic Segmentation (2018.12 ) [[Paper]](https://arxiv.org/abs/1812.03904)
- [ ] **[PS FPN]** Panoptic Feature Pyramid Networks (2019.01) [[Paper]](https://arxiv.org/abs/1901.02446)
- [ ] **[UPSNet]** UPSNet: A Unified Panoptic Segmentation Network (2019.01) [[Paper]](https://arxiv.org/abs/1901.03784) 


---
## Object Detection
- [x] **[RCNN]** Rich feature hierarchies for accurate object detection and semantic segmentation(2013.11 *CVPR 2014*) [[Paper]](https://arxiv.org/abs/1311.2524) [[Matlab]](https://github.com/rbgirshick/rcnn)
- [x] **[Fast R-CNN]** Fast R-CNN (2015.04 *ICCV 2015*) [[Paper]](https://arxiv.org/abs/1504.08083) [[Caffe]](https://github.com/rbgirshick/fast-rcnn)
- [x] **[YOLO]** You Only Look Once: Unified, Real-Time Object Detection (2015.06 *CVPR 2016*) [[Paper]](https://arxiv.org/abs/1506.02640) [[Project]](https://pjreddie.com/darknet/yolo/) [[DarkNet]](https://github.com/pjreddie/darknet) [[PyTorch]](https://github.com/xiongzihua/pytorch-YOLO-v1)
- [x] **[SSD]** SSD: Single Shot MultiBox Detector (2015.12 *ECCV 2016*) [[Paper]](https://arxiv.org/abs/1512.02325) [[Caffe]](https://arxiv.org/abs/1512.02325) [[PyTorch]](https://github.com/amdegroot/ssd.pytorch)
- [x] **[OHEM]** Training Region-based Object Detectors with Online Hard Example Mining (2016.01 *CVPR 2016*) [[Paper]](https://arxiv.org/abs/1604.03540) [[Caffe]](https://github.com/abhi2610/ohem) 
- [x] **[R-FCN]** R-FCN: Object Detection via Region-based Fully Convolutional Networks (2016.05 *NeurIPS 2016*) [[Paper]](https://arxiv.org/abs/1605.06409) [[Caffe]](https://github.com/daijifeng001/R-FCN)
- [ ] **[DAG]** Adversarial Examples for Semantic Segmentation and Object Detection (2017.03 *ICCV 2017*) [[Paper]](https://arxiv.org/abs/1703.08603) [[Caffe]](https://github.com/cihangxie/DAG)
- [x] **[Focal Loss]** Focal Loss for Dense Object Detection (2017.07) [[Paper]](https://arxiv.org/abs/1708.02002)
- [ ] **[DSOD]** DSOD: Learning Deeply Supervised Object Detectors from Scratch (2017.07 *ICCV 2017*) [[Paper]](https://arxiv.org/abs/1708.01241) [[Caffe with SSD]](https://github.com/szq0214/DSOD) [[PyTorch]](https://github.com/uoip/SSD-variants) [[MXNet]](https://github.com/eureka7mt/mxnet-dsod)
- [x] **[RFBNet]** Receptive Field Block Net for Accurate and FastObject Detection (2017.11 *ECCV 2018*) [[Paper]](https://arxiv.org/abs/1711.07767) [[PyTorch]](https://github.com/ruinmessi/RFBNet)
- [ ] **[Cascade R-CNN]** Cascade R-CNN: Delving into High Quality Object Detection (2017.12 *CVPR 2018*) [[Paper]](https://arxiv.org/abs/1712.00726) [[Caffe]](https://github.com/zhaoweicai/Detectron-Cascade-RCNN) [[Note_Zhihu]](https://zhuanlan.zhihu.com/p/41825737)
- [ ] **[CornerNet]** CornerNet: Detecting Objects as Paired Keypoints (2018.08 *ECCV 2018*) [[Paper]](https://arxiv.org/abs/1808.01244) [[PyTorch]](https://github.com/princeton-vl/CornerNet) 
- [x] **[Survey]** Deep Learning for Generic Object Detection: A Survey (2018.09 *IJCV 2018*) [[Paper]](https://arxiv.org/abs/1809.02165)
- [ ] **[Training From Scratch]** Rethinking ImageNet Pre-training (2018.11) [[Paper]](https://arxiv.org/abs/1811.08883)
- [x] **[GHM_Detection]** Gradient Harmonized Single-stage Detector (2018.11 *AAAI 2019*) [[Paper]](https://arxiv.org/abs/1811.05181) [[PyTorch]](https://github.com/libuyu/GHM_Detection)
- [ ] **[BoF]** Bag of Freebies for Training Object Detection Neural Networks (2019.02) [[Paper]](https://arxiv.org/abs/1902.04103) [[MxNet]](https://github.com/dmlc/gluon-cv)
- [ ] **[Generalized IoU]** Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression (2019.02, *CVPR 2019*) [[Paper]](https://arxiv.org/abs/1902.09630)

---
## Image Caption
- [x] **[Show, Attend and Tell]** Show, Attend and Tell: Neural Image Caption Generation with Visual Attention(2015) [[Paper]](https://arxiv.org/abs/1502.03044) [[TensorFlow]](https://github.com/yunjey/show-attend-and-tell) [[PyTorch]](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)
- [x] Image Captioning with Semantic Attention(2016) [[Paper]](https://arxiv.org/abs/1603.03925) [[Torch]](https://github.com/chapternewscu/image-captioning-with-semantic-attention)
- [x] Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering(2017) [[Paper]](https://arxiv.org/abs/1707.07998) [[Caffe]](https://github.com/peteanderson80/bottom-up-attention)
- [x] Convolutional Image Captioning(2017) [[Paper]](https://arxiv.org/abs/1711.09151) [[PyTorch]](https://github.com/aditya12agd5/convcap)
- [ ] CNN+CNN: Convolutional Decoders for Image Captioning (2018) [[Paper]](https://arxiv.org/abs/1805.09019)

---
## Generative Adversarial Networks
- [x] **[DCGAN]** Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (2015.11 *ICLR 2016*) [[Paper]](https://arxiv.org/abs/1511.06434) [[PyTorch]](https://github.com/pytorch/examples/tree/master/dcgan) [[TensorFlow]](https://github.com/carpedm20/DCGAN-tensorflow) [[Torch]](https://github.com/soumith/dcgan.torch)
- [ ] **[MUNIT]** Multimodal Unsupervised Image-to-Image Translation (2018.04 *ECCV 2018*) [[Paper]](https://arxiv.org/abs/1804.04732) [[PyTorch]](https://github.com/NVlabs/MUNIT)
- [ ] **[SAGAN]** Self-Attention Generative Adversarial Networks(2018.05) [[Paper]](https://arxiv.org/abs/1805.08318) [[PyTorch]](https://github.com/heykeetae/Self-Attention-GAN) 
- [ ] **[DIRT]** Diverse Image-to-Image Translation via Disentangled Representations(2018.08) [[Paper]](https://arxiv.org/abs/1808.00948) [[PyTorch]](https://github.com/HsinYingLee/DRIT) <font color=Gray >(Notes: maybe suitable for unpaired MR-CT synthesis for human body)</font>
- [x] **[VID2VID]** Video-to-Video Synthesis(2018.08) [[Paper]](https://arxiv.org/abs/1808.06601) [[PyTorch]](https://github.com/NVIDIA/vid2vid)
- [x] **[BigGAN]** Large Scale GAN Training for High Fidelity Natural Image Synthesis (2018.09) [[Paper]](https://arxiv.org/abs/1809.11096) [[PyTorch]](https://github.com/AaronLeong/BigGAN-pytorch)
- [ ] **[styleGAN]** A Style-Based Generator Architecture for Generative Adversarial Networks (2018.12) [[Paper]](https://arxiv.org/abs/1812.04948) 

---
## Attention Mechanism
- [x] Show, Attend and Tell: Neural Image Caption Generation with Visual Attention(2015.02) [[Paper]](https://arxiv.org/abs/1502.03044) [[TensorFlow]](https://github.com/yunjey/show-attend-and-tell) [[PyTorch]](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)
- [x] Image Captioning with Semantic Attention(2016.03) [[Paper]](https://arxiv.org/abs/1603.03925) [[Torch]](https://github.com/chapternewscu/image-captioning-with-semantic-attention)
- [x] Attention Is All You Need(2017.06) [[Paper]](https://arxiv.org/abs/1706.03762) [[PyTorch]](https://github.com/jadore801120/attention-is-all-you-need-pytorch) [[TensorFlow]](https://github.com/Kyubyong/transformer)
- [x] Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering(2017.07) [[Paper]](https://arxiv.org/abs/1707.07998) [[Caffe]](https://github.com/peteanderson80/bottom-up-attention)
- [x] Attention U-Net:Learning Where to Look for the Pancreas(2018.04) [[Paper]](https://arxiv.org/abs/1804.03999) [[PyTorch]](https://github.com/ozan-oktay/Attention-Gated-Networks)
- [x] Self-Attention Generative Adversarial Networks(2018.05) [[Paper]](https://arxiv.org/abs/1805.08318) [[PyTorch]](https://github.com/heykeetae/Self-Attention-GAN)
(Notes: 将自我注意机制引入到GAN的生成模型中，对于图像的纹理和几何上的联系提供全局的注意使得生成的图像更加的合理)
- [ ] Learning Visual Question Answering by Bootstrapping Hard Attention (2018.08) [[Paper]](https://arxiv.org/abs/1808.00300) [[PyTorch]](https://github.com/gnouhp/PyTorch-AdaHAN) (Note: Hard-Attention)
- [ ] Pervasive Attention: 2D Convolutional Neural Networks for Sequence-to-Sequence Prediction(2018.08) [[Paper]](https://arxiv.org/abs/1808.03867) [[PyTorch]](https://github.com/elbayadm/attn2d)

---
## Natural Language Processing Related
- [x] **[Pervasive Attention]** Pervasive Attention: 2D Convolutional Neural Networks for Sequence-to-Sequence Prediction(2018.08 *CoNLL 2018*) [[Paper]](https://arxiv.org/abs/1808.03867) [[PyTorch]](https://github.com/elbayadm/attn2d)

---
## Medical Image Analysis
- [x] **[Rician Normalization]** Normalization of T2W-MRI Prostate Images using Rician a priori (*Medical Imaging 2016: Computer-Aided Diagnosis*) [[Paper]](http://adsabs.harvard.edu/abs/2016SPIE.9785E..29L)
- [x] **[U-Net]** U-Net: Convolutional Networks for Biomedical Image Segmentation (2015.05) [[Paper]](https://arxiv.org/abs/1505.04597) [[PyTorch]](https://github.com/milesial/Pytorch-UNet) [[PyTorch(Hsu)]](https://github.com/Hsuxu/carvana-pytorch-uNet)
- [x] **[V-Net]** V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation (2016.06) [[Paper]](https://arxiv.org/abs/1606.04797) [[PyTorch]](https://github.com/mattmacy/vnet.pytorch) [[PyTorch(Hsu)]](https://github.com/Hsuxu/Magic-VNet)
- [x] **[XmasNet]** Prostate Cancer Diagnosis using Deep Learning with 3D Multiparametric MRI (2017.03 *Medical Imaging: Computer-Aided Diagnosis 2017*) [[Paper]](https://arxiv.org/abs/1703.04078) 
- [x] **[Tversky loss]** Tversky loss function for image segmentation using 3D fully convolutional deep networks (2017.06) [[Paper]](https://arxiv.org/abs/1706.05721)
- [ ] **[TDN]** Automated Detection of Clinically Significant Prostate Cancer in mp-MRI Images based on an End-to-End Deep Neural Network (2018.07 *TMI 2018*) [[Paper]](https://ieeexplore.ieee.org/iel7/42/8353174/08245842.pdf)
- [x] **[AnatomyNet]** AnatomyNet: Deep Learning for Fast and Fully Automated Whole-volume Segmentation of Head and Neck Anatomy (2018.08) [[Paper]](https://arxiv.org/abs/1808.05238)
- [x] Improving Data Augmentation for Medical Image Segmentation (2018.12 *MIDL 2018*) [[Paper]](https://openreview.net/forum?id=rkBBChjiG)
- [x] Computer-aided diagnosis of prostate cancer using a deep convolutional neural network from multiparametric MRI (*JMRI 2018*) [[Paper]](https://www.ncbi.nlm.nih.gov/pubmed/29659067)
- [ ] **[Ensemble]** A New Ensemble Learning Framework for 3D Biomedical Image Segmentation (2018.12 *AAAI 2019*) [[Paper]](https://arxiv.org/abs/1812.03945) [[code]](https://github.com/HaoZheng94/Ensemble)

---
## Other Applications

### Super Resolution
- [x] Enhanced Deep Residual Networks for Single Image Super-Resolution (2017.7 *CVPR 2017*) [[Paper]](https://arxiv.org/abs/1707.02921) [[PyTorch]](https://github.com/thstkdgus35/EDSR-PyTorch)
- [x] Deep Learning for Image Super-resolution: A Survey [[Paper]](https://arxiv.org/abs/1902.06068)

## Pose Estimation
- [ ] **[DeepPose]** DeepPose: Human Pose Estimation via Deep Neural Networks (2013.11 *CVPR 2014*) [[Paper]](https://arxiv.org/abs/1312.4659) [[Chainer]](https://github.com/mitmul/deeppose)
- [x] **[Hourglass]** Stacked Hourglass Networks for Human Pose Estimation (2016.03 *ECCV 2016*) [[Paper]](https://arxiv.org/abs/1603.06937) [[Torch]](https://github.com/anewell/pose-hg-train)
- [ ] **[MSPN]** Rethinking on Multi-Stage Networks for Human Pose Estimation (2019.01) [[Paper]](https://arxiv.org/abs/1901.00148)


---
## Training Tricks
- [x] **[Bag of Tricks]** Bag of Tricks for Image Classification with Convolutional Neural Networks (2018.12) [[Paper]](https://arxiv.org/abs/1812.01187)
### Data Augmentation
- [x] **[Pairing Samples]** Data Augmentation by Pairing Samples for Images Classification (2018.01) [[Paper]](https://arxiv.org/abs/1801.02929) 
- [ ] **[AutoAugment]** AutoAugment: Learning Augmentation Policies from Data (2018.05) [[Paper]](https://arxiv.org/abs/1805.09501) 
- [x] **[Albumentations]** Albumentations: fast and flexible image augmentations (2018.09) [[Paper]](https://arxiv.org/abs/1809.06839)
- [ ] **[Pixel Augmentation]** Pixel Level Data Augmentation for Semantic Image Segmentation using Generative Adversarial Networks (2018.11) [[Paper]](https://arxiv.org/abs/1811.00174)
 

### Normalization
- [x] **[Batch Normalization]** Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift (2015.02) [[Paper]](https://arxiv.org/abs/1502.03167)
- [x] **[Layer Normalization]** Layer Normalization(2016.07) [[Paper]](https://arxiv.org/abs/1607.06450)
- [x] **[Instance Normalization]** Instance Normalization: The Missing Ingredient for Fast Stylization (2016.07) [[Paper]](https://arxiv.org/abs/1607.08022)
- [x] **[Group Normalization]** Group Normalization (2018.03) [[Paper]](https://arxiv.org/abs/1803.08494) [[TensorFlow]](https://github.com/shaohua0116/Group-Normalization-Tensorflow)
- [x] How Does Batch Normalization Help Optimization? (2018.05 *NeurIPS 2018*) [[Paper]](https://arxiv.org/abs/1805.11604)
- [ ] **[Switchable Normalization]** Differentiable Learning-to-Normalize via Switchable Normalization (2018.06) [[Paper]](https://arxiv.org/abs/1806.10779) [[PyTorch]](https://github.com/switchablenorms/Switchable-Normalization)
- [x] Synchronized-BatchNorm [[PyTorch]](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch) [[MXNet]](https://github.com/zhanghang1989/MXNet-Gluon-SyncBN)


### Initialization
- [x] Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification (2015.02) [[Paper]](https://arxiv.org/abs/1502.01852)
- [x] Understanding the difficulty of training deep feedforward neural networks [[Paper]](http://proceedings.mlr.press/v9/glorot10a.html)
- [x] Training Techniques--An overview of gradient descent optimization algorithms [[url]](http://ruder.io/optimizing-gradient-descent/index.html)
- [x] **[ZeroInit]** Residual Learning Without Normalization via Better Initialization (2018.09 *ICLR2019*) [[Paper]](https://openreview.net/forum?id=H1gsz30cKX)


---
## AutoML
- [ ] Learning to Optimize(2016.06) [[Paper]](https://arxiv.org/abs/1606.01885)
- [ ] Neural Architecture Search with Reinforcement Learning (2016.11) [[Paper]](https://arxiv.org/abs/1611.01578)
- [ ] **[MetaQNN]** Designing Neural Network Architectures using Reinforcement Learning (2016.11 *ICLR 2017*) [[Paper]](https://arxiv.org/abs/1611.02167)
- [ ] **[Large-Scale Evolution]** Large-Scale Evolution of Image Classifiers (2017.03 *ICML 2017*) [[Paper]](https://arxiv.org/abs/1703.01041)
- [ ] **[Genetic CNN]** Genetic CNN [[Paper]](https://arxiv.org/abs/1703.01513)
- [ ] **[EAS]** Efficient Architecture Search by Network Transformation(2017.07 *AAAI 2018*) [[Paper]](https://arxiv.org/abs/1707.04873) [[TensorFlow ]](https://github.com/han-cai/EAS)
- [x] **[NASNet]** Learning Transferable Architectures for Scalable Image Recognition (2017.07) [[Paper]](https://arxiv.org/abs/1707.07012) [[TensorFlow]](https://github.com/tensorflow/models/tree/master/research/slim/nets/nasnet)
- [ ] Hierarchical Representations for Efficient Architecture Search (2017.11 *ICLR 2018*) [[Paper]](https://arxiv.org/abs/1711.00436)
- [ ] **[AmoebaNet-A]** Regularized Evolution for Image Classifier Architecture Search (2018.02) [[Paper]](https://arxiv.org/abs/1802.01548)
- [x] **[ENAS]** Efficient Neural Architecture Search via Parameter Sharing (2018.02 *ICML 2018*) [[Paper]](https://arxiv.org/abs/1802.03268) [[TensorFlow]](https://github.com/melodyguan/enas) [[PyTorch]](https://github.com/carpedm20/ENAS-pytorch)

---
## Dataset and Contest
### Dataset
- [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/)<!-- The main goal of this challenge is to recognize objects from a number of visual object classes in realistic scenes (i.e. not pre-segmented objects). It is fundamentally a supervised learning learning problem in that a training set of labelled images is provided. -->
- [Cityscapes](https://www.cityscapes-dataset.com/)<!-- Large-scale dataset that contains a diverse set of stereo video sequences recorded in street scenes from 50 different cities, with high quality pixel-level annotations of 5 000 frames in addition to a larger set of 20 000 weakly annotated frames. -->
- [NYUv2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)<!-- The NYU-Depth V2 data set is comprised of video sequences from a variety of indoor scenes as recorded by both the RGB and Depth cameras from the Microsoft Kinect. -->
- [PASCAL-Context](https://cs.stanford.edu/~roozbeh/pascal-context/)<!-- This dataset is a set of additional annotations for PASCAL VOC 2010. It goes beyond the original PASCAL semantic segmentation task by providing annotations for the whole scene. The statistics section has a full list of 400+ labels. -->
- [SUNRGBD](http://rgbd.cs.princeton.edu/)<!-- Dataset is captured by four different sensors and contains 10,335 RGB-D images, at a similar scale as PASCAL VOC. The whole dataset is densely annotated and includes 146,617 2D polygons and 64,595 3D bounding boxes with accurate object orientations, as well as a 3D room layout and scene category for each image. -->
- [ADE20k](http://groups.csail.mit.edu/vision/datasets/ADE20K/)<!-- ADE20K Dataset contains more than 20K scene-centric images exhaustively annotated with objects and object parts. Specifically, the benchmark is divided into 20K images for training, 2K images for validation, and another batch of held-out images for testing. There are totally 150 semantic categories included for evaluation, which include stuffs like sky, road, grass, and discrete objects like person, car, bed. Note that there are non-uniform distribution of objects occuring in the images, mimicking a more natural object occurrence in daily scene. -->
- [Grand Challenges in Biomedical Image Analysis](https://grand-challenge.org): 
<!-- Grand Challenges in Biomedical Image Analysis -->

---
### Contest
- [AI Challenger](https://challenger.ai/)