This dataset is for evaluating fine-grained visual object recognition by incorporating unconstrained OCR. Specifically, we have took sub-classes of the ImageNet "building" and "place of business" sets to evaluate ﬁne-grained classiﬁcation. The dataset consists of 28 categories with 24,255 images in total. 

Here we provide:

- the images, 
- the train/test splits
- our precomputed image-features
- our raw OCR output
- our OCR-features
- our baseline results 

Belonging with the paper:

Con-Text: Text Detection Using Background Connectivity for Fine-Grained Object Classiﬁcation
By Sezer Karaoglu, Jan van Gemert and Theo Gevers; presented at ACM-Multimedia 2013
https://staff.fnwi.uva.nl/j.c.vangemert/pub/KaraogluACMMM13conText.pdf

This package is structured as follows:

abby2bow/ --- convert OCR to n-gram BOW features
codebookPerIm/ --- our features per iamge
data/ --- images and labels
results/ --- our results
readme.txt/ --- this file.

A more detailed description follows:

*** abby2bow/

This directory contains our raw OCR results (OCR_bright_ASCI, OCR_dark_ASCI) and a python script (txt2vocab.py) to convert the OCR results to bag-of-Ngram counts which are used as the OCR features in the paper. We use the abby2bow/DescriptorIO.py script to write the features.

*** codebookPerIm/

This directory contains our features per image. We have OCR text-features and visual features.
Use the abby2bow/DescriptorIO.py script to read the features.

For visual features We use a standard bag-of-visual-word approach with 4,000 visual words obtained through k-means. 
We densely sample every 6 pixels with a scale of 1.2 and use standard (gray) SIFT as a descriptor through the online software binaries of Koen van de Sande from "Evaluating Color Descriptors for Object and Scene Recognition", TPAMI 2010, http://koen.me/research/colordescriptors/ and the DescriptorIO.py script to read/write features.
The spatial pyramid (SPM) is used for rough spatial matching. We use the following pyramid levels: 1x1, 3x1, 2x2. Each level in the pyramid has its own directory.

The OCR features were generated from our OCR output, see abby2bow/ 

We use the histogram intersection kernel as input to LIBSVM.

Each sub-directory contains a part of our features, each image is present in each sub-directory:

bigrams -- OCR features
dSp6Sc1_2 -- dense SIFT whole image (=Spatial Pyramid 1x1)
1x3SPM0 -- Spatial Pyramid 1x3
1x3SPM1 -- Spatial Pyramid 1x3
1x3SPM2 -- Spatial Pyramid 1x3
2x2SPM0 -- Spatial Pyramid 2x2
2x2SPM1 -- Spatial Pyramid 2x2
2x2SPM2 -- Spatial Pyramid 2x2
2x2SPM3 -- Spatial Pyramid 2x2

*** data

This directory contains the images "JPEGImages" and the labels, split into 3 random partitions (0,1,2). We follow the Pascal VOC format where each class has text file containing binary labels. Also text files containing the names of the train/test/train+test images are included.

*** results

This contains our results, for each of the 3 random splits of the data (0,1,2).
Results are given for OCR-only (bigram), for visual-only (spm) and for the combination of visual+text (spm_bigram). Each directory contains the average precision score per class (.AP), and the raw classification output, where the order of the lines is given by data/ImageSets/?/test.txt, where '?' denotes the corresponding random split of the data.

We hope you find this dataset useful, and if so, we hope you cite our paper.
If you have any questions you can contact us here:

S.Karaoglu@uva.nl
J.C.vanGemert@uva.nl


