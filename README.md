# Breast mass classification with a scratch CNN and frozen VGG16

CM3070 final project. Template: 3 CM3015 Machine Learning and Neural Networks; 3.2 Project Idea 2: Deep Learning Breast Cancer Detection.

This research prototype compares a small randomly initialised CNN with frozen ImageNet VGG16 features and a trained classifier head. It classifies provided mass crops, not whole mammograms, and is not a clinical diagnostic tool.

## Files
- CNN_VGG16_Comparison.ipynb: executable workflow with narrative and source appendices. Execution outputs have been cleared for this public copy.
- project.py: original recorded baseline source, unchanged.
- prepare_cbis.py: metadata mapping adapter, unchanged.
- requirements.txt: installation requirements. TensorFlow matches the recorded 2.21.0 version; other entries are not a complete original environment lock.
- VIDEO_GUIDE.md: demonstration preparation and spoken script.
- stakeholders/responses.csv: empty feedback template, not collected evidence.

## Setup
Use Python 3.12 in an isolated environment. From this folder run:

```bash
python -m pip install -r requirements.txt
python -m jupyterlab
```

Obtain CBIS-DDSM mass images and official train/test metadata from https://www.cancerimagingarchive.net/collection/cbis-ddsm/ and follow its access and attribution requirements. Set the local paths in notebook Section 2. No dataset images, trained weights, local overrides, run directories or patient-level tables are included here. Source modules require those local inputs to train or infer; downloading this repository alone does not provide a pretrained demonstration.

The original experiment resolved 1,593 crop mappings through local overrides and produced 1,696 crops from 892 patients. New downloads can use different paths. The adapter writes path_mapping_needed.csv when exact paths or unique DICOM series matches fail. Review candidates using DICOM metadata and pixels; never select the first file arbitrarily because a crop and ROI mask can share a series. Supply the corrected override CSV through OVERRIDES and rebuild the manifest. Inspect prepared training images and record the actual review in Section 6 before training.

Existing local runs should be demonstrated in their original project folder. Do not overwrite project.py or change an existing run's data/source to bypass hash checks. The portable public notebook has path and manifest-loading changes; baseline module code remains unchanged.

## Protocol
Official test patients held out; fold zero of a five-fold grouped development split supplies validation. This is one hold-out experiment, not full cross-validation. Inputs are 224 by 224 grayscale crops repeated over three channels, with model-specific scaling. Adam 0.0001, batch 16, maximum 30 epochs, patience 5, seed 42, training-only class weights. VGG16 is frozen; no augmentation or fine-tuning. Test decision threshold 0.5. Paired bootstrap resamples patients 1,000 times.

## Recorded results
The following are transcribed from saved notebook outputs from 28 September 2026, not newly reproduced by this repository packaging step. Test set: 378 crops from 201 patients.

| Model | Accuracy | ROC-AUC | Sensitivity | Specificity | False negatives |
|---|---:|---:|---:|---:|---:|
| Scratch CNN | 0.616 | 0.536 | 0.034 | 0.987 | 142/147 |
| Frozen VGG16 | 0.603 | 0.634 | 0.537 | 0.645 | 68/147 |
| Majority reference | 0.611 | 0.500 | 0.000 | 1.000 | 147/147 |

The scratch CNN learned weak discrimination. VGG16 improved sensitivity and ROC-AUC but still missed many malignant crops. Differences do not isolate pretraining because architecture and parameter counts also differ. Neither model is clinically validated. No stakeholder feedback is claimed.

## Attribution
Lee et al. (2017), A curated mammography data set for use in computer-aided detection and diagnosis research. https://doi.org/10.1038/sdata.2017.177

Simonyan and Zisserman (2015), Very Deep Convolutional Networks for Large-Scale Image Recognition. https://arxiv.org/abs/1409.1556

Disclose external code and AI assistance in accordance with your institution's requirements. Public visibility does not itself grant an open-source licence; no licence has been selected in this package.
