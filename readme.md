# Chexmate: Medical Domain Adaptation

Analysis of data drift for a Classification model from a dataset from one region
 to another region. Anlaysis includes a possible way of handling the data drift 
 by invariant feature extraction.


The different stages of experiments that are performed are:
1. Data Loading and Inspection
2. Baseline Model - DenseNet121
3. Neural Registration & comparison with Baseline
4. Domain Adaptation Techniques(Supervised)
    - Domain-Adversarial Neural Networks(DANN)
    - Invariant Feature Extraction
    - Meta Learning

## Dataset used:
- GLobal Dataset - (Kaggle) tawsifurrahman/tuberculosis-tb-chest-xray-dataset
- Indian Dataset - Private dataset (To be released soon)

## Model training Methodology:
- Train on Global Dataset and validate on Indian Dataset

```bash
python baseline.py --load-model outputs/saved_models/best_model_densenet121_gl_to_ind.pth --eval-only --test-dataset indian

Classification Report on Indian Dataset:
              precision    recall  f1-score   support

      Normal     0.1532    0.5810    0.2424      1000
          TB     0.8101    0.3576    0.4962      5000

    accuracy                         0.3948      6000
   macro avg     0.4817    0.4693    0.3693      6000
weighted avg     0.7007    0.3948    0.4539      6000

foo@bar$ python baseline.py --load-model outputs/saved_models/best_model_densenet121_gl_to_ind.pth --eval-only --test-dataset global

Classification Report on Global Dataset:
              precision    recall  f1-score   support

      Normal     0.9983    0.9854    0.9918      3500
          TB     0.9315    0.9914    0.9606       700

    accuracy                         0.9864      4200
   macro avg     0.9649    0.9884    0.9762      4200
weighted avg     0.9871    0.9864    0.9866      4200
```
