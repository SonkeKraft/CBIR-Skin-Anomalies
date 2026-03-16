# CBIR-Skin-Anomalies

##This GitHub repository includes:

- Pyhton script for CBIR of skin anomalies: Skin_cancer_CBIR_HNSW.py 
- Report: Report CBIR Skin Anomaly Retrieval
- Presentation: Presentation CBIR Skin anomalie retrieval
- Small example image set (!! this project uses the full set from kaggle !!):Images_subset.zip

##Access to the full data set used for the project
- Downlaod the image data base:https://www.kaggle.com/code/jaimemorillo/cbir-skin-cancer-lesions/input

##Run Skin_cancer_CBIR_HNSW.py in Pyhton and define:
(30) cnn_weight=0.6,      
(31) color_weight=0.2,    
(32) texture_weight=0.2   
(438) Data base path: dataset_folder = "./images"
(444) Image subset number: max_images=10000
(453) Select query image: query_idx
(453) select top-k similar images



