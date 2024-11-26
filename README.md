# Examen BentoML

Ce répertoire contient l'architecture basique afin de rendre l'évaluation pour l'examen BentoML.

```bash       
├── examen_bentoml          
│   ├── data       
│   │   ├── processed: folder with train and test scaled data
│   │   └── raw: raw data        
│   ├── models     
│   ├── src
│   │   ├── api
│   │   │   ├── service.py: to start the api and serve the model
│   │   │   └── test.py: to test the launched api
│   │   ├── data
│   │   │   ├── check_structure.py: module
│   │   │   ├── import_raw.py: to get the raw data in folder 'data/raw'
│   │   │   └── make_dataset.py: to train/test split and scale data in folder 'data/processed'
│   │   ├── docker
│   │   │   └── Dockerfile: workaround to solve the issue when building bento
│   │   └── model
│   │   │   └── train_model.py: to train, select and save the best model
│   ├── bentofile.yaml
│   ├── bentoml_cmd.txt: general cmds I used to run all the stuffs
│   ├── requirements.txt
│   └── README.md
```

Ordre chronologique du jeu des scripts, joué à la racine du projet : 
- le script import_raw.py permet de récupérer le dataset brut
- le script make_dataset.py prépare les datasets d'entraînement et de test (suppression de colonne, scaling)
- le script train_model.py entraîne plusieurs modèles (SVM Regressor, Random Forest Regressor, XGBoost Regressor) puis enregistre le meilleur
- le script service.py sert de base à la commande bentoml permettant de lancer l'API
- le script test.py lance des appels à l'API afin de tester son bon fonctionnement

J'ai essayé de diminuer la taille de l'image Docker mais je ne parviens pas à descendre en dessous de 1,6 Go donc après avoir posé le problème dans le chat Daniel, on m'a répondu d'enlever l'image. 
