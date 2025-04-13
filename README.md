# 6748_final
Spring 2025
Team M.O.S.
Mansi Malegaonkar
Osman Yardimci
Sudhir Rajbhandary

Data Sources
1. Initial IMBD from kaggle.  
   https://www.kaggle.com/datasets/anandshaw2001/imdb-data
   
2. IMDB non-commerical database. These files were joined using imdb_id to add actor and director info
   https://developer.imdb.com/non-commercial-datasets/
  a) title.crew.tsv (contains actors ids for each movie)
  b) title.principals.tsv (director ids for each movie)
  c) name.basics.tsv (contains names of actors and directors)

4. Finally, combined files from 1. and 2. above was split into train, validation, test, and combined_data.csv
  a) train_data.csv
  b) validation_data.csv
  c) test_data.csv
  d) combined_data.csv

Code Files
1. combined_eda.ipynb (exploratory data analysis)
2. knn.py (k nearest neighbor implementation)
3. neural_network.py
4. TFIDF_XGBoost.ipynb
5. XGBoost_final.ipynb
6. LogisticRegression-final.ipynb
7. SVM.ipynb
8. RandomForestClassifier.ipynb
9. GradientBoostingClassifier.ipynb
