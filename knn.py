import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from itertools import combinations
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2, SelectKBest, f_classif

warnings.filterwarnings('ignore')

#print display
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

#main imdb file
path_train = 'data/train_df.csv'
data_train = pd.read_csv(path_train) #reads data
data_train = data_train.fillna(0)

path_validation = 'data/validation_df.csv'
data_validation = pd.read_csv(path_validation) #reads data
data_validation = data_validation.fillna(0)

data_validation['release_year']=data_validation['release_date']
data_validation['release_month']=data_validation['release_year']

path_test = 'data/test_df.csv'
data_test = pd.read_csv(path_test) #reads data
data_test = data_test.fillna(0)

selected_columns = [ 'runtime','budget',
          'Action','Adventure','Animation','Comedy','Crime','Documentary','Drama',
          'Family','Fantasy','History','Horror','Music','Mystery','Romance',
          'Science Fiction','TV Movie','Thriller','War','Western',
          'production_companies_20th Century Fox','production_companies_Columbia Pictures',
          'production_companies_Metro-Goldwyn-Mayer','production_companies_New Line Cinema',
          'production_companies_Paramount','production_companies_Universal Pictures',
          'production_companies_Walt Disney Pictures','production_companies_Warner Bros. Pictures',
          'production_companies_other',
          'Mapped Value_Africa','Mapped Value_Asia','Mapped Value_Australia',
          'Mapped Value_Europe','Mapped Value_France','Mapped Value_Germany',
          'Mapped Value_India','Mapped Value_North America','Mapped Value_South America',
          'Mapped Value_United Kingdom','Mapped Value_United States of America',
          'spoken_languages_English','spoken_languages_French',
          'spoken_languages_Spanish','spoken_languages_other',
          'dir_total', 'dir_blockbuster', 'dir_success', 'dir_flop',
          'actor_total', 'actor_blockbuster', 'actor_success', 'actor_flop',
          'dir_act_total', 'dir_act_blockbuster', 'dir_act_success']

def extract_knn_fields(x):
    '''
    return x[['runtime','budget',
          'Action','Adventure','Animation','Comedy','Crime','Documentary','Drama',
          'Family','Fantasy','History','Horror','Music','Mystery','Romance',
          'Science Fiction','TV Movie','Thriller','War','Western',
          'production_companies_20th Century Fox','production_companies_Columbia Pictures',
          'production_companies_Metro-Goldwyn-Mayer','production_companies_New Line Cinema',
          'production_companies_Paramount','production_companies_Universal Pictures',
          'production_companies_Walt Disney Pictures','production_companies_Warner Bros. Pictures',
          'production_companies_other',
          'Mapped Value_Africa','Mapped Value_Asia','Mapped Value_Australia',
          'Mapped Value_Europe','Mapped Value_France','Mapped Value_Germany',
          'Mapped Value_India','Mapped Value_North America','Mapped Value_South America',
          'Mapped Value_United Kingdom','Mapped Value_United States of America',
          'spoken_languages_English','spoken_languages_French',
          'spoken_languages_Spanish','spoken_languages_other',
            'dir_total','dir_blockbuster','dir_success','dir_flop',
            'actor_total','actor_blockbuster','actor_success','actor_flop',
            'dir_act_total','dir_act_blockbuster','dir_act_success','dir_act_flop']]
    '''
    return x[selected_columns]

X_train = extract_knn_fields(data_train)
X_validation = extract_knn_fields(data_validation)
X_test = extract_knn_fields(data_test)

y_train = data_train['success_level']
y_validation = data_validation['success_level']
y_test = data_test['success_level']

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the DataFrame
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_validation = pd.DataFrame(scaler.fit_transform(X_validation), columns=X_validation.columns)
X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)

df_results = pd.DataFrame(columns=['k', 'accuracy','metric'])

distance_metric = ['minkowski', 'manhattan']

def combination():
    #loop
    max_combination=['']

    max_accuracy=0
    for r in range(1, len(X_train) + 1):
            print("r",r)
            for combination in combinations(X_train[-[max_combination]], 1):
                # Select the columns for the current combination
                print(r," ",list(max_combination, combination))
                X_selected_data_train = X_train[list(max_combination, combination)]
                X_selected_data_test = X_test[list(max_combination, combination)]

                # Perform operations with the selected columns
                knn = KNeighborsClassifier(n_neighbors=23)
                knn.fit(X_selected_data_train, y_train)

                # Make predictions on the test set
                y_pred = knn.predict(X_selected_data_test)

                # Calculate accuracy
                accuracy = accuracy_score(y_pred, y_test)
                # print("k=",k," Accuracy:", accuracy)
                new_row = {'k': 23, 'accuracy': accuracy, 'combination': combination}
                df_results = df_results._append(new_row, ignore_index=True)
                if accuracy>max_accuracy:
                    max_accuracy=accuracy
                    max_combination=combination
                    print(new_row, accuracy)

    print(df_results[df_results['accuracy']==df_results['accuracy'].max()])

def simple_knn():
    for i in distance_metric:
        for k in range(1,25):
            # Create a KNN classifier with n_neighbors=k
            knn = KNeighborsClassifier(n_neighbors=k, metric=i)
            knn.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred = knn.predict(X_validation)

            # Calculate accuracy
            accuracy = accuracy_score(y_pred, y_validation)
            #print("k=",k," Accuracy:", accuracy)
            new_row = {'k': k, 'accuracy': accuracy, 'metric':knn.metric}
            df_results = df_results._append(new_row, ignore_index=True)


    max = df_results[df_results['accuracy']==df_results['accuracy'].max()]
    print("max", max)
    print("max k", max['k'])
    print("max metric", max['metric'])
    print("y_train", y_train)
    knn = KNeighborsClassifier(n_neighbors=23, metric='manhattan')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize by row sum

    # Plot confusion matrix using seaborn heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percentage, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=['Blockbuster', 'Flop', 'Success'],
                yticklabels=['Blockbuster', 'Flop', 'Success'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Percentage)')
    plt.show()
    print("")
    #print(df_results)
    ###
    from sklearn.metrics import f1_score

    f1_per_class = f1_score(y_test, y_pred, average=None)
    print("F1 Score per class:", f1_per_class)
    f1_micro = f1_score(y_test, y_pred, average='micro')
    print("Micro F1 Score:", f1_micro)

    #knn elbow
    x1 = df_results[df_results['metric']=='minkowski']['k']
    y1 = df_results[df_results['metric']=='minkowski']['accuracy']

    x2 = df_results[df_results['metric']=='manhattan']['k']
    y2 = df_results[df_results['metric']=='manhattan']['accuracy']

    plt.plot(x1, y1, 'ro', label='minkowski')  # Plot the first set with red circles
    plt.plot(x2, y2, 'bs', label='manhattan')  # Plot the second set with blue squares
    # Add labels and title
    plt.xlabel('Size of K')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs K')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()

    from sklearn.metrics import f1_score
    f1_per_class = f1_score(y_test, y_pred, average=None)
    print("F1 Score per class:", f1_per_class)
    f1_micro = f1_score(y_test, y_pred, average='micro')
    print("Micro F1 Score:", f1_micro)


###

def knn_kbest_kfold():
    from sklearn.feature_selection import chi2, SelectKBest, f_classif

    df_kbest_summary = pd.DataFrame(columns=['scores', 'column_name', 'num_columns'])
    df_kbest_metric_summary = pd.DataFrame(columns=['scores', 'columns', 'k','metric'])

    #parameter={'n_neighbors': np.arange(1, 5, 1)}
    parameter={'n_neighbors': np.arange(1, 26, 1)}
    kf=KFold(n_splits=5,shuffle=True,random_state=42)


    for i in range(1,26):
    #for i in range(1, 5):
        ft = SelectKBest(chi2, k = i).fit(X_train, y_train)
        df_kbest = pd.DataFrame()
        df_kbest['column_name'] = X_train.columns
        df_kbest['num_columns'] = i
        df_kbest['scores'] = ft.scores_
        df_kbest = df_kbest.sort_values(by='scores',ascending=False)
        df_kbest = df_kbest.head(i)

        #save to summary for later analysis
        df_kbest_summary = df_kbest_summary._append(df_kbest)

        X_train_selected = X_train[df_kbest['column_name']]
        X_validation_selected = X_validation[df_kbest['column_name']]
        X_test_selected = X_test[df_kbest['column_name']]

        for j in distance_metric:
            knn = KNeighborsClassifier(metric=j)
            knn_cv=GridSearchCV(knn, param_grid=parameter, cv=kf, verbose=1, scoring='accuracy')
            knn_cv.fit(X_train_selected, y_train)
            validation_score=knn_cv.score(X_validation_selected, y_validation)
            test_score=knn_cv.score(X_test_selected, y_test)

            new_row = { 'num_columns': i, 'k':knn_cv.best_params_['n_neighbors'], 'metric': j,
                        'train_score': knn_cv.best_score_,
                        'validation_score':validation_score,
                        'test_score':test_score}
            print("new_row",new_row)

            df_kbest_metric_summary = df_kbest_metric_summary._append(new_row, ignore_index=True)

    best_test = df_kbest_metric_summary.iloc[df_kbest_metric_summary['test_score'].idxmax()]


    print("best_test", best_test)
    print(df_kbest_summary[df_kbest_summary['num_columns']==best_test['num_columns']])

    #plot k elbow curve

    print(df_kbest_metric_summary[df_kbest_metric_summary['metric']=='manhattan'])
    print(df_kbest_metric_summary[df_kbest_metric_summary['metric']=='manhattan'])

    x1 = df_kbest_metric_summary[df_kbest_metric_summary['metric']=='minkowski']['k']
    y1 = df_kbest_metric_summary[df_kbest_metric_summary['metric']=='minkowski']['test_score']

    x2 = df_kbest_metric_summary[df_kbest_metric_summary['metric']=='manhattan']['k']
    y2 = df_kbest_metric_summary[df_kbest_metric_summary['metric']=='manhattan']['test_score']

    plt.plot(x1, y1, 'ro', label='minkowski')  # Plot the first set with red circles
    plt.plot(x2, y2, 'bs', label='manhattan')  # Plot the second set with blue squares

    # Add labels and title
    plt.xlabel('Size of K')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs K')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()

    #
    x1 = df_kbest_metric_summary[df_kbest_metric_summary['metric']=='minkowski']['num_columns']
    y1 = df_kbest_metric_summary[df_kbest_metric_summary['metric']=='minkowski']['test_score']

    x2 = df_kbest_metric_summary[df_kbest_metric_summary['metric']=='manhattan']['num_columns']
    y2 = df_kbest_metric_summary[df_kbest_metric_summary['metric']=='manhattan']['test_score']

    plt.plot(x1, y1, 'ro', label='minkowski')  # Plot the first set with red circles
    plt.plot(x2, y2, 'bs', label='manhattan')  # Plot the second set with blue squares

    # Add labels and title
    plt.xlabel('Number of Columns')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Number of Columns')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()


    #cm of best
    X_train_selected = X_train[df_kbest_summary[df_kbest_summary['num_columns']==24]['column_name']]
    X_test_selected = X_test[df_kbest_summary[df_kbest_summary['num_columns']==24]['column_name']]

    knn = KNeighborsClassifier(n_neighbors=13)
    knn.fit(X_train_selected, y_train)

    # Make predictions on the test set
    y_pred = knn.predict(X_test_selected)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize by row sum

    # Plot confusion matrix using seaborn heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percentage, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=['Blockbuster', 'Flop', 'Success'],
                yticklabels=['Blockbuster', 'Flop', 'Success'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Percentage)')
    plt.show()
    '''