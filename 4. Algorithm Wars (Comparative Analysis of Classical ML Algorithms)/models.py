from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

def get_models(preprocessor_type = 'tabular'):
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=37),
        'Decision Tree': DecisionTreeClassifier(random_state=37),
        'Random Forest': RandomForestClassifier(random_state=37),
        'SVM (RBF)': SVC(random_state=37),
        'SVM (Linear)': LinearSVC(max_iter=5000, dual='auto', random_state=37),
        'K-Nearest Neighbors': KNeighborsClassifier(),   
        'Gradient Boosting': GradientBoostingClassifier(random_state=37),
        'XGBoost': XGBClassifier(random_state=37, eval_metric='logloss'),
    }

    if preprocessor_type == 'text':
        models['Multinomial Naive Bayes'] = MultinomialNB()
    elif preprocessor_type == 'tabular':
        models['Gaussian Naive Bayes'] = GaussianNB()

    return models
