import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV

app = Flask(__name__)

df = pd.read_csv('train.csv')
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'].fillna('S'))

numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

@app.route('/', methods=['GET', 'POST'])
def index():
    survived_count = df[df['Survived'] == 1].shape[0]
    result = None
    invalid_input = False
    record_exists = False

    if request.method == 'POST':
        pclass = request.form.get('Pclass')
        sex = request.form.get('Sex')
        age = request.form.get('Age')
        sibsp = request.form.get('SibSp')
        parch = request.form.get('Parch')
        fare = request.form.get('Fare')
        embarked = request.form.get('Embarked')

        try:
            pclass = int(pclass)
            sex = 1 if sex.lower() == 'male' else 0
            age = float(age)
            sibsp = int(sibsp)
            parch = int(parch)
            fare = float(fare)
            embarked_dict = {'S': 2, 'C': 0, 'Q': 1}
            embarked = embarked_dict.get(embarked.upper(), -1)

            if embarked == -1 or pclass not in [1, 2, 3]:
                raise ValueError

            # Prepare input data for prediction
            input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
            prediction = model.predict(input_data)

            result = 'Survived' if prediction == 1 else 'Not Survived'

        except ValueError:
            result = 'Invalid Input'
            invalid_input = True

    return render_template('index.html', result=result, survived_count=survived_count, invalid_input=invalid_input, record_exists=record_exists)

@app.route('/alldata.html')
def all_data():
    row_count = request.args.get('row_count', type=int)
    data = df.head(row_count).to_dict(orient='records')
    count = len(data)
    return render_template('alldata.html', data=data, count=count)

@app.route('/statistics.html')
def statistics():
    statistics_data = {
        'Survived': df['Survived'].describe(),
        'Pclass': df['Pclass'].describe(),
        'Age': df['Age'].describe(),
        'SibSp': df['SibSp'].describe(),
        'Parch': df['Parch'].describe(),
        'Fare': df['Fare'].describe()
    }
    total_passengers = df.shape[0]
    survived_count = df['Survived'].sum()
    not_survived_count = total_passengers - survived_count

    plt.figure(figsize=(10, 6))
    plt.bar(['Survived', 'Not Survived'], [survived_count, not_survived_count])
    plt.xlabel('Outcome')
    plt.ylabel('Count')
    plt.title('Survival Outcome Count')
    plt.savefig('static/survival_plot.png') 

    return render_template('statistics.html', statistics_data=statistics_data, survived_count=survived_count, not_survived_count=not_survived_count)

def categorize_who(row):
    if row['Sex'] == 1 and row['Age'] > 18:
        return 'man'
    elif row['Sex'] == 1 and row['Age'] <= 18:
        return 'child'
    elif row['Sex'] == 0 and row['Age'] > 18:
        return 'woman'
    else:
        return 'child'

df['who'] = df.apply(categorize_who, axis=1)

@app.route('/plotanalysis.html')
def plotanalysis():
    cols = ['Sex', 'Pclass', 'SibSp', 'Parch', 'Embarked', 'who']
    n_rows = 2
    n_cols = 3

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    for r in range(0, n_rows):
        for c in range(0, n_cols):
            i = r * n_cols + c  
            if i < len(cols):  
                ax = axs[r][c] 
               
                plot_data = df[[cols[i], 'Survived']].copy()
                plot_data['count'] = 1
                plot_data = plot_data.groupby([cols[i], 'Survived']).count().reset_index()
                sns.barplot(x=cols[i], y='count', hue='Survived', data=plot_data, ax=ax)
                ax.set_title(cols[i])
                ax.legend(title='Survived', loc='upper right')
    plt.tight_layout()

    for i, col in enumerate(cols):
        plt.figure(i)
        sns.barplot(x=col, y='count', hue='Survived', data=df.groupby([col, 'Survived']).size().reset_index(name='count'))
        plt.title(col)
        plt.legend(title='Survived', loc='upper right')
        plt.tight_layout()
        plt.savefig(f'static/img/{col}.png')

    plt.clf()

    g = sns.FacetGrid(df, col='who', row='Survived', margin_titles=True, height=4)
    g.map(plt.hist, 'Age', bins=20, alpha=0.6)
    g.add_legend()
    plt.savefig('static/img/Age.png')

    plt.close()

    return render_template('plotanalysis.html')

@app.route('/eda.html')
def eda_analysis():
    missing_values = df.isnull().sum()
    missing_values_dict = missing_values.to_dict()

    plt.figure(figsize=(12, 6))
    df['Age'].hist(bins=30)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.savefig('static/img/age_distribution.png')

    plt.figure(figsize=(8, 4))
    sns.countplot(x='Sex', data=df)
    plt.title('Sex Distribution')
    plt.savefig('static/img/sex_distribution.png')

    plt.figure(figsize=(8, 4))
    sns.barplot(x='Sex', y='Survived', data=df)
    plt.title('Survival by Sex')
    plt.savefig('static/img/survival_by_sex.png')

    return render_template('eda.html', missing_values_dict=missing_values_dict)

@app.route('/correlation.html')
def correlation():
    numeric_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    correlation_matrix = df[numeric_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=.5)
    plt.title('Correlation Matrix')
    plt.savefig('static/img/correlation_matrix.png')
    plt.close()

    return render_template('correlation.html')

@app.route('/acc.html')
def accuracy():
    df = pd.read_csv('train.csv')

    X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    y = df['Survived']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
    numeric_transformer = SimpleImputer(strategy='mean')

    categorical_features = ['Sex', 'Embarked']
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    model = GradientBoostingClassifier(random_state=42)

    param_grid = {
        'preprocessor__num__strategy': ['mean', 'median'],
        'clf__n_estimators': [50, 100, 200],
        'clf__learning_rate': [0.1, 0.05, 0.01],
        'clf__max_depth': [3, 4, 5]
    }

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', model)
    ])

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    print(f'Accuracy: {accuracy}')

    test_data = pd.read_csv('test.csv')

    X_test = test_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

    test_predictions = best_model.predict(X_test)

    submission = pd.DataFrame({
        'PassengerId': test_data['PassengerId'],
        'Survived': test_predictions
    })

    submission.to_csv('submission.csv', index=False)

    return render_template('acc.html', accuracy=accuracy)


if __name__ == '__main__':
    app.run(debug=True)
