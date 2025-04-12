import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Streamlit UI
st.title("Twitter Sentiment Analysis")

# Upload datasets
train_file = st.file_uploader("Upload Train Dataset", type=["csv"])
test_file = st.file_uploader("Upload Test Dataset", type=["csv"])

df_train = pd.read_csv(train_file) if train_file else None
df_test = pd.read_csv(test_file) if test_file else None

if df_train is not None:
    st.write("Train dataset loaded successfully.", df_train.head())

if df_test is not None:
    st.write("Test dataset loaded successfully.", df_test.head())


st.write("### Columns in Train Data:")
st.write(df_train.columns.tolist())  # Show column names as a list

st.write("### Columns in Test Data:")
st.write(df_test.columns.tolist())

# Preprocess Data
def preprocess_data(df):
    df['tweet'] = df['tweet'].str.replace("[^a-zA-Z#]", " ", regex=True)
    df['tweet'] = df['tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))
    return df

if st.button("Preprocess Train Data") and df_train is not None:
    df_train = preprocess_data(df_train)
    st.write("Train dataset preprocessed.", df_train.head())

if st.button("Preprocess Test Data") and df_test is not None:
    df_test = preprocess_data(df_test)
    st.write("Test dataset preprocessed.", df_test.head())

# Function to train and evaluate models
def run_algorithm(df_train, algorithm):
    if 'label' not in df_train.columns:
        st.warning("Train dataset must contain a 'label' column.")
        return

    X_train = df_train['tweet']
    y_train = df_train['label']

    vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
    X_train = vectorizer.fit_transform(X_train)

    model_dict = {
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(),
        "Logistic Regression": LogisticRegression(),
        "SVM": svm.SVC(kernel='linear', C=1, probability=True)
    }
    
    model = model_dict.get(algorithm)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X_train, y_train, cv=skf)

    accuracy = accuracy_score(y_train, y_pred)
    precision = precision_score(y_train, y_pred, average='weighted')
    recall = recall_score(y_train, y_pred, average='weighted')
    f1 = f1_score(y_train, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_train, y_pred)

    st.write(f"{algorithm} Results:")
    st.write(f"Accuracy: {accuracy}")
    st.write(f"Precision: {precision}")
    st.write(f"Recall: {recall}")
    st.write(f"F1 Score: {f1}")

    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap=plt.cm.Blues, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    st.pyplot(fig)

# Buttons for running models
algorithm = st.selectbox("Select Algorithm", ["Naive Bayes", "Random Forest", "Logistic Regression", "SVM"])
if st.button("Run Algorithm") and df_train is not None:
    run_algorithm(df_train, algorithm)

# Word Cloud
if st.button("Generate Word Cloud") and df_train is not None:
    positive_tweets = df_train[df_train['label'] == 0]['tweet']
    negative_tweets = df_train[df_train['label'] == 1]['tweet']
    
    positive_wordcloud = WordCloud(width=1000, height=500, max_words=100).generate(' '.join(positive_tweets))
    negative_wordcloud = WordCloud(width=1000, height=500, max_words=100).generate(' '.join(negative_tweets))
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(positive_wordcloud, interpolation='bilinear')
    axes[0].axis('off')
    axes[0].set_title('Postive Words')
    
    axes[1].imshow(negative_wordcloud, interpolation='bilinear')
    axes[1].axis('off')
    axes[1].set_title('Negative Words')
    
    st.pyplot(fig)
