import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

st.title("📊 Student Exam Result Classifier")

# --- Load your dataset directly here ---
df = pd.read_csv("student_habits_performance.csv")  # Replace with your actual CSV filename
st.subheader("🔍 Data Preview")
st.dataframe(df.head())

# --- Encode categorical variables ---
categorical_cols = ['gender', 'part_time_job', 'diet_quality', 'exercise_frequency',
                    'parental_education_level', 'internet_quality', 'mental_health_rating',
                    'extracurricular_participation']

le = LabelEncoder()
for col in categorical_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col])

# --- Feature-target split ---
X = df.drop(['student_id', 'exam_score'], axis=1)
y = (df['exam_score'] >= 50).astype(int)  # 1 = Pass, 0 = Fail

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model selection ---
st.subheader("⚙️ Select a Model")
model_name = st.selectbox("Choose a model", ['Logistic Regression', 'Decision Tree', 'KNN'])

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier()
}

model = models[model_name]

# --- Train and evaluate the model ---
model.fit(X_train, y_train)
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds) * 100
cm = confusion_matrix(y_test, preds)

# --- Show accuracy ---
st.subheader(f"✅ {model_name} Accuracy: {acc:.2f}%")

# --- Show confusion matrix ---
fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fail', 'Pass'])
disp.plot(ax=ax, cmap='Blues', values_format='d')
ax.set_title(f"Confusion Matrix - {model_name}")
st.pyplot(fig)
