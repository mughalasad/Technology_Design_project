import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the new dataset for clustering and visualization
@st.cache_data
def load_data(url):
    return pd.read_csv(url)

# URL to the dataset (replace with the actual URL)
data_url = 'df.csv'  # Replace with the actual URL to your dataset

df = load_data(data_url)

# Select the features used in the base dataset
features = ['cvssv3', 'attackvector', 'attackcomplexity', 'privilegesrequired', 'userinteraction', 'scope', 'confidentialityimpact', 'integrityimpact', 'availabilityimpact']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# Fit the KMeans model
kmeans = KMeans(n_clusters=5, random_state=42)  # Adjusted n_clusters to 5
df['cluster'] = kmeans.fit_predict(X_scaled)  # Use cluster labels directly

# Function to predict the cluster for new data
def predict_cluster(new_data):
    # Apply label encoding to user input data
    for feature in features:
        if feature != 'cvssv3':
            le = LabelEncoder()
            new_data[feature] = le.fit_transform(new_data[feature])
    new_data_scaled = scaler.transform(new_data[features])
    return kmeans.predict(new_data_scaled)

# Streamlit app
st.title("KMeans Clustering Prediction and Data Visualization")

# Take user input for the feature values
user_input = {}
user_input['report'] = st.text_area("Enter report:")  # Add a text area for the report
user_input['cvssv3'] = st.number_input("Enter value for cvssv3:", min_value=0.0, max_value=10.0, step=0.1)
user_input['attackvector'] = st.selectbox("Select value for attackvector:", ['NETWORK', 'ADJACENT', 'LOCAL', 'PHYSICAL'])
user_input['attackcomplexity'] = st.selectbox("Select value for attackcomplexity:", ['LOW', 'HIGH'])
user_input['privilegesrequired'] = st.selectbox("Select value for privilegesrequired:", ['NONE', 'LOW', 'HIGH'])
user_input['userinteraction'] = st.selectbox("Select value for userinteraction:", ['NONE', 'REQUIRED'])
user_input['scope'] = st.selectbox("Select value for scope:", ['UNCHANGED', 'CHANGED'])
user_input['confidentialityimpact'] = st.selectbox("Select value for confidentialityimpact:", ['NONE', 'LOW', 'HIGH'])
user_input['integrityimpact'] = st.selectbox("Select value for integrityimpact:", ['NONE', 'LOW', 'HIGH'])
user_input['availabilityimpact'] = st.selectbox("Select value for availabilityimpact:", ['NONE', 'LOW', 'HIGH'])

# Convert user input to DataFrame
new_data = pd.DataFrame([user_input])

# Remove the 'report' column from new_data before prediction
new_data = new_data.drop(columns=['report'])

# Predict the cluster for the user input
if st.button("Predict Threat score level"):
    predicted_clusters = predict_cluster(new_data)
    st.write(f"Predicted Threat score level: {predicted_clusters[0]}")

# Sidebar for plot selection
st.sidebar.title("Visualization Options")
plot_type = st.sidebar.selectbox("Select plot type:", ["Histogram", "Correlation Heatmap", "Pair Plot", "Cluster Plot"])
selected_columns = st.sidebar.multiselect("Select columns to visualize:", df.columns.tolist())

# Display the selected plot
if plot_type == "Histogram":
    st.header("Histogram")
    for col in selected_columns:
        st.subheader(f"Histogram for {col}")
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

elif plot_type == "Correlation Heatmap":
    st.header("Correlation Heatmap")
    if len(selected_columns) > 1:
        fig, ax = plt.subplots()
        corr = df[selected_columns].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Please select at least two columns for the correlation heatmap.")

elif plot_type == "Pair Plot":
    st.header("Pair Plot")
    if len(selected_columns) > 1:
        fig = sns.pairplot(df[selected_columns])
        st.pyplot(fig)
    else:
        st.warning("Please select at least two columns for the pair plot.")

elif plot_type == "Cluster Plot":
    st.header("Cluster Plot")
    if len(selected_columns) == 2:
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=selected_columns[0], y=selected_columns[1], hue='cluster', palette='viridis', ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Please select exactly two columns for the cluster plot.")