import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Example dataframe (replace with actual dataframe)
df = pd.DataFrame({
    'cvssv3': [9.8, 7.5, 7.5, 9.8, 5.5, 6.0, 4.0, 3.5, 8.0, 2.0],
    'attackvector': ['NETWORK', 'NETWORK', 'NETWORK', 'NETWORK', 'LOCAL', 'LOCAL', 'PHYSICAL', 'PHYSICAL', 'ADJACENT', 'ADJACENT'],
    'attackcomplexity': ['LOW', 'LOW', 'LOW', 'LOW', 'LOW', 'HIGH', 'HIGH', 'HIGH', 'LOW', 'LOW'],
    'privilegesrequired': ['NONE', 'NONE', 'NONE', 'NONE', 'LOW', 'LOW', 'HIGH', 'HIGH', 'NONE', 'NONE'],
    'userinteraction': ['NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'REQUIRED', 'REQUIRED', 'REQUIRED', 'NONE', 'NONE'],
    'scope': ['UNCHANGED', 'UNCHANGED', 'UNCHANGED', 'UNCHANGED', 'UNCHANGED', 'CHANGED', 'CHANGED', 'CHANGED', 'UNCHANGED', 'UNCHANGED'],
    'confidentialityimpact': ['HIGH', 'HIGH', 'HIGH', 'HIGH', 'NONE', 'LOW', 'LOW', 'LOW', 'HIGH', 'HIGH'],
    'integrityimpact': ['HIGH', 'NONE', 'NONE', 'HIGH', 'NONE', 'LOW', 'LOW', 'LOW', 'HIGH', 'HIGH'],
    'availabilityimpact': ['HIGH', 'NONE', 'NONE', 'HIGH', 'HIGH', 'LOW', 'LOW', 'LOW', 'HIGH', 'HIGH']
})

# Select the features used in the base dataset
features = ['cvssv3', 'attackvector', 'attackcomplexity', 'privilegesrequired', 'userinteraction', 'scope', 'confidentialityimpact', 'integrityimpact', 'availabilityimpact']

# Define all possible values for each categorical feature
possible_values = {
    'attackvector': ['NETWORK', 'ADJACENT', 'LOCAL', 'PHYSICAL'],
    'attackcomplexity': ['LOW', 'HIGH'],
    'privilegesrequired': ['NONE', 'LOW', 'HIGH'],
    'userinteraction': ['NONE', 'REQUIRED'],
    'scope': ['UNCHANGED', 'CHANGED'],
    'confidentialityimpact': ['NONE', 'LOW', 'HIGH'],
    'integrityimpact': ['NONE', 'LOW', 'HIGH'],
    'availabilityimpact': ['NONE', 'LOW', 'HIGH']
}

# Encode categorical features
label_encoders = {}
for feature in features:
    if df[feature].dtype == 'object':
        le = LabelEncoder()
        le.fit(possible_values[feature])
        df[feature] = le.transform(df[feature])
        label_encoders[feature] = le

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# Fit the KMeans model
kmeans = KMeans(n_clusters=5, random_state=42)  # Adjusted n_clusters to match the sample size
df['predicted_bin'] = kmeans.fit_predict(X_scaled) + 1  # Adjust bin range to be from 1 to 10

# Function to predict the bin for new data
def predict_bin(new_data):
    for feature in features:
        if feature in label_encoders:
            new_data[feature] = label_encoders[feature].transform(new_data[feature])
    new_data_scaled = scaler.transform(new_data[features])
    return kmeans.predict(new_data_scaled) + 1  # Adjust bin range to be from 1 to 10

# Streamlit app
st.title("KMeans Clustering Prediction and Data Visualization")

# Take user input for the feature values
user_input = {}
user_input['report'] = st.text_area("Enter report:")  # Add a text area for the report
user_input['cvssv3'] = st.number_input("Enter value for cvssv3:", min_value=0.0, max_value=10.0, step=0.1)
user_input['attackvector'] = st.selectbox("Select value for attackvector:", possible_values['attackvector'])
user_input['attackcomplexity'] = st.selectbox("Select value for attackcomplexity:", possible_values['attackcomplexity'])
user_input['privilegesrequired'] = st.selectbox("Select value for privilegesrequired:", possible_values['privilegesrequired'])
user_input['userinteraction'] = st.selectbox("Select value for userinteraction:", possible_values['userinteraction'])
user_input['scope'] = st.selectbox("Select value for scope:", possible_values['scope'])
user_input['confidentialityimpact'] = st.selectbox("Select value for confidentialityimpact:", possible_values['confidentialityimpact'])
user_input['integrityimpact'] = st.selectbox("Select value for integrityimpact:", possible_values['integrityimpact'])
user_input['availabilityimpact'] = st.selectbox("Select value for availabilityimpact:", possible_values['availabilityimpact'])

# Convert user input to DataFrame
new_data = pd.DataFrame([user_input])

# Remove the 'report' column from new_data before prediction
new_data = new_data.drop(columns=['report'])

# Predict the bin for the user input
if st.button("Predict Threat score level"):
    predicted_bins = predict_bin(new_data)
    st.write(f"Predicted Threat score level: {predicted_bins[0]}")

# Load the new dataset for visualization
@st.cache_data
def load_data(url):
    return pd.read_csv(url)

# URL to the dataset (replace with the actual URL)
data_url = 'df.csv'  # Replace with the actual URL to your dataset

df_new = load_data(data_url)

# Sidebar for plot selection
st.sidebar.title("Visualization Options")
plot_type = st.sidebar.selectbox("Select plot type:", ["Histogram", "Correlation Heatmap", "Pair Plot"])
selected_columns = st.sidebar.multiselect("Select columns to visualize:", df_new.columns.tolist())

# Display the selected plot
if plot_type == "Histogram":
    st.header("Histogram")
    for col in selected_columns:
        st.subheader(f"Histogram for {col}")
        fig, ax = plt.subplots()
        sns.histplot(df_new[col], kde=True, ax=ax)
        st.pyplot(fig)

elif plot_type == "Correlation Heatmap":
    st.header("Correlation Heatmap")
    if len(selected_columns) > 1:
        fig, ax = plt.subplots()
        corr = df_new[selected_columns].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Please select at least two columns for the correlation heatmap.")

elif plot_type == "Pair Plot":
    st.header("Pair Plot")
    if len(selected_columns) > 1:
        fig = sns.pairplot(df_new[selected_columns])
        st.pyplot(fig)
    else:
        st.warning("Please select at least two columns for the pair plot.")