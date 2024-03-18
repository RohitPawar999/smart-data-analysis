import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
import os
import re
import base64
from io import BytesIO
from sklearn.decomposition import PCA
from scipy.stats import boxcox
import shap
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from scipy import stats
from tpot import TPOTClassifier
from tpot import TPOTRegressor
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,f1_score
import openai
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer


st.set_page_config(page_title="Data Analysis Tool", page_icon="📊")
st.header("Smart Data Analysis 📊")


openai.api_key = "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"

if 'history' not in st.session_state:
    st.session_state.history = []


def print_null_sum(dataset):
    st.subheader("Sum of Null Values:")
    st.write(dataset.isnull().sum())
def header(dataset):
    st.write(f"Dataset head:")
    st.write(dataset.head())





def perform_analysis(dataset):
    st.subheader("Pandas Profiling Report:")
    report = ProfileReport(dataset, title='My Data')
    report_filename = os.path.join(os.getcwd(), "my_report.html")
    report.to_file(report_filename)
    with open(report_filename, "rb") as file:
        encoded_report = base64.b64encode(file.read()).decode()
    st.markdown(
        f"Download the detailed report: <a href='data:file/html;base64,{encoded_report}' download='my_report.html'>Download Report</a>",
        unsafe_allow_html=True)


def draw_graph(graph_type, x_axis, y_axis, dataset, z_axis=None):
    numeric_columns = dataset.select_dtypes(include=[int, float]).columns
    for col in numeric_columns:
        dataset[col] = dataset[col].fillna(np.random.choice(dataset[col].dropna()))
        dataset.drop_duplicates(inplace=True)

    label_encoder = LabelEncoder()
    for col in dataset.select_dtypes(include=['object']).columns:
        dataset[col] = label_encoder.fit_transform(dataset[col])

    dataset = dataset.astype(int)


    st.subheader("Graph:")
    fig, ax = plt.subplots(figsize=(20, 10))

    if graph_type == "Line Plot":
        ax.plot(dataset[x_axis], dataset[y_axis])
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_title("Line Plot")

    elif graph_type == "Bar Plot":
        ax.bar(dataset[x_axis], dataset[y_axis])
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_title("Bar Plot")

    elif graph_type == "Histogram":
        ax.hist(dataset[x_axis], bins=10)
        ax.set_xlabel(x_axis)
        ax.set_ylabel("Frequency")
        ax.set_title("Histogram")

    elif graph_type == "Pair Plot":
        sns.set_theme()
        sns.pairplot(dataset)
        ax.set_title("Pair Plot")

    elif graph_type == "Scatter Plot":
        ax.scatter(dataset[x_axis], dataset[y_axis])
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_title("Scatter Plot")

    elif graph_type == "Boxplot":
        sns.boxplot(x=x_axis, y=y_axis, data=dataset)
        ax.set_title("Boxplot")

    elif graph_type == "Violin Plot":
        sns.violinplot(x=x_axis, y=y_axis, data=dataset)
        ax.set_title("Violin Plot")

    elif graph_type == "Eventplot":
        ax.eventplot(dataset[x_axis], lineoffsets=0, colors='b')
        ax.set_title("Eventplot")

    elif graph_type == "Hexbin":
        ax.hexbin(dataset[x_axis], dataset[y_axis], gridsize=25, cmap='Blues')
        ax.set_title("Hexbin Plot")

    elif graph_type == "Pie Chart":
        st.warning("Pie Chart requires only one input. Please select only one column.")
        if x_axis:
            ax.pie(dataset[x_axis], labels=dataset.index, autopct='%1.1f%%', startangle=140)
            ax.set_title("Pie Chart")

    elif graph_type == "ECDF":
        n, bins, patches = ax.hist(dataset[x_axis], cumulative=True, bins=25, density=True, alpha=0.8)
        ax.set_title("Empirical Cumulative Distribution Function (ECDF)")

    elif graph_type == "2D Histogram":
        ax.hist2d(dataset[x_axis], dataset[y_axis], bins=25, cmap='Blues')
        ax.set_title("2D Histogram")

    elif graph_type == "3D Scatter Plot":
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(dataset[x_axis], dataset[y_axis], dataset[z_axis])
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_zlabel(z_axis)
        ax.set_title("3D Scatter Plot")

    elif graph_type == "3D Surface Plot-1":
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(dataset[x_axis], dataset[y_axis], dataset[z_axis], cmap='viridis')
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_zlabel(z_axis)
        ax.set_title("3D Surface Plot")

    else:
        st.warning("Invalid graph type selected.")
        return

    # Save the plot as an image
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    st.image(buf, use_column_width=True)

def display_statistics(dataset):
    st.subheader("Statistical Analysis:")
    st.write(dataset.describe())


def run_model( X_train, X_test, y_train, y_test,select_algo):
    # Assuming you have a dataset and target_column

    if select_algo == "Classification":
        # Assuming you have more than two classes in your target variable
        tpot_classifier = TPOTClassifier(
            generations=5,
            population_size=20,
            random_state=42,
            verbosity=2,
            config_dict='TPOT sparse',
            scoring='accuracy',  # Choose an appropriate scoring metric for multi-class classification
            n_jobs=-1  # Use all available CPU cores
        )

        # Fit AutoML model to the data
        tpot_classifier.fit(X_train, y_train)

        # Make predictions on the test set
        predictions = tpot_classifier.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        classification_report_str = classification_report(y_test, predictions)
        confusion_mat = confusion_matrix(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')

        st.write(f"Accuracy: {accuracy}")
        st.write("Classification Report:")
        st.write(classification_report_str)
        st.write("Confusion Matrix:")
        st.write(confusion_mat)
        st.write(f"F1-score: {f1}")

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=tpot_classifier.classes_, yticklabels=tpot_classifier.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')

        # Pass the Matplotlib figure to st.pyplot()
        st.pyplot(plt.gcf())

        explainer = shap.TreeExplainer(tpot_classifier.fitted_pipeline_.steps[-1][1])
        shap_values = explainer.shap_values(X_test)

        # Plot SHAP summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, plot_type="bar")
        st.pyplot(plt.gcf())

        # Access the best pipeline and display details
        best_pipeline = tpot_classifier.fitted_pipeline_
        st.write("Best Model:")
        st.write(best_pipeline.steps[-1])
    elif select_algo=="Regression":
        tpot_regressor = TPOTRegressor(verbosity=2, generations=5, population_size=20, random_state=42,
                                       config_dict='TPOT sparse', template='Regressor')

        # Fit TPOT regressor to the data
        tpot_regressor.fit(X_train, y_train)

        # Make predictions on the test set
        predictions = tpot_regressor.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

        # Display metrics
        st.write(f"Mean Squared Error: {mse}")
        st.write(f"Mean Absolute Error: {mae}")
        st.write(f"R-squared: {r2}")
        st.write(f"MAPE: {mape}%")

        explainer = shap.TreeExplainer(tpot_regressor.fitted_pipeline_.steps[-1][1])
        shap_values = explainer.shap_values(X_test)

        # Plot SHAP summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, plot_type="bar")
        st.pyplot(plt.gcf())

        # Visualization: Predicted vs Actual values
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=predictions)
        plt.title("Predicted vs Actual Values")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        st.pyplot(plt.gcf())
        best_pipeline = tpot_regressor.fitted_pipeline_
        st.write("Best Model:")
        st.write(best_pipeline.steps[-1])



def NLP_preprosseing(dataset, target_1, test_size_percentage_1):
    dataset.drop_duplicates(inplace=True)
    y = dataset[target_1]
    dataset = dataset.drop(target_1, axis=1)

    def remove_tags(raw_text):
        cleaned_text = re.sub(re.compile('<.*?>'), '', raw_text)
        return cleaned_text

    non_numeric_columns = dataset.select_dtypes(exclude=[int, float]).columns

    for col in non_numeric_columns:
        dataset[col] = dataset[col].apply(remove_tags)
        dataset[col] = dataset[col].apply(lambda x: x.lower())
        sw_list =  ENGLISH_STOP_WORDS
        dataset[col] = dataset[col].apply(lambda x: ' '.join([item for item in x.split() if item not in sw_list]))


    numeric_columns = dataset.select_dtypes(include=[int, float]).columns

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    header(dataset)

    st.subheader("Train-Test Split:")
    test_size = test_size_percentage_1/ 100.0
    X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size=test_size, random_state=42)


    st.write(f"Training Set Shape: {X_train.shape}")
    st.write(f"Testing Set Shape: {X_test.shape}")
    cv = CountVectorizer(min_df=2, max_features=5000)

    for col in X_train.columns:
        if X_train[col].dtype == 'O':  # Check if the column contains text data
            X_train[col] = cv.fit_transform(X_train[col]).toarray()

    for col in X_test.columns:
        if X_test[col].dtype == 'O':  # Check if the column contains text data
            X_test[col] = cv.transform(X_test[col]).toarray()

    y_test = pd.DataFrame( y_test)
    y_train = pd.DataFrame(y_train)
    st.session_state.X_train_1 = X_train.copy()
    st.session_state.X_test_1 = X_test.copy()
    st.session_state.y_train_1 = y_train.copy()
    st.session_state.y_test_1 = y_test.copy()


    return  X_train, X_test, y_train, y_test

def NLP_Training(X_train, X_test, y_train, y_test):
    tpot_classifier = TPOTClassifier(
        generations=5,
        population_size=20,
        offspring_size=None,
        verbosity=2,
        config_dict='TPOT sparse',
        scoring='accuracy',
        random_state=42,
        cv=5,
        n_jobs=-1,
        memory='auto',
        early_stop=5,
        periodic_checkpoint_folder=None,
        use_dask=False,
        template=None,
    )

    # Fit TPOT model to the data
    tpot_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = tpot_classifier.predict(X_test)

    # Evaluate the accuracy
    accuracy = metrics.accuracy_score(y_test, predictions)
    st.write(f"TPOT Accuracy: {accuracy}")

def transform_data(data, method):
    if method == "Log Transformation":
        return np.log1p(data)  # Adding 1 to avoid log(0)
    elif method == "Box-Cox Transformation":
        sns.distplot(data, hist=False, kde=True)
        transformed_data, best_lambda = boxcox(data)
        sns.distplot(transformed_data, hist=False, kde=True)
        return transformed_data
    else:
        return data

def preprocess_data(select, dataset, select1, test_size_percentage, columns_to_delete, target_column_3,outlier_handling,use_pca=False, n_components=None,select_transform=None):
    st.subheader("Null Value adjustment and removal of dublicates Preprocessing:")

    numeric_columns = dataset.select_dtypes(include=[np.number]).columns

    if select == "Remove Null Values":
        st.write(" methode:Remove Null Value ")

        dataset.dropna(inplace=True)
        dataset.drop_duplicates(inplace=True)


    elif select == "Add Mean Values":
        st.write(" methode:Add Mean Values ")

        dataset[numeric_columns] = dataset[numeric_columns].fillna(dataset[numeric_columns].mean())
        dataset.drop_duplicates(inplace=True)


    elif select == "Add Random Values from Column":
        st.write(" methode:Add Random Values from Column ")

        for col in numeric_columns:
            dataset[col] = dataset[col].fillna(np.random.choice(dataset[col].dropna()))
            dataset.drop_duplicates(inplace=True)

    header(dataset)

    st.header("Dataset after removing columns")

    if columns_to_delete and all(col in dataset.columns for col in columns_to_delete):
        dataset.drop(columns=columns_to_delete, inplace=True)

    st.session_state.preprocessed_dataset = dataset.copy()

    header(dataset)

    datetime_columns = dataset.select_dtypes(include=['datetime64']).columns
    st.subheader("Handling DateTime Columns if present:")
    for col in datetime_columns:
        # Extract useful features from datetime columns
        dataset[col + '_year'] = dataset[col].dt.year
        dataset[col + '_month'] = dataset[col].dt.month
        dataset[col + '_day'] = dataset[col].dt.day
        dataset[col + '_hour'] = dataset[col].dt.hour
        # You can add more features like minute, second, etc. as needed
    # Drop original datetime columns
    dataset.drop(columns=datetime_columns, inplace=True)
    # Display updated dataset
    header(dataset)



    st.subheader("Applying Encoding:")

    st.write("Encoding method:LabelEncoder ")

    label_encoder = LabelEncoder()
    for col in dataset.select_dtypes(include=['object']).columns:
        dataset[col] = label_encoder.fit_transform(dataset[col])

    dataset = dataset.astype(int)

    header(dataset)

    st.subheader("Outlier Handling:")

    if outlier_handling == "Z-Score":
        st.write("Outlier method: Z_score")

        z_scores = np.abs(stats.zscore(dataset[numeric_columns]))
        dataset = dataset[(z_scores < 3).all(axis=1)]  # Remove rows with z-scores > 3

    elif outlier_handling == "IQR":
        st.write("Outlier method: IQR")

        Q1 = dataset[numeric_columns].quantile(0.25)
        Q3 = dataset[numeric_columns].quantile(0.75)
        IQR = Q3 - Q1
        dataset = dataset[
            ~((dataset[numeric_columns] < (Q1 - 1.5 * IQR)) | (dataset[numeric_columns] > (Q3 + 1.5 * IQR))).any(
                axis=1)]

    header(dataset)

    y = dataset[target_column_3]
    dataset = dataset.drop(target_column_3, axis=1)
    numeric_columns_1 = dataset.select_dtypes(include=[np.number]).columns

    st.subheader("Data Transformation:")

    # Check if a transformation method is selected
    if select_transform:
        st.write(f"Transformation method: {select_transform}")
        if select_transform == "Log Transformation":
            dataset[numeric_columns_1]= np.log1p(dataset[numeric_columns_1])
        if select_transform == "Square root transformation":
            dataset[numeric_columns_1]==np.sqrt(dataset[numeric_columns_1])


    header(dataset)


    st.subheader("Now performing normalization:")

    if select1 == "Standard Scaler":
        st.write("Normalisation : Standard Scaler")

        scaler = StandardScaler()
        dataset[numeric_columns_1] = scaler.fit_transform(dataset[numeric_columns_1])

    elif select1 == "Min-Max Scaler":
        st.write("Normalisation : Min-Max Scaler")

        scaler = MinMaxScaler()
        dataset[numeric_columns_1] = scaler.fit_transform(dataset[numeric_columns_1])

    elif select1 == "Robust Scaler":
        st.write("Normalisation : Robust Scaler")

        scaler = RobustScaler()
        dataset[numeric_columns_1] = scaler.fit_transform(dataset[numeric_columns_1])
    st.session_state.normalized_dataset = dataset.copy()

    header(dataset)

    if use_pca:
        st.subheader("Applying PCA:")

        st.write(f"Number of components in pca: {n_components}")

        pca = PCA(n_components=n_components)
        dataset_pca = pca.fit_transform(dataset)
        # Show explained variance ratio
        st.write("Explained Variance Ratio:")

        st.write(pca.explained_variance_ratio_)

        # Visualize explained variance ratio
        st.bar_chart(pca.explained_variance_ratio_)

        dataset_pca = pd.DataFrame(dataset_pca, columns=[f"PC{i}" for i in range(1, n_components + 1)])
        dataset_pca[target_column_3] = y.values

        # Save PCA dataset to session state
        st.session_state.dataset_pca = dataset_pca.copy()
        dataset = dataset_pca
        dataset = dataset.drop(target_column_3, axis=1)

        # Display headers for PCA dataset
        header(dataset_pca)

    st.subheader("Train-Test Split:")

    test_size = test_size_percentage / 100.0
    X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size=test_size, random_state=42)
    st.session_state.X_train = X_train.copy()
    st.session_state.X_test = X_test.copy()
    st.session_state.y_train = y_train.copy()
    st.session_state.y_test = y_test.copy()

    st.write(f"Training Set Shape: {X_train.shape}")

    header(X_train)

    header(y_train)

    st.write(f"Testing Set Shape: {X_test.shape}")

    header(X_test)

    header(y_test)


def chatgpt(user_prompt):
    openai.api_key = "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"

    if user_prompt:
        messages = [
            {"role": "system", "content": "You are a kind helpful assistant."},
        ]
        # Add user input to messages
        messages.append({"role": "user", "content": user_prompt})  # Corrected parameter name

        # Call OpenAI API
        chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

        # Extract assistant's reply
        assistant_reply = chat.choices[0].message.content

        # Display ChatGPT's reply
        st.sidebar.info(f"Zira: {assistant_reply}")




def get_column_info(df):
    column_info = {}
    for column in df.columns:
        column_info[column] = {
            'Data Type': df[column].dtype,
            'Number of Unique Values': df[column].nunique()
        }
    return column_info

def information(dataset):
    st.write("Uploaded Dataset:")
    st.write(dataset)

    # Display dataset description
    st.markdown("<h2 style='text-align: center; font-size: 28px;'>Dataset Description:</h2>", unsafe_allow_html=True)
    st.write(dataset.describe())

    # Display dataset shape
    st.markdown("<h2 style='text-align: center; font-size: 28px;'>Dataset Shape:</h2>", unsafe_allow_html=True)
    st.write(dataset.shape)

    # Display dataset size
    st.markdown("<h2 style='text-align: center; font-size: 28px;'>Dataset Size:</h2>", unsafe_allow_html=True)
    st.write(dataset.size)

    # Display null values
    st.markdown("<h2 style='text-align: center; font-size: 28px;'>Null Values:</h2>", unsafe_allow_html=True)
    st.write(dataset.isnull().sum())


    # Display column information
    st.markdown("<h2 style='text-align: center; font-size: 28px;'>Column Information:</h2>", unsafe_allow_html=True)
    column_info = get_column_info(dataset)
    column_info_df = pd.DataFrame(column_info).T
    st.write(column_info_df)

    # Add the current outputs to the session state history


def main():
    st.write("About tool : The Data Analysis Tool is a versatile application designed to assist users in performing various data analysis tasks effortlessly. Whether you need to explore data, build predictive models, visualize trends, or preprocess your dataset, this tool has got you covered.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        dataset = pd.read_csv(uploaded_file)

        st.write("There are 6 Functions you can run:\n"
                 "1. Get Detail Analysis report\n"
                 "2. Get Header of Dataset\n"
                 "3. Get Basic Information about Dataset\n"
                 "4. Start Preprocessing\n"
                 "5. To Visualize Dataset\n"
                 "6. To run model")

        st.sidebar.write("Write Function name which you want to perform .....")

        user = st.sidebar.text_input("Write Function name")

        if user:
            st.sidebar.info(f"user: {user}")
            if "hi" in user.lower() or "hello" in user.lower():
                st.sidebar.info("Hello how can I help you ")
            elif "information" in user.lower():
                st.sidebar.info("Sure..")
                information(dataset)

            elif "report" in user.lower():
                st.sidebar.info("Sure..")
                perform_analysis(dataset)

            elif "header" in user.lower():
                st.sidebar.info("Sure..")
                header(dataset)

            elif "model" in user.lower():
                st.sidebar.info("Sure..")
                type = st.text_input("Type of model[Regression/Classification]")
                if type:
                    if type.lower() == "regression":
                        problem_type_1 = "Regression"
                        run_model(st.session_state.X_train, st.session_state.X_test, st.session_state.y_train,
                                  st.session_state.y_test, problem_type_1)
                    elif type.lower() == "classification":
                        problem_type_1 = "Classification"
                        run_model(st.session_state.X_train, st.session_state.X_test, st.session_state.y_train,
                                  st.session_state.y_test, problem_type_1)
                    else:
                        st.write("Invalid model type. Please enter 'Regression' or 'Classification'.")


            elif "visualization" in user.lower() or "visualize" in user.lower() or "visualise" in user.lower() or "graph" in user.lower():
                st.sidebar.info("Sure..")
                st.write("Select the parameters")
                with st.form(key="graph"):
                    graph_type = st.selectbox(
                        "Select Graph Type",
                        ["Line Plot", "Bar Plot", "Histogram", "Pair Plot", "Scatter Plot",
                         "Boxplot", "Violin Plot", "Eventplot", "Hexbin", "Pie Chart", "ECDF", "2D Histogram",
                         "3D Scatter Plot", "3D Surface Plot-1"]
                    )

                    x_axis = st.selectbox("Select X-axis", dataset.columns)
                    y_axis = st.selectbox("Select Y-axis", dataset.columns)
                    z_axis = st.selectbox("Select Z-axis", dataset.columns)

                    submit_graph = st.form_submit_button(label='Graph')

                if submit_graph:
                    st.write("Here is the graph:")
                    draw_graph(graph_type, x_axis, y_axis, dataset, z_axis)




            elif "preprocessing" in user.lower() or "preprocess" in user.lower():
                st.sidebar.info("Sure..")

                st.write("Select the parameters")

                with st.form(key='preprocessing'):

                    Null_value = st.selectbox(

                        "Select method for null value",

                        ["Remove Null Values", "Add Mean Values", "Add Random Values from Column", None]

                    )

                    normalisation = st.selectbox(

                        "Select the method for normalization",

                        ["Standard Scaler", "Min-Max Scaler", "Robust Scaler", None]

                    )

                    test_size_percentage = st.slider("Test Set Percentage", 1, 50, 20, 1,

                                                     key="test_size_percentage")

                    columns_to_delete = st.multiselect("Select columns to delete:", dataset.columns)

                    outlier_handling = st.selectbox("Select Outlier Handling Method",

                                                    ["None", "Z-Score", "IQR"])

                    select_transformation = st.selectbox("Select transformation Method",

                                                         ["Log Transformation", "Square root transformation", None])

                    target_column_3 = st.selectbox(

                        "Select the target Please ignore the columns which are deleted ",

                        dataset.columns, key="target_column_3")

                    n_components = st.slider("Number of Components", 2, min(len(dataset.columns), 20), 2)

                    use_pca = st.checkbox("Enable PCA")

                    submit = st.form_submit_button(label='data_preprocessing')

                if submit:
                    preprocess_data(Null_value, dataset, normalisation, test_size_percentage, columns_to_delete,

                                    target_column_3, outlier_handling, use_pca, n_components, select_transformation)

            else:
                st.sidebar.info("Please write correct Fuction name..")




if __name__ == "__main__":
    main()