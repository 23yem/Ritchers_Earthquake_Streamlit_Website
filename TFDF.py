import tensorflow as tf
import streamlit as st
from streamlit.components import v1 as components
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import joblib 


from data_preprocessing_LGBM import data_preprocessing

st.set_page_config(layout="wide")

# Load part 1
with open('TFDF_1.html', 'r') as file:    
    html_content_1 = file.read()

components.html(html_content_1, width = None, height=1000)


#Add some spacing between elements on the website
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")


st.markdown("""
    <style>
    .element-container {
        margin-bottom: 0px;
    }
    .stSlider, .stSelectbox {
        padding-bottom: 10px;
    }
    .stMarkdown {
        padding-bottom: 1px;
    }
    </style>
    """, unsafe_allow_html=True)


st.markdown("""
    <div style="text-align: center">
        <h1 style="color: #7af5b9;">Using my Best Deployed and Saved  Tensor Flow Decision Forests (TFDF)!</h1>
        <h3 style="color: #7af5b9;">Please select the values that you want, and then press the "predict" button to see if they would survive the titanic</h3>
        <h3 style="color: #7af5b9;">Just a heads up that some of these values are categorial like "q" or "t" and it isn't clear what they mean. So don't worry if you don't understand what certain values mean. Just randomly pick.</h3>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")
st.write("")

# Create a column layout so the text below is less wide
col1, col2, col3 = st.columns([1,6,1])


# User input for each feature
with col2:

    # Geo Levels
    st.markdown('<h4 style="color:white;">Geographic Region Level 1 (0-30)</h4>', unsafe_allow_html=True)
    geo_level_1_id = st.slider('Geo Level 1 ID', 0, 30)
    st.markdown('<hr>', unsafe_allow_html=True)

    st.markdown('<h4 style="color:white;">Geographic Region Level 2 (0-1427)</h4>', unsafe_allow_html=True)
    geo_level_2_id = st.slider('Geo Level 2 ID', 0, 1427)
    st.markdown('<hr>', unsafe_allow_html=True)

    st.markdown('<h4 style="color:white;">Geographic Region Level 3 (0-12567)</h4>', unsafe_allow_html=True)
    geo_level_3_id = st.slider('Geo Level 3 ID', 0, 12567)
    st.markdown('<hr>', unsafe_allow_html=True)

    # Building Characteristics
    st.markdown('<h4 style="color:white;">Number of floors before the earthquake</h4>', unsafe_allow_html=True)
    count_floors_pre_eq = st.slider('Count Floors Pre-EQ', 1, 10)
    st.markdown('<hr>', unsafe_allow_html=True)

    st.markdown('<h4 style="color:white;">Age of the building in years</h4>', unsafe_allow_html=True)
    age = st.slider('Age', 0, 100)
    st.markdown('<hr>', unsafe_allow_html=True)

    st.markdown('<h4 style="color:white;">Normalized area of the building footprint</h4>', unsafe_allow_html=True)
    area_percentage = st.slider('Area Percentage', 1, 100)
    st.markdown('<hr>', unsafe_allow_html=True)

    st.markdown('<h4 style="color:white;">Normalized height of the building footprint</h4>', unsafe_allow_html=True)
    height_percentage = st.slider('Height Percentage', 1, 100)
    st.markdown('<hr>', unsafe_allow_html=True)

    # Categorical Features
    land_surface_condition = st.selectbox('Land Surface Condition', ['n', 'o', 't'])
    foundation_type = st.selectbox('Foundation Type', ['h', 'i', 'r', 'u', 'w'])
    roof_type = st.selectbox('Roof Type', ['n', 'q', 'x'])
    ground_floor_type = st.selectbox('Ground Floor Type', ['f', 'm', 'v', 'x', 'z'])
    other_floor_type = st.selectbox('Other Floor Type', ['j', 'q', 's', 'x'])
    position = st.selectbox('Position', ['j', 'o', 's', 't'])
    plan_configuration = st.selectbox('Plan Configuration', ['a', 'c', 'd', 'f', 'm', 'n', 'o', 'q', 's', 'u'])

    # Binary Features
    st.markdown('<h4 style="color:white;">Superstructure Material Types</h4>', unsafe_allow_html=True)
    has_superstructure_adobe_mud = st.checkbox('Adobe/Mud')
    has_superstructure_mud_mortar_stone = st.checkbox('Mud Mortar - Stone')
    has_superstructure_stone_flag = st.checkbox('Stone')
    has_superstructure_cement_mortar_stone = st.checkbox('Cement Mortar - Stone')
    has_superstructure_mud_mortar_brick = st.checkbox('Mud Mortar - Brick')
    has_superstructure_cement_mortar_brick = st.checkbox('Cement Mortar - Brick')
    has_superstructure_timber = st.checkbox('Timber')
    has_superstructure_bamboo = st.checkbox('Bamboo')
    has_superstructure_rc_non_engineered = st.checkbox('RC Non-Engineered')
    has_superstructure_rc_engineered = st.checkbox('RC Engineered')
    has_superstructure_other = st.checkbox('Other')

    # Family Count
    st.markdown('<h4 style="color:white;">Number of families living in the building</h4>', unsafe_allow_html=True)
    count_families = st.slider('Count Families', 1, 10)
    st.markdown('<hr>', unsafe_allow_html=True)

    # Secondary Use
    st.markdown('<h4 style="color:white;">Secondary Uses of the Building</h4>', unsafe_allow_html=True)
    has_secondary_use = st.checkbox('Has Secondary Use')

    st.write("")
    st.write("")
    st.write("")

    def predict(geo_level_1_id, geo_level_2_id, geo_level_3_id, count_floors_pre_eq, age, area_percentage, height_percentage, 
        land_surface_condition, foundation_type, roof_type, ground_floor_type, other_floor_type, position, plan_configuration, 
        has_superstructure_adobe_mud, has_superstructure_mud_mortar_stone, has_superstructure_stone_flag, 
        has_superstructure_cement_mortar_stone, has_superstructure_mud_mortar_brick, has_superstructure_cement_mortar_brick, 
        has_superstructure_timber, has_superstructure_bamboo, has_superstructure_rc_non_engineered, has_superstructure_rc_engineered, 
        has_superstructure_other, count_families, has_secondary_use):
       
        # Convert inputs to model's expected format
        geo_level_1_id = int(geo_level_1_id)
        geo_level_2_id = int(geo_level_2_id)
        geo_level_3_id = int(geo_level_3_id)
        count_floors_pre_eq = int(count_floors_pre_eq)
        age = int(age)
        area_percentage = int(area_percentage)
        height_percentage = int(height_percentage)
        land_surface_condition = str(land_surface_condition)
        foundation_type = str(foundation_type)
        roof_type = str(roof_type)
        ground_floor_type = str(ground_floor_type)
        other_floor_type = str(other_floor_type)
        position = str(position)
        plan_configuration = str(plan_configuration)
        has_superstructure_adobe_mud = int(has_superstructure_adobe_mud)
        has_superstructure_mud_mortar_stone = int(has_superstructure_mud_mortar_stone)
        has_superstructure_stone_flag = int(has_superstructure_stone_flag)
        has_superstructure_cement_mortar_stone = int(has_superstructure_cement_mortar_stone)
        has_superstructure_mud_mortar_brick = int(has_superstructure_mud_mortar_brick)
        has_superstructure_cement_mortar_brick = int(has_superstructure_cement_mortar_brick)
        has_superstructure_timber = int(has_superstructure_timber)
        has_superstructure_bamboo = int(has_superstructure_bamboo)
        has_superstructure_rc_non_engineered = int(has_superstructure_rc_non_engineered)
        has_superstructure_rc_engineered = int(has_superstructure_rc_engineered)
        has_superstructure_other = int(has_superstructure_other)
        count_families = int(count_families)
        has_secondary_use = int(has_secondary_use)
        

        # Prepare the input data in the correct format
        preprocessed_data = data_preprocessing(geo_level_1_id, geo_level_2_id, geo_level_3_id, count_floors_pre_eq, age, area_percentage, height_percentage, 
                                            land_surface_condition, foundation_type, roof_type, ground_floor_type, other_floor_type, position, plan_configuration, 
                                            has_superstructure_adobe_mud, has_superstructure_mud_mortar_stone, has_superstructure_stone_flag, 
                                            has_superstructure_cement_mortar_stone, has_superstructure_mud_mortar_brick, has_superstructure_cement_mortar_brick, 
                                            has_superstructure_timber, has_superstructure_bamboo, has_superstructure_rc_non_engineered, has_superstructure_rc_engineered, 
                                            has_superstructure_other, count_families, has_secondary_use)


        model = load_model("model1_RF")

        # # Print dataframe of the data
        #st.dataframe(preprocessed_data)


        # Assuming model is your trained TF-DF model and test_data is your test data
        predictions = model.predict(preprocessed_data)

        # Extract the probabilities and convert to a NumPy array (like .predict() normally does for tensorflow neural network with multi-class classification)
        predictions = np.array([prediction['probabilities'] for prediction in predictions])
    

        # Print table that tells the percentage for each class
        st.markdown("<h4>This table shows the percentage certainty for each column. The '0' column <span style='color: green; font-weight: bold;'> is low damage </span>, the '1' column is <span style='color: gray; font-weight: bold;'>medium damage</span>, and the '2' column is <span style='color: red; font-weight: bold;'>high damage</span></h4>.", unsafe_allow_html=True)        
        predictions_percent = np.around(predictions*100, 2)

        predictions_percent = pd.DataFrame(predictions_percent, columns = model.classes_) # Convert to dataframe. model.classes_ is an array of the unique classes for the multi-class classification

        predictions_percent = predictions_percent.astype(str)

        predictions_percent.iloc[0, 0] = f"{predictions_percent.iloc[0, 0]}%"
        predictions_percent.iloc[0, 1] = f"{predictions_percent.iloc[0, 1]}%"
        predictions_percent.iloc[0, 2] = f"{predictions_percent.iloc[0, 2]}%"
        st.write(predictions_percent)


        

        # Get the class with the highest probability
        predictions = predictions.argmax(axis=-1)

        return predictions



    # Make the buttons bigger
    st.markdown("""
        <style>
        .stButton>button {
            font-size: 10px;
            padding: 20px 40px;
        }
        </style>
        """, unsafe_allow_html=True)


    if st.button('Predict'):
        prediction = predict(geo_level_1_id, geo_level_2_id, geo_level_3_id, count_floors_pre_eq, age, area_percentage, height_percentage, 
        land_surface_condition, foundation_type, roof_type, ground_floor_type, other_floor_type, position, plan_configuration, 
        has_superstructure_adobe_mud, has_superstructure_mud_mortar_stone, has_superstructure_stone_flag, 
        has_superstructure_cement_mortar_stone, has_superstructure_mud_mortar_brick, has_superstructure_cement_mortar_brick, 
        has_superstructure_timber, has_superstructure_bamboo, has_superstructure_rc_non_engineered, has_superstructure_rc_engineered, 
        has_superstructure_other, count_families, has_secondary_use)


        if prediction == 0:
            st.markdown(f'## Prediction: Your building experienced <span style="color: green; font-weight: bold;">very low damage!</span> Very lucky! ', unsafe_allow_html=True)
        elif prediction == 1:
            st.markdown(f'## Prediction: Your building experienced <span style="color: gray; font-weight: bold;">a medium amout of damage.</span> I hope you are okay! ', unsafe_allow_html=True)
        else:
            st.markdown(f'## Prediction: Your building experienced <span style="color: red; font-weight: bold;">very HIGH damage!</span> Your building is most likely destroyed!', unsafe_allow_html=True)







# Load part 2
with open('TFDF_2.html', 'r') as file:    
    html_content_2 = file.read()

components.html(html_content_2, width = None, height=6000)