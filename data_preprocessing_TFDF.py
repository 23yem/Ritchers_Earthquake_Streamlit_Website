import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import tensorflow as tf
import numpy as np
import os
import joblib
import tensorflow_decision_forests as tfdf



# Set Global random seed to make sure we can replicate any model that we create (no randomness)
np.random.seed(42)
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

os.environ['TF_DETERMINISTIC_OPS'] = '1'


def data_preprocessing(geo_level_1_id, geo_level_2_id, geo_level_3_id, count_floors_pre_eq, age, area_percentage, height_percentage, 
    land_surface_condition, foundation_type, roof_type, ground_floor_type, other_floor_type, position, plan_configuration, 
    has_superstructure_adobe_mud, has_superstructure_mud_mortar_stone, has_superstructure_stone_flag, 
    has_superstructure_cement_mortar_stone, has_superstructure_mud_mortar_brick, has_superstructure_cement_mortar_brick, 
    has_superstructure_timber, has_superstructure_bamboo, has_superstructure_rc_non_engineered, has_superstructure_rc_engineered, 
    has_superstructure_other, count_families, has_secondary_use):
    # Create a dictionary where keys are column names and values are data

    # Create an initial dataframe with all the values in the categorial columns that we want, so that when we do pd.get_dummies(), we have all the possible one-hot enconded values. 
    # This is important since I don't ask the users to put in values for all columns, so if I only use their data and do pd.get_dummies(), the one-hot encoded data won't have all possible columns
    # Only the first row is important
    data = {
        'geo_level_1_id': [geo_level_1_id, 0, 0, 0],
        'geo_level_2_id': [geo_level_2_id, 0, 0,0],
        'geo_level_3_id': [geo_level_3_id, 0, 0,0],
        'count_floors_pre_eq': [count_floors_pre_eq, 0, 0,0],
        'age': [age, 0, 0,0],
        'area_percentage': [area_percentage, 0, 0,0],
        'height_percentage': [height_percentage, 0, 0,0],
        'land_surface_condition': [land_surface_condition, 'n', 'o', 'o'],
        'foundation_type': [foundation_type, 'h','r','u'],
        'roof_type': [roof_type, 'n', 'q', 'x'],
        'ground_floor_type': [ground_floor_type, 'f', 'v', 'x'],
        'other_floor_type': [other_floor_type, 'q', 'q', 'q'],
        'position': [position, 's', 's', 's'],
        'plan_configuration': [plan_configuration, 'u','u','u'],
        'has_superstructure_adobe_mud': [has_superstructure_adobe_mud, 0, 0, 0],
        'has_superstructure_mud_mortar_stone': [has_superstructure_mud_mortar_stone, 0, 0, 0],
        'has_superstructure_stone_flag': [has_superstructure_stone_flag, 0, 0, 0],
        'has_superstructure_cement_mortar_stone': [has_superstructure_cement_mortar_stone, 0, 0, 0],
        'has_superstructure_mud_mortar_brick': [has_superstructure_mud_mortar_brick, 0, 0, 0],
        'has_superstructure_cement_mortar_brick': [has_superstructure_cement_mortar_brick, 0, 0, 0],
        'has_superstructure_timber': [has_superstructure_timber, 0, 0, 0],
        'has_superstructure_bamboo': [has_superstructure_bamboo, 0, 0, 0],
        'has_superstructure_rc_non_engineered': [has_superstructure_rc_non_engineered, 0, 0, 0],
        'has_superstructure_rc_engineered': [has_superstructure_rc_engineered, 0, 0, 0],
        'has_superstructure_other': [has_superstructure_other, 0, 0, 0],
        'count_families': [count_families, 0, 0, 0],
        'has_secondary_use': [has_secondary_use, 0, 0, 0]
    }

    df = pd.DataFrame(data)

    df = pd.get_dummies(df) # one hot encoding

    df = df.head(1) # This will keep only the first row of the Dataframe and discard the rest. The rest of the rows was just to ensure we have all the important one-hot encoded columns

    features = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id', 'count_floors_pre_eq', 'age', 'area_percentage', 'height_percentage',
    'has_superstructure_adobe_mud', 'has_superstructure_mud_mortar_stone', 'has_superstructure_stone_flag', 'has_superstructure_cement_mortar_stone',
    'has_superstructure_mud_mortar_brick', 'has_superstructure_cement_mortar_brick', 'has_superstructure_timber', 'has_superstructure_bamboo', 
    'has_superstructure_rc_non_engineered', 'has_superstructure_rc_engineered', 'has_superstructure_other', 'count_families', 'has_secondary_use', 
    'land_surface_condition_n', 'land_surface_condition_o', 'foundation_type_h', 'foundation_type_r', 'foundation_type_u', 'roof_type_n', 
    'roof_type_q', 'roof_type_x', 'ground_floor_type_f', 'ground_floor_type_v', 'ground_floor_type_x', 'other_floor_type_q', 'position_s',
    'plan_configuration_u']

    df = df[features]

    user_input = df.astype('float32')

    # # Set the display options so that I see all the columns
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.expand_frame_repr', False)
    # pd.set_option('max_colwidth', None)
    # print(user_input)


    #user_input.to_csv('output.csv', index=False)


    user_input = tfdf.keras.pd_dataframe_to_tf_dataset(user_input , task = tfdf.keras.Task.CLASSIFICATION) # convert it for tensorflow decision forest model. This is the format tfdf models accept data

    return user_input





# data_preprocessing(
#     geo_level_1_id=6, 
#     geo_level_2_id=487, 
#     geo_level_3_id=12198, 
#     count_floors_pre_eq=2, 
#     age=30, 
#     area_percentage=6, 
#     height_percentage=5, 
#     land_surface_condition='t', 
#     foundation_type='r', 
#     roof_type='n', 
#     ground_floor_type='f', 
#     other_floor_type='x', 
#     position='t', 
#     plan_configuration='d', 
#     has_superstructure_adobe_mud=1, 
#     has_superstructure_mud_mortar_stone=1, 
#     has_superstructure_stone_flag=0, 
#     has_superstructure_cement_mortar_stone=0, 
#     has_superstructure_mud_mortar_brick=0, 
#     has_superstructure_cement_mortar_brick=0, 
#     has_superstructure_timber=0, 
#     has_superstructure_bamboo=0, 
#     has_superstructure_rc_non_engineered=0, 
#     has_superstructure_rc_engineered=0, 
#     has_superstructure_other=0, 
#     count_families=1, 
#     has_secondary_use=0
# )

