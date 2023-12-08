import pandas as pd
import numpy as np
from typing import NamedTuple, Union
import torch
from sklearn.model_selection import train_test_split


class Batch(NamedTuple):
    cgm: Union[np.ndarray, torch.Tensor]
    timestamps: Union[np.ndarray, torch.Tensor]
    meals: Union[np.ndarray, torch.Tensor] 
    demographics: Union[np.ndarray, torch.Tensor] 
    diagnosis: Union[np.ndarray, torch.Tensor] 

class PatientInfo(NamedTuple):
    patient_ids: np.ndarray
    train_ids: np.ndarray
    test_ids: np.ndarray

SPLIT_SEED = 21
NUM_MEALS_THRESHOLD = 5
#ARC_LENGTH_THRESHOLD = 16.
ARC_LENGTH_THRESHOLD = 0.
MEAL_COVARIATES = [
    'total_grams', 'total_carb', 'total_sugar',
    # leave these out because we don't want to assume we have access to this info (harder to get in the wild)
    #'glycemic_index_glucose', 'glycemic_load_glucose', 'glycemic_index_bread', 'glycemic_load_bread',
    'total_dietary_fiber', 'total_fat', 'total_protein', 
]
DEMOGRAPHICS_COVARIATES = ["gender", "age", "weight"]

def load_data(seed, domain_adaptation=False):
    cgms_df = pd.read_csv("per-meal-data/cgms.csv", parse_dates=["timestamp"])
    meal_series_df = pd.read_csv("per-meal-data/meal_contexts.csv", parse_dates=["timestamp"])
    meals_df = pd.read_csv("per-meal-data/meals.csv", parse_dates=["timestamp"])
    demo_df = pd.read_csv("per-meal-data/demographics.csv")

    # Process categoricals in demographics
    demo_df = demo_df.assign(gender=(demo_df["gender"] == "F").astype(float))
    # Fill in missing demographics data with nans
    missing_patient_ids = set(cgms_df['patient_id']) - set(demo_df['patient_id'])
    missing_demo = pd.DataFrame({'patient_id': list(missing_patient_ids)})
    for col in demo_df.columns:
        if col != 'patient_id':
            missing_demo.loc[:, col] = np.nan
    demo_df = pd.concat([demo_df, missing_demo], ignore_index=True)
    demo_df = demo_df.assign(diagnosis=demo_df["diagnosis"].replace({"Pre-D": 0, "T2D": 1}))

    # Remove patients with missing diagnosis data
    not_nan_patient = demo_df["diagnosis"].notna() 
    demo_df = demo_df[not_nan_patient]
    cgms_df = cgms_df[cgms_df["patient_id"].isin(demo_df["patient_id"])]
    meal_series_df = meal_series_df[meal_series_df["patient_id"].isin(demo_df["patient_id"])]
    meals_df = meals_df[meals_df["patient_id"].isin(demo_df["patient_id"])]

    # Split into train and test sets with stratified sampling
    diagnosis = demo_df["diagnosis"].values
    patient_id = demo_df["patient_id"].values

    if domain_adaptation:
        train_ids, test_ids, train_diag, test_diag = train_test_split(patient_id, diagnosis, test_size=0.5, random_state=seed, stratify=diagnosis)
        print(f"Train: Pre-D: {(train_diag == 0.).sum()} ({(train_diag == 0.).mean():.2f}) | T2D: {(train_diag == 1.).sum()} ({(train_diag == 1.).mean():.2f}) | Total: {len(train_diag)}")
        print(f"Test: Pre-D: {(test_diag == 0.).sum()} ({(test_diag == 0.).mean():.2f}), T2D: {(test_diag == 1.).sum()} ({(test_diag == 1.).mean():.2f}) | Total: {len(test_diag)}")
        print(f"Train set patient_ids: {train_ids}")
        print(f"Test set patient_ids: {test_ids}")
    else:
        train_ids, test_ids = patient_id, patient_id  # share patients between train and test

    NAN_THRESHOLD = 0.
    print(f"Keeping responses with at most {NAN_THRESHOLD * 100}% NaNs")
    meal_id_kept = meals_df.query("cgm_nan_frac <= @NAN_THRESHOLD")["meal_id"]
    print(f"Kept {len(meal_id_kept)} meals out of {len(meals_df)}")
    cgms_df_kept = cgms_df[cgms_df["meal_id"].isin(meal_id_kept)]
    print(f"Kept {len(cgms_df_kept)} CGM readings out of {len(cgms_df)}")
    meal_series_df_kept = meal_series_df[meal_series_df["meal_id"].isin(meal_id_kept)]
    print(f"Kept {len(meal_series_df_kept)} meal contexts out of {len(meal_series_df)}")
    meals_df_kept = meals_df[meals_df["meal_id"].isin(meal_id_kept)]
    print(f"Kept {len(meals_df_kept)} meals out of {len(meals_df)}")

    meal_covariates = MEAL_COVARIATES
    print(f"Keeping meal covariates: {meal_covariates}")
    #meal_covariates = [
    #    'total_grams', 'glucose', 'total_carb',
    #    'available_carb', 'starch', 'total_sugar', 'added_sugar_avail_carb',
    #    'added_sugar_total_sugar', 'glycemic_index_glucose',
    #    'glycemic_load_glucose', 'glycemic_index_bread', 'glycemic_load_bread',
    #    'fructose', 'galactose', 'lactose', 'maltose', 'sucrose',
    #    'total_dietary_fiber', 'insoluble_dietary_fiber',
    #    'soluble_dietary_fiber', 'total_fat', 'total_protein', 'animal_protein',
    #    'vegetable_protein', 'energy', 'total_pufa', 'total_mufa', 'omega3',
    #    'total_grains', 'whole_grains', 'refined_grains', 'alcohol'
    #]
    meals_df_kept = meals_df_kept.loc[:, ["patient_id", "timestamp", "meal_id"] + meal_covariates]

    print(f"Keeping demographics covariates: {DEMOGRAPHICS_COVARIATES}")
    # leave out diagnosis, hba1c, ALT, insulin since they will be correlated with type 2 diagnosis, also lab measurements aren't readily available at home
    # leave out SBP, DBP, LDL, HDL since they aren't that important for reconstructing glucose; and can have spurious correlation with type2

    demo_df = demo_df.loc[:, ["patient_id", "diagnosis"] + DEMOGRAPHICS_COVARIATES].set_index("patient_id")

    """ Collect data into arrays

    For each meal_id, we get the cgm, meal_series, and the meal.

    For each meal in the meal context, we keep only a subset of the covariates

    For the meal context, we need to place it on the same grid as the CGM data. Then we must split the meal context in half. The second part includes the meal.

    We also split the CGM into two parts, before and after the meal. 
    """
    #if NAN_THRESHOLD != 0.:
    #    raise NotImplementedError("Need to handle NaNs in CGM data")
    #assert cgms_df_kept.isna().sum().sum() == 0, "There are NaNs in the CGM data!"
    assert meal_series_df_kept.isna().sum().sum() == 0, "There are NaNs in the meal covariate data!"

    cgm_arrays = []
    timestamp_arrays = []
    meal_series_arrays = []
    demographics_arrays = []
    all_diagnosis = []
    all_patient_ids = []
    for ((meal_id, cgm), (_, meal_series), (_, meal)) in zip(cgms_df_kept.groupby("meal_id"), meal_series_df_kept.groupby("meal_id"), meals_df_kept.groupby("meal_id")):
        #if (meal["total_carb"].values  < 20) and (meal["total_sugar"].values < 10):
        #    continue
        patient_id = meal["patient_id"].iloc[0]
        cgm_array = cgm["glucose"].values[..., None]
        timestamp_array = (cgm["timestamp"].dt.hour * 60. + cgm["timestamp"].dt.minute).values[..., None]  # TODO: make this cyclic?

        arc_length = np.sqrt(np.sum(np.square(np.diff(cgm_array[:, 0], axis=-1)), axis=-1))
        if arc_length < ARC_LENGTH_THRESHOLD:  # ignore flat curves; median arc_length is about 16
            continue

        start_time = cgm["timestamp"].iloc[0]
        time_index = (meal_series["timestamp"] - start_time) // pd.Timedelta(minutes=5)
        meal_series_array = np.zeros((len(cgm), len(meal_covariates)))
        #meal_series_array = np.full((len(cgm), len(meal_covariates)), np.nan)
        meal_series_array[time_index, :] = meal_series[meal_covariates].values

        diagnosis = demo_df.loc[patient_id, "diagnosis"]
        demographics_array = demo_df.loc[patient_id, DEMOGRAPHICS_COVARIATES].values

        cgm_arrays.append(cgm_array)
        timestamp_arrays.append(timestamp_array)
        meal_series_arrays.append(meal_series_array)
        demographics_arrays.append(demographics_array)
        all_diagnosis.append(diagnosis)
        all_patient_ids.append(patient_id)

    all_cgm_array = np.stack(cgm_arrays)
    all_timestamp_array = np.stack(timestamp_arrays)
    all_meal_series_array = np.stack(meal_series_arrays)
    all_demographics_array = np.stack(demographics_arrays)
    all_diagnosis = np.array(all_diagnosis)
    all_patient_ids = np.array(all_patient_ids)

    # group meals by patient, drop patients with too few meals 
    patient_groups = {}
    for i, patient_id in enumerate(all_patient_ids):
        if patient_id not in patient_groups:
            patient_groups[patient_id] = []
        patient_groups[patient_id].append(i)
    print(f"Number of patients: {len(patient_groups)}")
    kept_patient_ids = {patient_id: indices for patient_id, indices in patient_groups.items() if len(indices) >= NUM_MEALS_THRESHOLD}
    print(f"Number of patients with at least {NUM_MEALS_THRESHOLD} meals: {len(kept_patient_ids)}")
    all_kept_indices = np.concatenate(list(kept_patient_ids.values()))
    
    all_patient_ids = all_patient_ids[all_kept_indices]
    all_cgm_array = all_cgm_array[all_kept_indices]
    all_timestamp_array = all_timestamp_array[all_kept_indices]
    all_meal_series_array = all_meal_series_array[all_kept_indices]
    all_demographics_array = all_demographics_array[all_kept_indices]
    all_diagnosis = all_diagnosis[all_kept_indices]

    return Batch(cgm=all_cgm_array, timestamps=all_timestamp_array, meals=all_meal_series_array, demographics=all_demographics_array, diagnosis=all_diagnosis), PatientInfo(patient_ids=all_patient_ids, train_ids=train_ids, test_ids=test_ids)

def normalize_train_test(train_arrays, test_arrays):
    new_train_arrays = []
    new_test_arrays = []
    train_means = []
    train_stds = []
    for train_array, test_array in zip(train_arrays, test_arrays):
        axes = tuple(range(train_array.ndim - 1))  # don't reduce across the feature/channel dimension 
        train_mean = np.nanmean(train_array, axis=axes)
        train_std = np.nanstd(train_array, axis=axes)
        train_array = (train_array - train_mean) / train_std
        test_array = (test_array - train_mean) / train_std
        new_train_arrays.append(train_array)
        new_test_arrays.append(test_array)
        train_means.append(train_mean)
        train_stds.append(train_std)
    return tuple(new_train_arrays), tuple(new_test_arrays), (train_means, train_stds)

def preprocess_train_test(seed, domain_adaptation=False):
    all_arrays, patient_info = load_data(seed=seed, domain_adaptation=domain_adaptation)
    train_idx, train_patient_ids = zip(*[(i, patient_id) for i, patient_id in enumerate(patient_info.patient_ids) if patient_id in set(patient_info.train_ids)])
    train_idx, train_patient_ids = np.array(train_idx), np.array(train_patient_ids)
    test_idx, test_patient_ids = zip(*[(i, patient_id) for i, patient_id in enumerate(patient_info.patient_ids) if patient_id in set(patient_info.test_ids)])
    test_idx, test_patient_ids = np.array(test_idx), np.array(test_patient_ids)
    if domain_adaptation:
        assert set(train_patient_ids) & set(test_patient_ids) == set()
    else:
        assert set(train_patient_ids) == set(test_patient_ids)
    train_arrays = Batch(*tuple(arr[train_idx] for arr in all_arrays))
    test_arrays = Batch(*tuple(arr[test_idx] for arr in all_arrays))
    train_x, test_x, (train_mean, train_std) = normalize_train_test(
        (train_arrays.cgm, train_arrays.timestamps, train_arrays.meals, train_arrays.demographics),
        (test_arrays.cgm, test_arrays.timestamps, test_arrays.meals, test_arrays.demographics),
    )
    train_y, test_y = train_arrays.diagnosis, test_arrays.diagnosis
    train_arrays = Batch(*(train_x + (train_y,)))
    test_arrays = Batch(*(test_x + (test_y,)))
    print(f"Loading data with seed {seed}")
    print(f"Number of train examples: {train_arrays[0].shape[0]}")
    print(f"Numer of test examples: {test_arrays[0].shape[0]}")
    return train_arrays, test_arrays, (train_patient_ids, test_patient_ids), (train_mean, train_std)