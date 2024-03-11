import pandas as pd
import numpy as np
import os
from datetime import datetime
import sys
import pickle
from collections import defaultdict


def __datetime(date_str):
    return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')


def prompt1(texts, dates):
    count = 1
    patient_string = ""
    date_prev = None
    # dict_texts = defaultdict(int)
    hospital_visits = len(texts)

    for i, t in enumerate(texts):
        date_now = dates[i]

        if count == 1:
            patient_string += " On visit " + str(count) + ", they had the following diagnoses: " + ", ".join(
                j for j in t) + "."

        else:
            days = (__datetime(date_now) - __datetime(date_prev))
            patient_string += " On visit " + str(count) + ", which was " + str(days).split()[
                0] + " days after the previous visit, the patient had the following diagnoses: " + ", ".join(
                j for j in t) + "."
        count += 1
        date_prev = date_now

    if count == 2:
        text = "The patient had " + str(
            count - 1) + " total visit to the hospital." + patient_string
    else:
        text = "The patient had " + str(
            count - 1) + " total visits to the hospital." + patient_string

    return text.strip()


def prompt2(texts):
    dict_frequencies = defaultdict(int)
    hospital_visits = len(texts)
    for i, t in enumerate(texts):
        for j in t:
            dict_frequencies[j]+=1

    test_start = "Here are the patient's past hospital visit diagnoses, along with their frequencies of occurrence across the " +str(hospital_visits)+ " hospital visits they have had so far:\n\n"

    test_dict = ""

    # sorted = sorted(dict_frequencies, key=lambda x:dict_frequencies.values())
    sorted_dict = {k: v for k, v in sorted(dict_frequencies.items(), key=lambda item: item[1], reverse=True)}


    for k,v in sorted_dict.items():
        test_dict = test_dict + k+": "+str(v)+"\n"

    full_text = test_start + test_dict
    return full_text.strip()


def create_dataset(data_path = "", code= "diabetes", data = "test", prompt = "prompt2", output = "MIMIC_finetune_"):
    print(data)
    texts = generate_texts(data_path, code, data, output)
    df = pd.DataFrame(columns=["PatientId", "Text", "Label", "Text_label"])

    for d in texts:
        tex = d[2]
        date = d[3]
        if prompt == "prompt1":
            processed_text = prompt1(tex, date)
        elif prompt == "prompt2":
            processed_text = prompt2(tex)
        else:
            print("Please specify a prompt!!!!!!!!!!")
        id = d[0]
        label = d[1]
        if label == 0:
            t_label = "Low"
        else:
            t_label = "High"
        df.loc[len(df)]= {"PatientId": id , "Text":processed_text, "Label": label, "Text_label":t_label}
    df.to_csv( os.path.join(".", "data",output+data+"_"+code+"_"+prompt+".csv"), index=False)
    return df


def generate_texts(data_path = "", code= "diabetes", data = "test", output = ""):
    mimic_code = pd.read_csv(os.path.join(data_path, "merge_MIMIC_all_" + code + ".csv"), low_memory=False)

    with open(os.path.join(data_path, output + code + ".bencs."+data), 'rb') as fp:
        data_mimic = pickle.load(fp)

    req_data_mimic = [i[0] for i in data_mimic]

    mimic_code_req = mimic_code[mimic_code["subject_id"].isin(req_data_mimic)]
    groupby_mimic = mimic_code_req.groupby(["subject_id"])

    patient_texts = []

    for i, row in groupby_mimic:
        row.sort_values(by=["admittime"], inplace=True)
        label = row["label"].values[0]
        group_by_hid = row.groupby(["admittime"])

        pt_text = []
        pt_dates = []
        for j, row_had in group_by_hid:
            long_title = np.unique(row_had["long_title"].values)
            long_title_list = [i.replace(", unspecified","") for i in long_title if i.isnumeric() == False]

            date_now = row_had['admittime'].values[0]

            pt_text.append(long_title_list)
            pt_dates.append(date_now)

        list_patient = [i[0], label, pt_text, pt_dates ]
        patient_texts.append(list_patient)

    return patient_texts


create_dataset(data_path = "/data/", code= "diabetes", data = "test", prompt="prompt2", output = "MIMIC_new_task_finetune_")
create_dataset(data_path = "/data/", code= "diabetes", data = "train", prompt = "prompt2", output = "MIMIC_new_task_finetune_")
create_dataset(data_path = "/data/", code= "diabetes", data = "valid", prompt="prompt2", output ="MIMIC_new_task_finetune_")

create_dataset(data_path = "/data/", code= "sud", data = "test", prompt="prompt2", output = "MIMIC_new_task_finetune_")
create_dataset(data_path = "/data/", code= "sud", data = "train", prompt = "prompt2", output = "MIMIC_new_task_finetune_")
create_dataset(data_path = "/data/", code= "sud", data = "valid", prompt="prompt2", output ="MIMIC_new_task_finetune_")

create_dataset(data_path = "/data/", code= "oud", data = "test", prompt="prompt2", output = "MIMIC_new_task_finetune_")
create_dataset(data_path = "/data/", code= "oud", data = "train", prompt = "prompt2", output = "MIMIC_new_task_finetune_")
create_dataset(data_path = "/data/", code= "oud", data = "valid", prompt="prompt2", output = "MIMIC_new_task_finetune_")

