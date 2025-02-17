import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

from classifier import Classifier
from const import sales_speed
from extract_words import extract_words
from sampling import split_data

load_dotenv()

print("Hello Zillow")

# 1. prepare data
print("Load & Split Data")
zillow = pd.read_csv("dataset/2. zillow_cleaned.csv")
(
    X_train,
    X_test,
    y_train,
    y_test,
    desc_train,
    desc_test,
    zpid_train,
    zpid_test,
    df_words,
) = split_data()


# 2. extract words
dir = "dataset/word_counts"
if not os.path.exists(dir):
    print("Extracting Discriminative Words")
    extract_words()
else:
    print(f"The directory '{dir}' already exists. No need to extract words.")


# 3. llm estimation
dir = "result/llm_result.csv"
if os.path.exists(dir):
    print(f"{dir} already exists. Skipping LLM estimation.")
else:
    print("Executing LLM")
    from chain import (
        basic_grader,
        build_full_template,
        build_words_template,
        full_llm,
        words_llm,
        zillow_to_str,
    )

    results = {}

    for idx, row in tqdm(
        X_test.iterrows(), desc="Generating Responses", total=len(X_test)
    ):

        print(
            f"\nProperty ID: {zpid_test["zpid"][idx]}, {y_test["duration"][idx]} Days"
        )

        # 3-1. basic
        basic_result = basic_grader.invoke(
            input={
                "description": desc_test["description"][idx],
                "attributes": zillow_to_str(row),
            }
        )
        print(f"Basic: {basic_result}")

        # 3-2. with words
        words_template = build_words_template(city=row["city"], single=row["single"])
        words_grader = words_template | words_llm
        words_result = words_grader.invoke(
            input={
                "description": desc_test["description"][idx],
                "attributes": zillow_to_str(row),
            }
        )
        print(f"Words: {words_result}")

        # 3-3. with words & meanings
        full_template = build_full_template(city=row["city"], single=row["single"])
        full_grader = full_template | full_llm
        full_result = full_grader.invoke(
            input={
                "description": desc_test["description"][idx],
                "attributes": zillow_to_str(row),
            }
        )
        print(f"Full: {full_result}")

        results[zpid_test["zpid"][idx]] = {
            "GT": y_test["duration"][idx],
            "GT_bool": (
                "yes"
                if y_test["duration"][idx] <= sales_speed[row["city"]][row["single"]]
                else "no"
            ),
            "basic_tom": basic_result.TOM,
            "words_tom": words_result.TOM,
            "full_tom": full_result.TOM,
            "basic_reason": basic_result.REASON,
            "words_reason": words_result.REASON,
            "full_reason": full_result.REASON,
        }

        print("")

    results_df = pd.DataFrame.from_dict(results, orient="index")
    output_file = "result/llm_result.csv"
    results_df.to_csv(output_file, index_label="zpid")
    print(f"LLM Results saved to {output_file}")


# 3-4. evaluate
def evaluate_llm(llm_result, prediction_column):
    y_true = llm_result["GT_bool"]
    y_pred = llm_result[prediction_column]

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy for {prediction_column}: {accuracy:.4f}")

    report = classification_report(y_true, y_pred)
    print(f"Classification Report for {prediction_column}:\n{report}")

    return accuracy, report


llm_result = pd.read_csv("result/llm_result.csv")

results_summary = {}

for case in ["basic_tom", "words_tom", "full_tom"]:
    accuracy, report = evaluate_llm(llm_result, case)
    results_summary[case] = {
        "accuracy": accuracy,
        "report": report,
        "feature_importance": None,
        "best_params": None,
    }


# 4. classical ml prediction
print("Execute ML Prediction")


# 4-1. simple pre-process
def create_binary_target(X, y, sales_speed):
    binary_target = []

    for idx, row in X.iterrows():
        city = row["city"]
        single = row["single"]
        threshold = sales_speed[city][single]

        if y.loc[idx] <= threshold:
            binary_target.append("yes")
        else:
            binary_target.append("no")

    return pd.Series(binary_target)


def preprocess_data(X, y):
    y = y["duration"]
    y = create_binary_target(X, y, sales_speed)
    y = y.map({"no": 0, "yes": 1})
    X = X.drop(columns=["address"])
    X = pd.get_dummies(X, columns=["city", "submarket"])
    X.columns = X.columns.str.replace(" ", "_")
    X.reset_index(inplace=True, drop=True)

    return X, y


X_train, y_train = preprocess_data(X_train, y_train)
X_test, y_test = preprocess_data(X_test, y_test)


for model_type in [
    "logistic",
    "logistic_balance",
    "rf",
    "rf_balance",
    "xgb",
    "xgb_balance",
    # "lightgbm", "lightgbm_balance",
]:

    # 4-2. model select & fit
    classifier = Classifier(model_type=model_type)
    classifier.fit(X_train, y_train)

    # 4-3. predict & evaluate
    accuracy, report = classifier.evaluate(X_test, y_test)
    print(f"Accuracy for {model_type}: {accuracy:.4f}")
    print(f"Classification Report:\n{report}")

    # 4-4. feature importance
    if model_type == "logistic":
        importance = np.abs(classifier.model.coef_[0])
    elif model_type == "rf":
        importance = classifier.model.feature_importances_
    elif model_type == "xgb":
        importance = classifier.model.feature_importances_

    feature_importance = (
        pd.Series(importance, index=X_train.columns)
        .sort_values(ascending=False)
        .head(5)
    )

    results_summary[model_type] = {
        "accuracy": accuracy,
        "report": report,
        "feature_importance": feature_importance,
        "best_params": classifier.best_params,
    }


# 5. final output
output_dir = "result/eval_results.txt"
with open(output_dir, "w") as f:
    for model_type, result in results_summary.items():
        f.write(f"Model: {model_type}\n")
        f.write(f"Accuracy: {result['accuracy']:.4f}\n")
        f.write(f"Report:\n{result['report']}\n")

        # for ml predictions
        if result["feature_importance"] is not None:
            f.write("Feature Importance:\n")
            for feature, importance in result["feature_importance"].items():
                f.write(f"{feature} {importance:.4f}\n")
            f.write("\n")
        if result["best_params"] is not None:
            f.write("Best Parameters:\n")
            for param, value in result["best_params"].items():
                f.write(f"{param} {value}\n")
            f.write("\n")

        f.write("\n")

print(f"Results summary saved to {output_dir}")
