from typing import Annotated

import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from const import sales_speed
from meanings import meanings_to_str

load_dotenv()


def zillow_to_str(row: pd.Series) -> str:

    city_mapping = {
        "CH": "Chicago (IL)",
        "NY": "New York (NY)",
        "LA": "Los Angeles (CA)",
    }

    city = city_mapping.get(row["city"], "Unknown City")
    type = "Single House" if row["single"] == 0 else "Condo/Townhouse"
    address = row["address"]
    parking = row["parking"]
    bathroom = row["bathroom"]
    bedroom = row["bedroom"]
    age = row["age"]
    living = row["living"]

    return (
        f"City: {city}, "
        f"Type: {type}, "
        f"Address: {address}, "
        f"Number of Parking Spaces: {parking}, "
        f"Number of Bathrooms: {bathroom}, "
        f"Number of Bedrooms: {bedroom}, "
        f"Age: {age} Years, "
        f"Living Space: {living} Square Meters"
    )


def words_to_str(
    city: str,
    single: int,
    path: str = "dataset/word_counts",
    percentage: int = 0.25,
    n_words: int = 10,
) -> str:

    city_mapping = {
        "CH": "Chicago (IL)",
        "NY": "New York (NY)",
        "LA": "Los Angeles (CA)",
    }
    single_mapping = {0: "Single House", 1: "Condo/Townhouse"}
    level_mapping = {0: "Fast-Selling", 2: "Slow-Selling"}

    statement = ""

    for level in [0, 2]:
        df = pd.read_csv(
            f"{path}/{percentage}/{city}_{single}_group_{level}_zscore.csv", header=None
        )
        words = df.iloc[:, 0].head(n_words).tolist()
        res = f"{level_mapping[level]} {single_mapping[single]} in {city_mapping[city]}: {words}\n"
        statement += res

    return statement


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# class BinaryTOM(BaseModel):
#     """Estimated time-on-market(TOM) until the property is sold."""

#     TOM: Annotated[
#         str,
#         Field(description="Determine if the property was sold fast: 'yes' or 'no'."),
#     ]
#     REASON: Annotated[
#         str, Field(description="Provide a brief reason for the decision.")
#     ]

class MultiTOM(BaseModel):
    """Estimated time-on-market(TOM) until the property is sold."""
    
    TOM: Annotated[
        str,
        Field(description="Determine if the property was sold 'fast', 'moderate' or 'slow'."),
    ]
    REASON: Annotated[
        str, Field(description="Provide a brief reason for the decision.")
    ]


# basic

def build_basic_system(sales_speed: dict, th_idx: int, city: str, type: int):
    
    basic_system = f"""
    You are an expert realtor.
    The user will provide following information about properties that already have been sold on Zillow.
    - Description: text written by the realtor
    - Attributes: basic information such as the number of rooms, size, etc.
    Your task is to determine whether the property was sold fast, moderate, or slow.
    - fast-selling: 0 to {sales_speed[city][type]["fast"][th_idx]} days
    - slow-selling: {sales_speed[city][type]["slow"][th_idx]} to 365 days
    - moderate-selling: {sales_speed[city][type]["fast"][th_idx]} to {sales_speed[city][type]["slow"][th_idx]} days
    Respond with a class name: 'fast', 'moderate', or 'slow', and provide a brief reason (around 200 characters) for your decision.
    """

    return basic_system


def build_basic_template(basic_system: str):

    basic_template = ChatPromptTemplate.from_messages(
        [
            ("system", basic_system),
            ("human", """[Description]\n{description}\n\n[Attributes]\n{attributes}\n\n"""),
        ]
    )

    return basic_template

# basic_llm = llm.with_structured_output(BinaryTOM)
basic_llm = llm.with_structured_output(MultiTOM)



# with words

def build_words_system(sales_speed: dict, th_idx: int, city: str, type: int):
    
    words_system = f"""
    You are an expert realtor.
    The user will provide following information about properties that already have been sold on Zillow.
    - Description: text written by the realtor
    - Attributes: basic information such as the number of rooms, size, etc.
    - Discriminative Words: words commonly used in homes that sell quickly or late
    Your task is to determine whether the property was sold fast, moderate, or slow.
    - fast-selling: 0 to {sales_speed[city][type]["fast"][th_idx]} days
    - slow-selling: {sales_speed[city][type]["slow"][th_idx]} to 365 days
    - moderate-selling: {sales_speed[city][type]["fast"][th_idx]} to {sales_speed[city][type]["slow"][th_idx]} days
    Respond with a class name: 'fast', 'moderate', or 'slow', and provide a brief reason (around 200 characters) for your decision.
    """

    return words_system


def build_words_template(city: str, single: int, words_system: str, th_val: float):

    discriminative_words = words_to_str(city=city, single=single, percentage=th_val)
    words_template = ChatPromptTemplate.from_messages(
        [
            ("system", words_system),
            (
                "human",
                "[Description]\n{description}\n\n"
                "[Attributes]\n{attributes}\n\n"
                "[Discriminative Words]\n{words}",
            ),
        ]
    ).partial(words=discriminative_words)

    return words_template


words_llm = llm.with_structured_output(MultiTOM)


# with words & meanings

def build_full_system(sales_speed: dict, th_idx: int, city: str, type: int):
    
    words_system = f"""
    You are an expert realtor.
    The user will provide following information about properties that already have been sold on Zillow.
    - Description: text written by the realtor
    - Attributes: basic information such as the number of rooms, size, etc.
    - Discriminative Words: words commonly used in homes that sell quickly or late
    - Meanings: implications of these words for potential buyers in the context of real estate transactions
    Your task is to determine whether the property was sold fast, moderate, or slow.
    - fast-selling: 0 to {sales_speed[city][type]["fast"][th_idx]} days
    - slow-selling: {sales_speed[city][type]["slow"][th_idx]} to 365 days
    - moderate-selling: {sales_speed[city][type]["fast"][th_idx]} to {sales_speed[city][type]["slow"][th_idx]} days
    Respond with a class name: 'fast', 'moderate', or 'slow', and provide a brief reason (around 200 characters) for your decision.
    """

    return words_system


def build_full_template(city: str, single: int, full_system: str, th_val: float, th_idx: int):

    discriminative_words = words_to_str(city=city, single=single, percentage=th_val)
    meanings = meanings_to_str(city, single, th_idx)

    full_template = ChatPromptTemplate.from_messages(
        [
            ("system", full_system),
            (
                "human",
                "[Description]\n{description}\n\n"
                "[Attributes]\n{attributes}\n\n"
                "[Discriminative Words]\n{words}\n"
                "[Meanings]\n{meanings}",
            ),
        ]
    ).partial(words=discriminative_words, meanings=meanings)

    return full_template


full_llm = llm.with_structured_output(MultiTOM)


if __name__ == "__main__":

    zillow = pd.read_csv("dataset/2. zillow_cleaned.csv")
    sample = zillow.iloc[0]
    th_idx = 4 # 0(5%) to 5(30%)
    th_val = 0.25
    gt_class = (
        "fast"
        if sample["duration"] <= sales_speed[sample["city"]][sample["single"]]["fast"][th_idx]
        else "slow"
        if sample["duration"] <= sales_speed[sample["city"]][sample["single"]]["slow"][th_idx]
        else "moderate"
    )
        
    print(f"\n{sample["zpid"]}: {sample["duration"]} Days ({gt_class} {int(th_val*100)}%)")

    basic_system = build_basic_system(sales_speed, th_idx, sample["city"], sample["single"])
    basic_template = build_basic_template(basic_system)
    basic_grader = basic_template | basic_llm
    basic_result = basic_grader.invoke(
        input={
            "description": sample["description"],
            "attributes": zillow_to_str(sample),
        }
    )
    print(f"Basic: {basic_result}")

    words_system = build_words_system(sales_speed, th_idx, sample["city"], sample["single"])
    words_template = build_words_template(sample["city"], sample["single"], words_system, th_val)
    words_grader = words_template | words_llm
    words_result = words_grader.invoke(
        input={
            "description": sample["description"],
            "attributes": zillow_to_str(sample),
        }
    )
    print(f"Words: {words_result}")

    full_system = build_full_system(sales_speed, th_idx, sample["city"], sample["single"])
    full_template = build_full_template(sample["city"], sample["single"], full_system, th_val, th_idx)
    full_grader = full_template | full_llm
    full_result = full_grader.invoke(
        input={
            "description": sample["description"],
            "attributes": zillow_to_str(sample),
        }
    )
    print(f"Full: {full_result}")
