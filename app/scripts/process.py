import datetime
import numpy as np
import pandas as pd


def preprocess(
    data,
    impression_weight=None,
    engagement_weight=None,
    click_weight=None,
    conversion_weight=None,
):
    """
    Prepare dataframe from CSV with channel (optional), date, ad_id,
    spend, impressions, engagements, clicks and conversions
    for bandit optimization
    """

    # Standardize column name input format
    data.columns = [column.lower().replace(" ", "_") for column in data.columns]

    # Rename columns from Facebook export
    data.rename(columns={"reporting_ends": "date"}, inplace=True)
    data.rename(columns={"amount_spent_(eur)": "cost"}, inplace=True)
    data.rename(columns={"post_engagement": "engagements"}, inplace=True)
    data.rename(columns={"link_clicks": "clicks"}, inplace=True)
    data.rename(columns={"purchases": "conversions"}, inplace=True)
    if "reporting_starts" in data.columns:  # pragma: no cover
        data.drop(["reporting_starts"], axis="columns", inplace=True)

    # Rename columns from Google export
    data.rename(columns={"day": "date"}, inplace=True)
    if "currency" in data.columns:  # pragma: no cover
        data.drop(["currency"], axis="columns", inplace=True)

    # Drop rows with missing required entries
    data.dropna(axis="index", inplace=True, subset=["ad_id", "date"])

    # Set relevant empty and NaN values to 0 for calculations
    data = data.replace("", np.nan)
    for column in ["cost", "impressions", "engagements", "clicks", "conversions"]:
        if column in data.columns:
            # data[column] = data[column].fillna(value=0.0, downcast='infer')
            data[column] = data[column].fillna(value=0.0)
        else:  # pragma: no cover
            data[column] = 0.0

    # Remove rows with 0 cost (ads that did not run)
    data = data[data["cost"] != 0]

    # If not provided, set weights to respective cost ratios
    weights = {}
    for weight in ["impression", "engagement", "click", "conversion"]:
        if locals()[weight + "_weight"] is None:
            if data[weight + "s"].sum() == 0:  # pragma: no cover
                weights[weight + "_weight"] = 0
            else:
                weights[weight + "_weight"] = (
                    data["cost"].sum() / data[weight + "s"].sum()
                )
        else:
            weights[weight + "_weight"] = locals()[weight + "_weight"]

    # Create successes column as weighted sum of success metrics
    data["successes"] = [
        row["impressions"] * weights["impression_weight"]
        + row["engagements"] * weights["engagement_weight"]
        + row["clicks"] * weights["click_weight"]
        + row["conversions"] * weights["conversion_weight"]
        for index, row in data.iterrows()
    ]

    # Create trials column as costs + successes + 1
    # to guarantee successes <= trials and correct for free impressions
    data["trials"] = [
        int(row["cost"]) + row["successes"] + 1 for index, row in data.iterrows()
    ]

    # Drop processed columns
    drop = ["cost", "impressions", "engagements", "clicks", "conversions"]
    data.drop(drop, axis="columns", inplace=True)

    return data.reset_index(drop=True)


def filter_dates(data, cutoff):
    """Return data with dates in cutoff range"""
    data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d").dt.date
    data = data.loc[
        data["date"] >= datetime.date.today() - datetime.timedelta(days=cutoff)
    ]
    return data


def reindex_options(data):
    """
    Process dataframe with ad_id, date, trials and successes;
    return options and dataframe with option id column
    """
    combinations = data.drop(["date", "trials", "successes"], axis="columns")
    options = combinations.drop_duplicates().reset_index().drop("index", axis="columns")
    data["option_id"] = 0
    for i in range(len(data)):
        option_id = options.loc[options["ad_id"] == data.iloc[i]["ad_id"]].index[0]
        data.at[i, "option_id"] = option_id
    return [options, data]
