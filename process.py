from datetime import date
import numpy as np
import pandas as pd


def preprocess(data, click_weight, conversion_weight):
    """
    Prepare dataframe from CSV with channel, date, ad_id, impressions, clicks
    and conversions for bandit optinmization
    """
    # Standardize column name input format
    data.columns = \
        [column.lower().replace(" ", "_") for column in data.columns]
    # Rename columns from Facebook export
    data.rename(columns={'reporting_ends': 'date'}, inplace=True)
    data.rename(columns={'link_clicks': 'clicks'}, inplace=True)
    data.rename(columns={'purchases': 'conversions'}, inplace=True)
    # Set empty and NaN impressions, clicks and conversions to 0
    # for calculations
    data = data.replace('', np.nan)
    data['impressions'].fillna(value=0, downcast='infer', inplace=True)
    data['clicks'].fillna(value=0, downcast='infer', inplace=True)
    data['conversions'].fillna(value=0, downcast='infer', inplace=True)
    # Create successes column as weighted sum of clicks and conversions
    # maximise successes to trials = impressions
    data['successes'] = [min(row['clicks'] * click_weight +
                             row['conversions'] * conversion_weight,
                             row['impressions'])
                         for index, row in data.iterrows()]
    # Rename impressions trials
    data.rename(columns={'impressions': 'trials'}, inplace=True)
    return data


def reindex_options(data):
    """
    Process dataframe with (channel), ad_id, date, trials and successes;
    return options and dataframe with option id column
    """
    if 'channel' in data.columns:
        combinations = data[['channel', 'ad_id']]
    else:
        combinations = data['ad_id']
    options = combinations.drop_duplicates().reset_index().drop('index',
                                                                axis='columns')
    data['option_id'] = 0
    for i in range(len(data)):
        data.at[i, 'option_id'] = options.loc[
            options['ad_id'] == data.iloc[i]['ad_id']].index[0]
    return [options, data]


def add_days(data):
    """
    Process dataframe with ad_id, date, trials, successes;
    return dataframe with new days_ago column
    """
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d').dt.date
    data['days_ago'] = 0
    for i in range(len(data)):
        data.at[i, 'days'] = (date.today() - data.iloc[i]['date']).days
    return data
