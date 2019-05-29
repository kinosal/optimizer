from datetime import date
import numpy as np
import pandas as pd


def facebook(data, click_weight, purchase_weight):
    """
    Process dataframe from Facebook CSV with Ad-, Ad Set-, Campaign ID,
    Reporting Ends, Impressions, Link Clicks, Purchases;
    return normalized columns for further processing
    """
    # Standardize column name input format
    data.columns = [column.lower().replace(" ", "") for column in data.columns]
    # Set empty and NaN Impressions, Link Clicks and Purchases to 0
    # for calculations
    data = data.replace('', np.nan)
    data['impressions'].fillna(value=0, downcast='infer', inplace=True)
    data['linkclicks'].fillna(value=0, downcast='infer', inplace=True)
    data['purchases'].fillna(value=0, downcast='infer', inplace=True)
    # Create successes column as weighted sum of link clicks and purchases
    # maximise successes to trials = impressions
    data['successes'] = [min(row['linkclicks'] * click_weight +
                             row['purchases'] * purchase_weight,
                             row['impressions'])
                         for index, row in data.iterrows()]
    # Extract and rename relevant columns
    data = data[['adid', 'adsetid', 'campaignid',
                 'reportingends', 'impressions', 'successes']]
    data.columns = ['ad_id', 'adset_id', 'campaign_id',
                    'date', 'trials', 'successes']
    return data


def reindex_options(data):
    """
    Process dataframe with ad-, adset-, campaign id, date, trials, successes;
    return options and dataframe with new option id column
    """
    options = data[['ad_id', 'adset_id', 'campaign_id']] \
        .drop_duplicates().reset_index().drop('index', axis='columns')
    data['option_id'] = 0
    for i in range(len(data)):
        data.at[i, 'option_id'] = options.loc[
            options['ad_id'] == data.iloc[i]['ad_id']].index[0]
    return [options, data]


def add_days(data):
    """
    Process dataframe with ad-, adset-, campaign id, date, trials, successes;
    return dataframe with new days_ago column
    """
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d').dt.date
    data['days_ago'] = 0
    for i in range(len(data)):
        data.at[i, 'days'] = (date.today() - data.iloc[i]['date']).days
    return data
