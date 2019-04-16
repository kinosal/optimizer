from datetime import date
import numpy as np
import pandas as pd


def facebook(data, purchase_factor):
    """
    Process dataframe from Facebook CSV with Ad-, Ad Set-, Campaign ID,
    Reporting Ends, Impressions, Link Clicks, Purchases;
    return normalized columns for further processing
    """
    # Set empty and NaN Impressions, Link Clicks and Purchases to 0
    # for calculations
    data = data.replace('', np.nan)
    data['Impressions'].fillna(value=0, downcast='infer', inplace=True)
    data['Link Clicks'].fillna(value=0, downcast='infer', inplace=True)
    data['Purchases'].fillna(value=0, downcast='infer', inplace=True)
    # Create successes column as Link Clicks + Purchases * purchase factor,
    # maximise successes (alpha) to trials = impressions (alpha + beta) for PDF
    data['successes'] = [min(row['Link Clicks'] +
                             row['Purchases'] * purchase_factor,
                             row['Impressions'])
                         for index, row in data.iterrows()]
    # Extract and rename relevant columns
    data = data[['Ad ID', 'Ad Set ID', 'Campaign ID',
                 'Reporting Ends', 'Impressions', 'successes']]
    data.columns = ['ad_id', 'adset_id', 'campaign_id',
                    'date', 'trials', 'successes']
    return data


def add_option_id(data):
    """
    Process dataframe with ad-, adset-, campaign id, date, trials, successes;
    return dataframe with new option id column
    """
    options = data[['ad_id', 'adset_id', 'campaign_id']] \
        .drop_duplicates().reset_index().drop('index', axis='columns')
    data['option_id'] = 0
    for i in range(len(data)):
        data.at[i, 'option_id'] = options.loc[
            options['ad_id'] == data.iloc[i]['ad_id']].index[0]
    return [options, data]


def add_distance(data):
    """
    Process dataframe with ad-, adset-, campaign id, date, trials, successes;
    return dataframe with new days_ago column
    """
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d').dt.date
    data['days_ago'] = 0
    for i in range(len(data)):
        data.at[i, 'days_ago'] = (date.today() - data.iloc[i]['date']).days
    return data
