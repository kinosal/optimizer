"""Unit tests."""

import datetime
import unittest

from app.config import TestingConfig
from app import create_app


class TestSetup(unittest.TestCase):
    """Unit testing setup."""

    def setUp(self):
        """Create new app and database for each test."""
        self.app = create_app(TestingConfig)
        self.app_context = self.app.app_context()
        self.app_context.push()

    def tearDown(self):
        """Remove databse and app context after each test."""
        self.app_context.pop()


class Tests(TestSetup):
    def test_index(self):
        response = self.app.test_client().get('/')
        assert response.status_code == 200

    def test_ping(self):
        response = self.app.test_client().get('/ping')
        assert response.status_code == 200

    def test_json(self):
        date = str(datetime.date.today() - datetime.timedelta(days=1))
        data = {
            "optimize": ["clicks", "engagements", "conversions"],
            "stats": [
                {
                    "date": date,
                    "ad_id": 1,
                    "cost": 15,
                    "impressions": 1000,
                    "engagements": 100,
                    "clicks": 10,
                    "conversions": 1
                }
            ]
        }

        response = self.app.test_client().post('/json', json=data)

        assert response.status_code == 200
        assert b'ad_id' in response.data
        assert b'ad_share' in response.data

    def test_get_form(self):
        response = self.app.test_client().get('/form')
        assert response.status_code == 200

    def test_post_form(self):
        response = self.app.test_client().post('/form', data={
            'trials_1': 1000, 'successes_1': 100,
            'trials_2': 1000, 'successes_2': 50
        })
        assert response.status_code == 200
        assert b'option' in response.data
        assert b'ad_share' in response.data

    def test_get_csv(self):
        response = self.app.test_client().get('/csv')
        assert response.status_code == 200

    def test_post_csv_status_channel(self):
        date = str(datetime.date.today() - datetime.timedelta(days=1))
        response = self.app.test_client().post('/csv', data={
            'ads': """channel,date,ad_id,cost,impressions,engagements,clicks,conversions
                      facebook,{},1,1500,1000,100,10,1""".format(date),
            'update': 'false',
            'impression_weight': '',
            'engagement_weight': '',
            'click_weight': '',
            'conversion_weight': '',
            'output': 'status'
        })
        assert response.status_code == 200
        assert b'ad_id' in response.data
        assert b'ad_status' in response.data

    def test_post_csv_status_weights(self):
        date = str(datetime.date.today() - datetime.timedelta(days=1))
        response = self.app.test_client().post('/csv', data={
            'ads': """date,ad_id,cost,impressions,engagements,clicks,conversions
                      {},1,1500,1000,100,10,1""".format(date),
            'update': 'false',
            'impression_weight': '100',
            'engagement_weight': '0',
            'click_weight': '0',
            'conversion_weight': '0',
            'output': 'status'
        })
        assert response.status_code == 200
        assert b'ad_id' in response.data
        assert b'ad_status' in response.data

    def test_post_csv_shares(self):
        date = str(datetime.date.today() - datetime.timedelta(days=1))
        response = self.app.test_client().post('/csv', data={
            'ads': """date,ad_id,cost,impressions,engagements,clicks,conversions
                      {},1,1500,1000,100,10,1""".format(date),
            'update': 'false',
            'impression_weight': '',
            'engagement_weight': '',
            'click_weight': '',
            'conversion_weight': '',
            'output': 'share'
        })
        assert response.status_code == 200
        assert b'ad_id' in response.data
        assert b'ad_share' in response.data
