"""Unit tests."""

import os
import datetime
import unittest

from app.config import TestingConfig
from app import create_app, db
from app.models.models import User


class TestSetup(unittest.TestCase):
    """Unit testing setup."""

    def setUp(self):
        """Create new app and database for each test."""
        self.app = create_app(TestingConfig)
        self.app_context = self.app.app_context()
        self.app_context.push()
        db.create_all()

    def tearDown(self):
        """Remove databse and app context after each test."""
        db.session.remove()
        db.drop_all()
        self.app_context.pop()


class TestBasic(TestSetup):
    """Basic default tests."""

    def test_index(self):
        response = self.app.test_client().get('/')
        assert response.status_code == 200

    def test_ping(self):
        response = self.app.test_client().get('/ping')
        assert response.status_code == 200


class TestApp(TestSetup):
    """Test app factory."""

    def test_json(self):
        data = {
            "optimize": ["clicks", "engagements", "conversions"],
            "stats": [
                {
                    "date": str(datetime.date.today() - datetime.timedelta(days=1)),
                    "ad_id": "1234",
                    "cost": 1000,
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
                      facebook,{},1234,1000,1000,100,10,1""".format(date),
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


class TestApi(TestSetup):
    """Test api routes."""

    def setUp(self):
        super(TestApi, self).setUp()
        self.payload = {
            "optimize": ["clicks", "engagements", "conversions"],
            "stats": [
                {
                    "date": str(datetime.date.today() - datetime.timedelta(days=1)),
                    "ad_id": "1234",
                    "cost": 1000,
                    "impressions": 1000,
                    "engagements": 100,
                    "clicks": 10,
                    "conversions": 1
                }
            ]
        }
        db.session.add(User(name='Test', api_key='valid_key'))
        db.session.commit()

    def test_ads_400(self):
        response = self.app.test_client().post('/api/v1/ads', json={})
        assert response.status_code == 400

    def test_ads_401(self):
        response = self.app.test_client().post('/api/v1/ads', json=self.payload)
        assert response.status_code == 401

        response = self.app.test_client().post(
            '/api/v1/ads', json=self.payload, headers={"API_KEY": "invalid_key"}
        )
        assert response.status_code == 401

    def test_ads_200(self):
        response = self.app.test_client().post(
            '/api/v1/ads', json=self.payload, headers={"API_KEY": "valid_key"}
        )
        assert response.status_code == 200
        assert b'ad_id' in response.data
        assert b'ad_share' in response.data
