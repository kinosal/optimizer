import datetime
import pytest
from api import app


@pytest.fixture
def client():
    return app.test_client()


def test_index(client):
    response = client.get('/')
    assert response.status_code == 200


def test_json(client):
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

    response = client.post('/json', json=data)

    assert response.status_code == 200
    assert b'ad_id' in response.data
    assert b'ad_share' in response.data


def test_get_form(client):
    response = client.get('/form')
    assert response.status_code == 200


def test_post_form(client):
    response = client.post('/form', data={
        'trials_1': 1000, 'successes_1': 100,
        'trials_2': 1000, 'successes_2': 50
    })
    assert response.status_code == 200
    assert b'option' in response.data
    assert b'ad_share' in response.data


def test_get_csv(client):
    response = client.get('/csv')
    assert response.status_code == 200


def test_post_csv(client):
    date = str(datetime.date.today() - datetime.timedelta(days=1))
    response = client.post('/csv', data={
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
