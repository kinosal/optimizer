"""Endpoints for API v1."""

from typing import List, Dict

from flask import request, abort
from flask_restx import Api, Resource, fields
import pandas as pd

from app.api import require_auth
from app.api.v1 import api_v1
from app.scripts import process as pro
from app.scripts import bandit as ban

api = Api(
    api_v1,
    version="1.0",
    title="Ad Optimizer",
    description="Optimizing your ads",
)

ads = api.namespace("ads", description="Analyze ad variants")

stats = api.model(
    "stats",
    {
        "channel": fields.String,
        "date": fields.Date(format="iso8601"),  # YYYY-MM-DD
        "ad_id": fields.String,
        "cost": fields.Integer,  # ad spend in cents
        "impressions": fields.Integer,
        "engagements": fields.Integer,
        "clicks": fields.Integer,
        "conversions": fields.Integer,
    },
)

ads_request = api.model(
    "Ads Request",
    {
        "optimize": fields.List(
            fields.String,
            required=True,
            description="any combination of 'impressions', 'engagements', 'clicks' and 'conversions'"
        ),
        "stats": fields.List(fields.Nested(stats), required=True),
    },
)

ads_response = api.model(
    "Ads Response",
    {
        "channel": fields.String,
        "ad_id": fields.String,
        "ad_share": fields.Float,
    },
)


@ads.route('')
class Ads(Resource):
    @require_auth
    @api.doc(
        responses={200: "Success", 400: "Bad Request", 401: "Unauthorized"},
        params={"API_KEY": {"in": "header"}},
    )
    @api.expect(ads_request, validate=True)
    @api.marshal_list_with(ads_response)
    def post(self) -> List[Dict]:
        """Get optimal next period budget shares for ad options."""
        if not request.json['optimize']:  # pragma: no cover
            if not request.json['stats']:
                abort(400, '"optimize" and "stats" keys are empty')
            abort(400, '"optimize" key is empty')
        if not request.json['stats']:  # pragma: no cover
            abort(400, '"stats" key is empty')

        weights = {
            'impression_weight': 0,
            'engagement_weight': 0,
            'click_weight': 0,
            'conversion_weight': 0,
        }
        for metric in request.json['optimize']:
            weights[metric[:-1] + '_weight'] = None

        data = pd.DataFrame(request.json['stats'])
        data = pro.preprocess(data, **weights)
        data = pro.filter_dates(data, cutoff=14)
        [options, data] = pro.reindex_options(data)

        bandit = ban.Bandit(
            num_options=len(options),
            memory=True,
            shape='linear',
            cutoff=14,
            cut_level=0.5,
        )
        bandit.add_daily_results(data)
        shares = bandit.calculate_shares(accelerate=True)
        options['ad_share'] = shares.tolist()

        return options.to_dict(orient='records')
