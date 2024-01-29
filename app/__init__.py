"""Flask app factory."""

import os
import ast
from io import StringIO

from flask import Flask, request, render_template
from flask_sqlalchemy import SQLAlchemy
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
import numpy as np
import pandas as pd
from scipy.stats import beta
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

# from facebook_business.api import FacebookAdsApi
# from facebook_business.adobjects.ad import Ad

from app.scripts import process as pro
from app.scripts import bandit as ban


db = SQLAlchemy()


CUTOFF = 14
CUT_LEVEL = 0.5
SHAPE = "linear"


def create_app(config_class: object):
    """Create Flask app.

    Args:
        config_class: configuration for Flask app
    """
    if os.environ.get("FLASK_ENV") == "production":  # pragma: no cover
        sentry_sdk.init(
            dsn="https://db6bdfc312434b7687d739a0c44ec603@sentry.io/1513206",
            integrations=[FlaskIntegration(), SqlalchemyIntegration()],
            traces_sample_rate=1.0,
        )
    app = Flask(__name__)
    app.config.from_object(config_class)
    db.init_app(app)

    from app.api.v1 import api_v1

    app.register_blueprint(api_v1)

    @app.route("/ping", methods=["GET", "POST"])
    def ping():
        """Return string to show the server is alive."""
        return "Server is here"

    @app.route("/", methods=["GET"])
    def root():
        """Root page with links to simple and CSV form."""
        return render_template("index.html")

    @app.route("/json", methods=["POST"])
    def json():
        """Get optimal next period budget shares for ad options."""
        # TODO: Remove this route in favor of api/v1/ads after informing users

        if "optimize" not in request.json:  # pragma: no cover
            if "stats" not in request.json:
                return '"optimize" and "stats" keys missing in posted JSON object'
            return '"optimize" key missing in posted JSON object'
        if "stats" not in request.json:  # pragma: no cover
            return '"stats" key missing in posted JSON object'

        if not request.json["optimize"]:  # pragma: no cover
            if not request.json["stats"]:
                return '"optimize" and "stats" keys are empty'
            return '"optimize" key is empty'
        if not request.json["stats"]:  # pragma: no cover
            return '"stats" key is empty'

        weights = {
            "impression_weight": 0,
            "engagement_weight": 0,
            "click_weight": 0,
            "conversion_weight": 0,
        }
        for metric in request.json["optimize"]:
            weights[metric[:-1] + "_weight"] = None

        data = pd.DataFrame(request.json["stats"])
        data = pro.preprocess(data, **weights)
        data = pro.filter_dates(data, cutoff=CUTOFF)
        [options, data] = pro.reindex_options(data)

        bandit = ban.Bandit(
            num_options=len(options),
            memory=True,
            shape=SHAPE,
            cutoff=CUTOFF,
            cut_level=CUT_LEVEL,
        )
        bandit.add_daily_results(data)
        shares = bandit.calculate_shares(accelerate=True)
        options["ad_share"] = shares.tolist()

        return options.to_json(orient="records")

    @app.route("/form", methods=["GET", "POST"])
    def form():
        """
        Provide form with cumulated trial and success inputs for multiple options,
        return options with suggested budget share for next period
        """

        if request.method == "POST":
            entries = [value for value in list(request.form.values()) if value]
            num_options = int(len(entries) / 2)
            options = pd.DataFrame([{"option": str(i + 1)} for i in range(num_options)])
            trials = [int(entries[i * 2]) for i in range(num_options)]
            successes = [int(entries[i * 2 + 1]) for i in range(num_options)]
            bandit = ban.Bandit(num_options=num_options, memory=False)
            for i in range(num_options):
                bandit.add_results(
                    option_id=i, trials=trials[i], successes=successes[i]
                )
            shares = bandit.calculate_shares(accelerate=False)
            options = format_results(options, shares)
            records = options.to_dict("records")
            columns = options.columns.values
            save_plot(bandit)
            return render_template(
                "form_result.html",
                records=records,
                columns=columns,
                plot="/static/images/plot.png",
            )

        return render_template("form.html")

    @app.route("/csv", methods=["GET", "POST"])
    def csv():
        """
        Provide form to paste ads CSV with daily breakdown of channel (optional),
        ad_id, impressions, engagements, clicks and conversions;
        return options with suggested budget share or status for next period and
        provide direct upload to Facebook via API
        """

        if request.method == "POST":
            if request.form["update"] == "true":  # pragma: no cover
                app_id = request.form["app_id"]
                app_secret = request.form["app_secret"]
                access_token = request.form["access_token"]
                channels = ast.literal_eval(request.form["channels"])
                records = ast.literal_eval(request.form["records"])
                updatable = ["facebook", "instagram"]
                indices = []
                for channel in updatable:
                    if channel in channels:
                        indices.append(channels.index(channel))
                results = pd.DataFrame(columns=["ad_id", "ad_status"])
                for index in indices:
                    for record in records[index]:
                        results.loc[len(results)] = [
                            record["ad_id"],
                            record["ad_status"],
                        ]
                updated = update_facebook(app_id, app_secret, access_token, results)
                records = updated.to_dict("records")
                columns = updated.columns.values
                return render_template(
                    "update_result.html", records=records, columns=columns
                )

            weights = {}
            for weight in [
                "impression_weight",
                "engagement_weight",
                "click_weight",
                "conversion_weight",
            ]:
                if request.form[weight] == "":
                    weights[weight] = None
                else:
                    weights[weight] = int(request.form[weight])

            data = pd.read_csv(StringIO(request.form["ads"]), sep=None, engine="python")

            try:
                data = pro.preprocess(
                    data,
                    weights["impression_weight"],
                    weights["engagement_weight"],
                    weights["click_weight"],
                    weights["conversion_weight"],
                )
            except Exception as error:  # pragma: no cover
                print(error)
                message = "Cannot pre-process your data. \
                         Please check the CSV input format and try again."
                return render_template(
                    "csv.html",
                    error=message,
                    output=request.form["output"],
                    impression_weight=request.form["impression_weight"],
                    engagement_weight=request.form["engagement_weight"],
                    click_weight=request.form["click_weight"],
                    conversion_weight=request.form["conversion_weight"],
                    ads=request.form["ads"],
                )

            try:
                data = pro.filter_dates(data, cutoff=CUTOFF)
            except Exception as error:  # pragma: no cover
                print(error)
                message = "Please check your dates (format should be YYYY-MM-DD)."
                return render_template(
                    "csv.html",
                    error=message,
                    output=request.form["output"],
                    impression_weight=request.form["impression_weight"],
                    engagement_weight=request.form["engagement_weight"],
                    click_weight=request.form["click_weight"],
                    conversion_weight=request.form["conversion_weight"],
                    ads=request.form["ads"],
                )
            if data.empty:  # pragma: no cover
                error = (
                    "Please include results with data from the past "
                    + str(CUTOFF)
                    + " days."
                )
                return render_template(
                    "csv.html",
                    error=error,
                    output=request.form["output"],
                    impression_weight=request.form["impression_weight"],
                    engagement_weight=request.form["engagement_weight"],
                    click_weight=request.form["click_weight"],
                    conversion_weight=request.form["conversion_weight"],
                    ads=request.form["ads"],
                )

            [options, data] = pro.reindex_options(data)

            bandit = ban.Bandit(
                num_options=len(options),
                memory=True,
                shape=SHAPE,
                cutoff=CUTOFF,
                cut_level=CUT_LEVEL,
            )
            bandit.add_daily_results(data)
            shares = bandit.calculate_shares(accelerate=True)

            output = request.form["output"]
            if output == "status":
                results = format_results(options, shares, status=True)
            elif output == "share":
                results = format_results(options, shares, status=False).round(2)

            if "channel" in options.columns:
                channel_shares = (
                    format_results(options, shares, status=False)
                    .groupby("channel")["ad_share"]
                    .sum()
                    .round(2)
                )
                channels = []
                records = []
                for name, group in results.groupby("channel"):
                    channels.append(name)
                    group = group.drop(["channel"], axis=1)
                    columns = group.columns.values
                    records.append(group.to_dict("records"))
                return render_template(
                    "csv_result_channels.html",
                    channels=channels,
                    channel_shares=channel_shares,
                    records=records,
                    columns=columns,
                )

            records = results.to_dict("records")
            columns = results.columns.values
            return render_template("csv_result.html", records=records, columns=columns)

        return render_template("csv.html")

    @app.after_request
    def after_request(response):
        """
        Clear server cache after every request to display newest plot image
        """
        response.headers["Cache-Control"] = (
            "no-cache, no-store, must-revalidate, public, max-age=0"
        )
        response.headers["Expires"] = 0
        response.headers["Pragma"] = "no-cache"
        return response

    def format_results(options, shares, status=False):
        """
        Return ACTIVE/PAUSED instead of numeric share for options if desired
        """
        results = options.copy()
        if status:
            status = shares > 0
            results["ad_status"] = status.tolist()
            results["ad_status"] = results["ad_status"].replace(True, "ACTIVE")
            results["ad_status"] = results["ad_status"].replace(False, "PAUSED")
        else:
            results["ad_share"] = shares.tolist()
            results["ad_share"] = results["ad_share"]
        return results

    def save_plot(bandit):
        """
        Save plot with bandit options' PDFs (beta distributions)
        """
        x = np.linspace(0, 1, 100)
        means = []
        stds = []
        for i in range(len(bandit.trials)):
            a = bandit.successes[i]
            b = bandit.trials[i] - bandit.successes[i]
            means.append(beta.mean(a, b))
            stds.append(beta.std(a, b))
            plt.plot(x, beta.pdf(x, a, b), label="option " + str(i + 1))

        i_min, min_mean = min(enumerate(means), key=lambda x: x[1])
        i_max, max_mean = max(enumerate(means), key=lambda x: x[1])
        plt.xlim(max(min_mean - 10 * stds[i_min], 0), min(max_mean + 10 * stds[i_max], 1))
        plt.xlabel("Success rate")
        plt.ylabel("Probablity density")
        plt.grid()
        plt.yticks([])
        plt.legend()
        plt.savefig("app/static/images/plot.png")
        plt.clf()

    def update_facebook(app_id, app_secret, access_token, options):  # pragma: no cover
        """
        Update status of ads on Facebook if different from respective suggestion;
        return dataframe with updated ads
        """
        api = FacebookAdsApi.init(app_id, app_secret, access_token)
        updated = []
        # Determine number of required batches since
        # Facebook only allows 50 API calls per batch
        num_options = len(options.index)
        batch_size = 50
        batches = int(num_options / batch_size) + 1
        # Split options into batches and loop through those
        i = 0
        for _ in range(batches):
            option_batch = options.loc[i : i + batch_size, :]
            i += batch_size
            update_batch = api.new_batch()
            # Loop through options within batch, compare current and suggested
            # ad status and update if changed
            for _, row in option_batch.iterrows():
                ad_id = str(row["ad_id"])
                ad = Ad(ad_id)
                ad.api_get(fields=[Ad.Field.status])
                old_status = ad[Ad.Field.status]
                new_status = row["ad_status"]
                if old_status != new_status:
                    ad[Ad.Field.status] = new_status
                    updated.append([ad_id, old_status + " -> " + new_status])
                    ad.api_update(
                        batch=update_batch,
                        fields=[],
                        params={Ad.Field.status: new_status},
                    )
            update_batch.execute()
        return pd.DataFrame(updated, columns=["ad_id", "updated"])

    return app
