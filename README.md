# Variant Optimizer

Source code for my blog post about [When and when not to A/B test](https://medium.com/@nikolasschriefer/when-and-when-not-to-a-b-test-c901f3ad96d9)

Contains the code to simulate variant testing with Split (A/B) test and multi-armed bandit (Thompson sampling) and provides a simple Python (Flask) web app to demo the bandit method and optimize the ad delivery of a Facebook marketing campaign at [optimizer.stagelink.com](https://optimizer.stagelink.com)

## Structure

This project contains two parts, a simulator and a web app incl. API.

1) Simulator: Defined in app/scripts/simulation.py, consuming split.py and bandit.py (the two competing optimization models)

2) App: Endpoints/routes defined in app (init), consuming scripts/bandit.py (optimization model) and scripts/process.py (for data pre-processing). HTML views in templates, plot image for form result page saved as static/images/plot.png, JS to add form elements in static/js/form.js.

## Contribution

Please submit any [issues](https://github.com/kinosal/optimizer/issues) you have. If you have any ideas how to further improve the optimizer please get in touch or feel free to fork this project and create a pull request with your proposed updates.

## Environment

- Python 3.8
- Flask (web framework)
- zappa (deployment to AWS lambda)
- numpy (Python computing package)
- pandas (Python data analytics library)
- scipy (Python statistics library)

## Example simulation

Import simulation
```Python
from app.scripts import simulation as sim
import random
```

Define variants' true success rates and number of trials per period
```Python
true_rates = [random.uniform(0.01, 0.04) for _ in range(0, 50)]
trials = len(true_rates) * 100
```

A) Calculate results comparing Split and Bandit
```Python
results = sim.compare_methods(
    periods=28, true_rates=true_rates, deviation=0.5, change=0.05,
    trials=trials, max_p=0.1, rounding=True, accelerate=True)
```

B) Calculate results comparing parameter values for one optimization model
```Python
results = sim.compare_params(
    method='split', param='max_p',
    values=[0.05, 0.1, 0.15, 0.2],
    periods=28, true_rates=true_rates, deviation=0.5, change=0,
    trials=trials, max_p='param', rounding=True, accelerate=True,
    memory=False, shape='linear', cutoff=14, cut_level=0.5)
```

Visualize the results
```Python
sim.plot(results['periods'], results['parameters'], relative=True)
```
