# SVR-CPO
This repository contains the implementation, data, and results for a Conditional Portfolio Optimization (CPO) framework leveraging Support Vector Regression (SVR). The project includes all necessary datasets, model training scripts, hyperparameter tuning files, trained models, and backtesting results.

## Repository Structure
daily_dbs/

Contains all databases used in the project, including historical datasets for model training and backtesting.

*_daily_*.csv: Main time-series datasets used for training the XGBoost and SVR models.

market_caps: Market caps database used to train the KMeans model used for portfolio classification

risk_free_rate.csv: Historical data for the USA risk free rate used to calculate sortino and sharpe ratio.

dbs_backtesting.csv: Database used for the backtesting.

Other CSV/XLSX files with market features and benchmark information.

hyperparams_tunning/

Files generated during the hyperparameter tuning process, including parameter grids and optimization results for SVR and XGBoost models.

plots2/

Backtesting results in visual form. Includes:

Boxplots comparing strategies

Histograms of performance metrics

Other plots derived from simulation results

And two files containing the results for all the 1,000 simulations made: in csv and pkl form.

Aditionally, the table with the mean of all the performance metrics across all the simulations.

source/

All core code for the project:

Backtesting scripts

Utility and function definitions

Classifier and optimization logic

trained_models/

Serialized (.pkl) trained models for different asset groups and configurations.
These files can be loaded directly for inference or further testing without retraining.

## Key Features

Implements Conditional Portfolio Optimization with SVR

Backtesting framework for performance evaluation

Hyperparameter tuning for model optimization

Visualization of simulation and performance metrics

Pre-trained models for reproducibility

## Requirements

To run the code, install dependencies:

pip install -r requirements.txt

Usage

Place any additional datasets in daily_dbs/.

Use scripts in source/ for training or backtesting.

View results in plots2/ after running simulations.

## License

MIT License
