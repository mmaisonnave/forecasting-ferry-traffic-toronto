import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt


class RedemptionModel:

    def __init__(self, X, target_col, exog_cols):
        '''
        Args:
        X (pandas.DataFrame): Dataset of predictors, output from load_data()
        target_col (str): column name for target variable
        '''
        self._predictions = {}
        self.X = X
        self.target_col = target_col
        self.exog_cols = exog_cols
        self.results = {} # dict of dicts with model results

    def score(self, truth, preds):
        # Score our predictions - modify this method as you like
        return {
            'MAPE': MAPE(truth[truth!=0], preds[truth!=0]),
            'MAE': MAE(truth, preds),
            'MSE': MSE(truth, preds),
        }


    def run_models(self, n_splits=4, test_size=365):
        '''Run the models and store results for cross validated splits in
        self.results.
        '''

        # As requested, I am leaving the base model in place, I am grouping all
        # other models in a dictionary, so models can be added easily.
        modelname2method = {
            'Historical Average By Day': self._redemptions_from_previous_years_by_day,
            'ARIMA with Exogenous': self._arima_model_with_exog,
        }

        # Time series split
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        cnt = 0 # keep track of splits
        for train, test in tscv.split(self.X):
            X_train = self.X.iloc[train]
            X_test = self.X.iloc[test]
            # Base model - please leave this here
            preds = self._base_model(X_train, X_test)
            if 'Base' not in self.results:
                self.results['Base'] = {}
            self.results['Base'][cnt] = self.score(X_test[self.target_col],
                                preds)
            self.plot(preds, 'Base')

            for modelname, method in modelname2method.items():
                preds = method(X_train, X_test)
                if modelname not in self.results:
                    self.results[modelname] = {}
                self.results[modelname][cnt] = self.score(
                    X_test[self.target_col], preds)
                self.plot(preds, modelname)

            cnt += 1


    def _base_model(self, train, test):
        '''
        Our base, too-simple model.
        Your model needs to take the training and test datasets (dataframes)
        and output a prediction based on the test data.

        Please leave this method as-is.

        '''
        res = sm.tsa.seasonal_decompose(train[self.target_col],
                                        period=365)
        res_clip = res.seasonal.apply(lambda x: max(0,x))
        res_clip.index = res_clip.index.dayofyear
        res_clip = res_clip.groupby(res_clip.index).mean()
        res_dict = res_clip.to_dict()
        return pd.Series(index = test.index, 
                         data = map(lambda x: res_dict[x], test.index.dayofyear))
    
    def _build_exog_prediction_for_test(self, train, test):
        # Copy part of the data frame:
        exog_preds = pd.DataFrame(index=test.index)
        for col in self.exog_cols:
            history = train[col]
            history.index = history.index.dayofyear
            history = history.groupby(history.index).mean()
            history_dict = history.to_dict()
            exog_preds[col] = list(map(lambda x: history_dict[x], test.index.dayofyear))
        return exog_preds
    
    

    def _arima_model_with_exog(self, train, test):
	# Extract the target series
        y_train = train[self.target_col].reset_index(drop=True)
        exog_train = train[self.exog_cols].reset_index(drop=True)
        exog_test = self._build_exog_prediction_for_test(train, test)

        # Fit the ARIMA model
        model = ARIMA(y_train, order=(2, # P: autoregressive order
                                    0, # D: differencing order
                                    5), # Q: moving average order
                                    exog=exog_train
                                    ) # (p,d,q)
                                    
        model_fit = model.fit()
        # Forecast for the length of the test set
        forecast = model_fit.get_forecast(steps=len(test),
                                          exog=exog_test
        ).predicted_mean
        
        # Return a series with the same index as the test set
        
        return pd.Series(forecast.values, index = test.index)



    

    def _redemptions_from_previous_years_by_day(self, train, test):
        history = train[self.target_col]

        # Transform index to numbers to use groupby
        history.index = history.index.dayofyear

        # Group by using average as grouping cunction
        history = history.groupby(history.index).mean()

        # Return the historic average as a prediction.
        history_dict = history.to_dict()
        
        return pd.Series(index = test.index, 
                        data = map(lambda x: history_dict[x], test.index.dayofyear))
    
    

    def plot(self, preds, label):
        # plot out the forecasts
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.scatter(self.X.index, self.X[self.target_col], s=0.4, color='grey',
            label='Observed')
        ax.plot(preds, label = label, color='red')
        plt.legend()
