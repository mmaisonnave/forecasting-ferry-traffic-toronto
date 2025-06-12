"""
Model.py

This module contains the Model class, which implements various time series forecasting models.
It includes methods for running models, scoring predictions, and plotting results.

"""
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from prophet import Prophet
import logging

def initialize_logger():

    # Allow printing logs to stdout
    _ALLOW_STDOUT = False


    # Retrieve log file path from configuration
    log_filepath = 'log/experiments.log'

    # Create a custom logger
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)  # Set the minimum level of severity to capture
    logger.propagate = False  # Prevent propagation to the root logger

    # Set up file handler
    file_handler: logging.FileHandler = logging.FileHandler(log_filepath)

    # Set up console handler if enabled
    console_handler = None
    if _ALLOW_STDOUT:
        console_handler = logging.StreamHandler()

    # Set log message format
    formatter: logging.Formatter = logging.Formatter('[%(asctime)s] [%(levelname)-7s] %(message)s')
    file_handler.setFormatter(formatter)
    if _ALLOW_STDOUT and console_handler:
        console_handler.setFormatter(formatter)

    # Attach handlers to the logger if they haven't been added already
    if not logger.hasHandlers():
        logger.addHandler(file_handler)
        if _ALLOW_STDOUT and console_handler:
            logger.addHandler(console_handler)

    logging.getLogger("cmdstanpy").disabled = True #  turn 'cmdstanpy' logs off


    return logger

logger = initialize_logger()

class Model:

    def __init__(self, X, target_col, experiment_configuration=None):
        '''
        Args:
        X (pandas.DataFrame): Dataset of predictors, output from load_data()
        target_col (str): column name for target variable
        '''
        assert isinstance(X, pd.DataFrame), "X must be a pandas DataFrame"

        logger.debug("Initializing Model class, Model(target_col=%s).", target_col)
        logger.debug("Dataframe shape=%s", X.shape)
        logger.debug("Dataframe columns=%s", X.columns)
        logger.debug("Experiment configuration=%s", experiment_configuration)

        self._predictions = {}
        self.X = X
        self.target_col = target_col
        self.results = {} # dict of dicts with model results
        self.experiment_configuration = experiment_configuration

        self.modelname2method = {
            'HISTORIAL_AVERAGE_BY_DAY': self._redemptions_from_previous_years_by_day,
            'ARIMAX': self._arimax_model,
            'ARIMAX_ON_RESIDUALS': self._arimax_on_residuals,
            'SARIMAX': self._sarimax_model,
            'PROPHET': self._prophet_model,
            'ENSEMBLE': self._ensemble_model,
        }


    def score(self, truth, preds):
        """ Calculate and return a dictionary of scores for the predictions.
        Args:
            truth (pd.Series): True values of the target variable.
            preds (pd.Series): Predicted values of the target variable.
        Returns:
            dict: Dictionary containing MAPE, MAE, and MSE scores.
        """
        return {
            'MAPE': MAPE(truth[truth!=0], preds[truth!=0]),
            'MAE': MAE(truth, preds),
            'MSE': MSE(truth, preds),
        }


    def run_models(self, n_splits=4, test_size=365):
        '''Run the models and store results for cross validated splits in
        self.results.
        '''


        # Filter modelname2method using self.experiment_configuration['MODELS_TO_RUN']
        # self.experiment_configuration['MODELS_TO_RUN']
        modelname2method = {}
        if self.experiment_configuration is not None:
            modelname2method = {k: v for k, v in self.modelname2method.items()
                                if k in self.experiment_configuration['MODELS_TO_RUN']}
            
        logger.debug("Models to run: %s", modelname2method.keys())

        # Time series split
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        cnt = 0 # keep track of splits
        for train, test in tscv.split(self.X):
            X_train = self.X.iloc[train]
            X_test = self.X.iloc[test]
            logger.debug("Running split %d: train shape=%s, test shape=%s", cnt, X_train.shape, X_test.shape)
            # Base model - please leave this here
            preds = self._base_model(X_train, X_test)
            assert isinstance(preds, pd.Series), "Base model must return a pandas Series"
            assert len(preds) == len(X_test), "Base model predictions must match test set length"

            if 'Base' not in self.results:
                self.results['Base'] = {}

            logger.debug("RUNNING BASE MODEL...")
            self.results['Base'][cnt] = self.score(X_test[self.target_col],
                                preds)
            
            logger.debug("X_test[self.target_col].shape=%s, preds.shape=%s",
                                X_test[self.target_col].shape, preds.shape)
            
            logger.debug("X_test[self.target_col] values=%s",
                                X_test[self.target_col].values[:5])
            logger.debug("preds values=%s", preds.values[:5])
            self.plot(preds, 'Base')
            logger.debug('')

            for modelname, method in modelname2method.items():
                logger.debug("RUNNING MODEL %s...", modelname)
                preds = method(X_train.copy(), X_test.copy())
                if modelname not in self.results:
                    self.results[modelname] = {}
                self.results[modelname][cnt] = self.score(
                    X_test[self.target_col], preds)
                self.plot(preds, modelname)

                logger.debug("X_test[self.target_col].shape=%s, preds.shape=%s",
                                X_test[self.target_col].shape, preds.shape)
                logger.debug("X_test[self.target_col] values=%s",
                                X_test[self.target_col].values[:5])
                logger.debug("preds values=%s", preds.values[:5])
                logger.debug("")

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
    
    def _build_exog_prediction_for_test(self, train, test_index, exog_cols):
        # Copy part of the data frame:
        exog_preds = pd.DataFrame(index=test_index)
        for col in exog_cols:
            history = train[col]
            history.index = history.index.dayofyear
            history = history.groupby(history.index).mean()
            history_dict = history.to_dict()
            exog_preds[col] = list(map(lambda x: history_dict[x], test_index.dayofyear))
        return exog_preds
    

    def _redemptions_from_previous_years_by_day(self, train, test):
        """
        Predict redemption counts based on the average of the same day in previous years.
        Args:
            train (pd.DataFrame): Training data containing the target column, used to compute the historical average.
            test (pd.DataFrame): Test data, only used for indexing the predictions.
        Returns:
            pd.Series: Predicted redemption counts for the test set, indexed by the test set index.
        """

        # This method calculates the average redemption count for each day of the year
        # from the training data using the groupby operation with `mean`` as the aggregation function.
        history = train[self.target_col]
        history.index = history.index.dayofyear
        history = history.groupby(history.index).mean()
        history_dict = history.to_dict()
        
        return pd.Series(
            index = test.index,
            data = map(lambda x: history_dict[x],
                       test.index.dayofyear)).round().astype(int)


    def _arimax_model(self, train, test):
        """ ARIMA model with exogenous variables (ARIMAX).
        Args:
            train (pd.DataFrame): Training data containing the target column and exogenous variables.
            test (pd.DataFrame): Test data, only used for indexing the predictions and exog variables.
        Returns:
            pd.Series: Predicted values for the test set, indexed by the test set index.
        """

        # ========== Extract the target series ==========
        y_train = train[self.target_col].reset_index(drop=True)


        # ========== Exogenous variables ==========
        # Exog variables for training obtained from the training set
        # At forecasting time, we assume we do not have access to future exogenous variables,
        # so we build the exogenous variables for the test set based on the training set
        # We build the exogenous variables for the test set by averaging the training set
        # for each day of the year.
        exog_cols = self.experiment_configuration['ARIMAX']['exog_cols']
        exog_train = train[exog_cols].reset_index(drop=True)
        exog_test = self._build_exog_prediction_for_test(train, 
                                                         test.index,
                                                         exog_cols
                                                         )
        
        # ========== Fit the ARIMA model ==========
        model = ARIMA(y_train, order=(self.experiment_configuration['ARIMAX']['p'],  # P: autoregressive order
                                      self.experiment_configuration['ARIMAX']['d'],  # D: differencing order
                                      self.experiment_configuration['ARIMAX']['q']), # Q: moving average order
                                    exog=exog_train
                                    )
                                    
        model_fit = model.fit()

        # ========== Forecast for the length of the test set ==========
        forecast = model_fit.get_forecast(steps=len(test),
                                          exog=exog_test).predicted_mean
        
        # ========== Return the forecasted series with the same index as the test set ==========
        return pd.Series(forecast.values, index = test.index).round().astype(int)


    
    def _arimax_on_residuals(self, train, test):
        """ 
        ARIMA with exog variables (ARIMAX) calculated on the time series after removing seasonal component.
        
        The seasonal component for the training set is calculated using seasonal decomposition.
        The seasonal component for testing is assumed not availabe at forecasting time, 
        So it is predicted using the base model on the training set.

        The seasonal component is added back to the forecast at the end.

        Args:
            train (pd.DataFrame): Training data containing the target column and exogenous variables.
            test (pd.DataFrame): Test data, only used for indexing the predictions and exog variables.
        Returns:
            pd.Series: Predicted values for the test set, indexed by the test set index.
        """
        
        # ========== Extract Seasonal Component ==========
        # Seasonal component for the training set is calculated using seasonal decomposition
        # Seasonal component for the test set is predicted using the base model
        seasonal_train = sm.tsa.seasonal_decompose(train[self.target_col], period=365).seasonal.reset_index(drop=True)
        seasonal_test = self._base_model(train, test).reset_index(drop=True)


        # ========== Remove Seasonal Component from the target variable ==========
        y_train = train[self.target_col].reset_index(drop=True) - seasonal_train

   
        # ========== Exogenous variables ==========
        # Exog variables for training obtained from the training set
        # At forecasting time, we assume we do not have access to future exogenous variables,
        # so we build the exogenous variables for the test set based on the training set
        # We build the exogenous variables for the test set by averaging the training set
        # for each day of the year.
        exog_cols = self.experiment_configuration['ARIMAX_ON_RESIDUALS']['exog_cols']
        exog_train = train[exog_cols].reset_index(drop=True)
        exog_test = self._build_exog_prediction_for_test(train,
                                                         test.index,
                                                         exog_cols)

        # ========== Fit the ARIMA model on the residuals ==========
        model = ARIMA(
            y_train,
            order=(self.experiment_configuration['ARIMAX_ON_RESIDUALS']['p'],
                   self.experiment_configuration['ARIMAX_ON_RESIDUALS']['d'],
                   self.experiment_configuration['ARIMAX_ON_RESIDUALS']['q']),  
            exog=exog_train,  # Include exogenous variables,
        )
        model_fit = model.fit()

        # ========== Forecast for the length of the test set ==========
        forecast = model_fit.get_forecast(steps=len(test), exog=exog_test).predicted_mean

        # ========== Add the seasonal component back to the forecast ==========
        forecast += seasonal_test.values

        # ========== Return the forecasted series with the same index as the test set ==========
        return pd.Series(forecast.values, index=test.index).round().astype(int)
    



    def _sarimax_model(self, train, test):
        """ SARIMAX model with exogenous variables.
        Args:
            train (pd.DataFrame): Training data containing the target column and exogenous variables.
            test (pd.DataFrame): Test data, only used for indexing the predictions and exog variables.
        Returns:
            pd.Series: Predicted values for the test set, indexed by the test set index.
        """
        # ========== Extract the target series ==========
        y_train = train[self.target_col].reset_index(drop=True)

        # ========== Exogenous variables ==========
        # Exog variables for training obtained from the training set
        # At forecasting time, we assume we do not have access to future exogenous variables,
        # so we build the exogenous variables for the test set based on the training set
        # We build the exogenous variables for the test set by averaging the training set
        # for each day of the year.
        exog_cols = self.experiment_configuration['SARIMAX']['exog_cols']
        exog_train = train[exog_cols].reset_index(drop=True)
        exog_test = self._build_exog_prediction_for_test(train, 
                                                         test.index,
                                                         exog_cols
                                                         )


        # ========== Fit the SARIMAX model ==========
        model = SARIMAX(y_train,
                        order=(self.experiment_configuration['SARIMAX']['p'],
                               self.experiment_configuration['SARIMAX']['d'],
                               self.experiment_configuration['SARIMAX']['q'],), # (p,d,q)
                        seasonal_order=(self.experiment_configuration['SARIMAX']['P'],
                                        self.experiment_configuration['SARIMAX']['D'],
                                        self.experiment_configuration['SARIMAX']['Q'],
                                        self.experiment_configuration['SARIMAX']['s'],), # (P,D,Q,s)
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                        exog=exog_train  # Include exogenous variables
                        )

        model_fit = model.fit()

        # ========== Forecast for the length of the test set ==========
        forecast = model_fit.get_forecast(steps=len(test), 
                                          exog=exog_test).predicted_mean

        # ========== Return the forecasted series with the same index as the test set ==========
        return pd.Series(forecast.values, index = test.index).round().astype(int)

    
    
    def _prophet_model(self, train, test):
        """ Prophet model for time series forecasting.
        Args:
            train (pd.DataFrame): Training data containing the target column.
            test (pd.DataFrame): Test data, only used for indexing the predictions.
        Returns:
            pd.Series: Predicted values for the test set, indexed by the test set index.
        """
        # ========== Prepare the DataFrame for Prophet ==========
        df = train[[self.target_col]].reset_index()
        df = df.rename(columns={df.columns[0]: 'ds', self.target_col: 'y'})

        # ========== Initialize and fit the Prophet model ==========
        model = Prophet(yearly_seasonality=self.experiment_configuration['PROPHET']['yearly_seasonality'],
                        daily_seasonality=self.experiment_configuration['PROPHET']['daily_seasonality'],
                        weekly_seasonality=self.experiment_configuration['PROPHET']['weekly_seasonality'],
                        )
        model.fit(df)

        # ========== Prepare the future DataFrame for forecasting ==========
        future = test.reset_index().rename(columns={'index': 'ds'})
        future = future.rename(columns={future.columns[0]: 'ds'})


        # ========== Drop "Redemption Count" and "Sales Count" to be sure there is no data leakage ==========
        future = future.drop(columns=['Redemption Count', 'Sales Count']).reset_index(drop=True)

        logger.debug("Future DataFrame for Prophet:\n%s", future.head())

        # ========== Make predictions using the Prophet model ==========
        forecast = model.predict(future)

        logger.debug("forecast DataFrame for Prophet:\n%s", forecast.head())

        # ========== Return the forecasted series with the same index as the test set ==========
        return pd.Series(forecast['yhat'].values, index=test.index).round().astype(int)


    def _ensemble_model(self, train, test):
        """ Ensemble model that averages predictions from multiple models.
        Args:
            train (pd.DataFrame): Training data containing the target column.
            test (pd.DataFrame): Test data, only used for indexing the predictions.
        Returns:
            pd.Series: Predicted values for the test set, indexed by the test set index.
        """
       
        # ========== Only include models specified in the experiment configuration =========
        preds_dict = {
            model_name: method(train.copy(), test.copy()) 
            for model_name, method in self.modelname2method.items() 
            if model_name in self.experiment_configuration['ENSEMBLE']['included-models']
        }

        # ========== Convert predictions into a DataFrame for easy averaging =========
        preds_df = pd.DataFrame(preds_dict)
        
        # ========== Average across models (axis=1) to get ensemble predictions =========
        ensemble_preds = preds_df.mean(axis=1)

        # ========== Return the ensemble predictions with the same index as the test set ==========
        return pd.Series(ensemble_preds.values, index=test.index).round().astype(int)




    def plot(self, preds, label):
        """ 
        Plot the predictions against the actual values.

        This method only plots values after 2021 since the prediction only start after that date.

        Args:
            preds (pd.Series): Predicted values.
            label (str): Label for the plot legend.
        """
        # plot out the forecasts, truncated to dates after 2021
        mask = self.X.index >= pd.Timestamp('2021-01-01')
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.scatter(self.X.index[mask], self.X[self.target_col][mask], s=0.4, color='grey',
            label=self.target_col)
        ax.plot(preds[preds.index >= pd.Timestamp('2021-01-01')], label=label, color='red')
        plt.legend()
        plt.show()
        plt.close()
