import asyncio
import datetime as dt
import multiprocessing
import pickle
import time
import typing
from collections import defaultdict
from multiprocessing import Pool, shared_memory

import aiostream
import numpy as np
import pandas as pd
import sklearn
import sklearn.impute
import sklearn.pipeline
import statsmodels.api as sm
from lineartree import LinearTreeRegressor
from sklearn.linear_model import LinearRegression

import stock_data_dealer.iqfeed_io
import stock_data_dealer.utils
from stock_data_dealer.controllers import MarketDayChecker

from trading_research_core import DailyOhlcRetriever
from trading_research_core.caching_utils import Cache


class WquCapstoneResearchShared:
    tl_data_cache = {}
    MARKET_DAY_CHECKER = MarketDayChecker()

    def __init__(self, cache_dir=None):
        self.top_list_controller = stock_data_dealer.controllers.TopListController()
        self.filter_type = 'FCP'
        cache_dir = cache_dir if cache_dir is not None else '../cache/'
        self.cache = Cache(cache_dir)

    @classmethod
    def filtered_min_volume(cls, symbol, date):
        prev_trade_day = cls.MARKET_DAY_CHECKER.get_trading_day_offset(date, -1)

        try:
            ts = stock_data_dealer.iqfeed_io.IqfeedSmartReader.get_ticks_df_for_symbol_date(symbol, date)
            ts_prev = stock_data_dealer.iqfeed_io.IqfeedSmartReader.get_ticks_df_for_symbol_date(symbol, prev_trade_day)
        except Exception as e:
            print(e)
            return False

        curr_day_pre_market = ts[ts.index.time < dt.time(9, 30)]
        prev_day_after_hours = ts_prev[ts_prev.index.time > dt.time(16, 0)]

        if len(curr_day_pre_market) == 0 or len(prev_day_after_hours) == 0:
            return False

        tot_after_hours_volume = curr_day_pre_market.LastSize.cumsum()[-1] + prev_day_after_hours.LastSize.cumsum()[-1]
        if tot_after_hours_volume <= 1000:
            return False

        return True

    @classmethod
    def main_filter(cls, date, symbols_list):
        black_list = ['ZXZZT']
        filtered_1 = [
            s for s in symbols_list
            if not any(c in s for c in ['+', '.', '-'])
               and s not in black_list
        ]

        filtered_2 = []
        for s in filtered_1:
            if len(s) >= 4 and s[:-1] in filtered_1:
                continue
            filtered_2.append(s)

        filtered_3 = [
            s for s in filtered_2
            if not (len(s) == 5 and s[-1] in ('W', 'S', 'U', 'R', 'Z'))
        ]

        filtered_4 = [s for s in filtered_3 if cls.filtered_min_volume(s, date)]

        return filtered_4

    @classmethod
    def main_filter_worker(cls, date, symbols_list):
        filtered_symbols = cls.main_filter(date, symbols_list)
        return date, filtered_symbols

    async def get_symbols_per_day_unfiltered(
            self,
            start_date=dt.date(2021, 4, 1),
            end_date=dt.date(2021, 9, 30)
    ):

        top_list_symbols_async_gen = \
            self.top_list_controller.get_top_list_symbols_per_day(
                start_date,
                end_date,
                dt.time(9, 25),
                self.filter_type
            )
        top_list_symbols_per_day_raw = await aiostream.stream.list(top_list_symbols_async_gen)

        top_list_symbols_per_day_to_keep = []
        for top_list_sym_date in top_list_symbols_per_day_raw:
            if len(top_list_sym_date[1]) == 0:
                continue
            top_list_symbols_per_day_to_keep.append(top_list_sym_date)

        print(len(top_list_symbols_per_day_to_keep), "days of top list data.")

        cache_filename = 'symbols_per_day_unfiltered.pickle'
        self.cache.pickle_dump(
            top_list_symbols_per_day_to_keep,
            cache_filename
        )

        return top_list_symbols_per_day_to_keep

    def get_symbols_per_day_filtered(self, symbols_per_day_raw):

        with Pool() as pool:
            mapped_results = pool.starmap(self.main_filter_worker, symbols_per_day_raw)
        top_list_symbols_per_day_to_keep_filtered = mapped_results

        cache_filename = 'symbols_per_day_filtered.pickle'
        self.cache.pickle_dump(
            top_list_symbols_per_day_to_keep_filtered,
            cache_filename
        )

        return top_list_symbols_per_day_to_keep_filtered

    @classmethod
    def get_5_min_after_hour_interval_slot(cls, t):
        '''
        returns bin # of offset to t in after hours trading.
        Ex. if t=16:01, will return 0.
        '''
        if dt.time(16, 0) <= t.time() < dt.time(20, 0):
            return (dt.datetime.combine(dt.date.min, t.time()) -
                    dt.datetime.combine(dt.date.min, dt.time(16, 0))).seconds // 60 // 5

        if not (dt.time(4, 0) <= t.time() < dt.time(9, 30)):
            raise Exception(f"{t} not in range between either 16:00-20:00, or 4:00-8:30")

        overnight_5_min_intervals = \
            ((dt.datetime.combine(dt.date.min, dt.time(20, 0)) -
              dt.datetime.combine(dt.date.min, dt.time(16, 0))) \
             .seconds // 60) // 5

        since_premarket_open_5_min_intervals = \
            (dt.datetime.combine(dt.date.min, t.time()) -
             dt.datetime.combine(dt.date.min, dt.time(4, 0))).seconds \
            // 60 // 5

        return overnight_5_min_intervals + since_premarket_open_5_min_intervals

    @classmethod
    def time_to_str(cls, time):
        return time.strftime("%H:%M")

    @classmethod
    def transform_numeric_offset_to_time_since_open(cls, offset):
        return dt.datetime.combine(dt.date(2020, 1, 1), dt.time(9, 30)) + dt.timedelta(minutes=offset * 5)

    @classmethod
    def rolling_outlier_remove(cls, series):
        r = series.rolling(30)  # 30 seconds
        r_mean = r.mean()
        r_outlier_thresh = 2.5 * r.std()
        return series[abs(r_mean - series) < r_outlier_thresh]

    @classmethod
    def get_first_price_cross_after_hours_slot(
            cls, after_hours_total_no_outliers, prev_day_last_reg_trade_price):
        """
        first time when price crosses above 80% range of
        (previous day reg hours last trade price) - (after hours high trade price)
        """

        thresh_price_80_pct_range = prev_day_last_reg_trade_price + \
                                    (after_hours_total_no_outliers.max() - prev_day_last_reg_trade_price) * 0.8
        first_above_thresh = np.argmax(after_hours_total_no_outliers >= thresh_price_80_pct_range)

        first_above_80_pct_after_hours_slot = \
            cls.get_5_min_after_hour_interval_slot(
                after_hours_total_no_outliers.index[first_above_thresh])
        return first_above_80_pct_after_hours_slot

    @classmethod
    def cluster_corr(cls, corr_array, inplace=False):
        """
        Rearranges the correlation matrix, corr_array, so that groups of highly
        correlated variables are next to eachother

        Parameters
        ----------
        corr_array : pandas.DataFrame or numpy.ndarray
            a NxN correlation matrix

        Returns
        -------
        pandas.DataFrame or numpy.ndarray
            a NxN correlation matrix with the columns and rows rearranged

        https://wil.yegelwel.com/cluster-correlation-matrix/
        """
        import scipy.cluster.hierarchy as sch

        pairwise_distances = sch.distance.pdist(corr_array)
        linkage = sch.linkage(pairwise_distances, method='complete')
        cluster_distance_threshold = pairwise_distances.max() / 2
        idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold,
                                            criterion='distance')
        idx = np.argsort(idx_to_cluster_array)

        if not inplace:
            corr_array = corr_array.copy()

        if isinstance(corr_array, pd.DataFrame):
            return corr_array.iloc[idx, :].T.iloc[idx, :]
        return corr_array[idx, :][:, idx]

    @classmethod
    def get_x_for_reg_y(cls, y):
        return \
            np.linspace(
                1,
                len(y),
                len(y)
            ).reshape(-1, 1)

    @classmethod
    def fit_lin_reg_tree(cls, ts):
        x_for_regr = cls.get_x_for_reg_y(ts)

        regr = LinearTreeRegressor(
            base_estimator=LinearRegression(),
            criterion='mse',
            max_depth=10
        )

        regr.fit(x_for_regr, ts)

        return regr

    @classmethod
    def get_count_of_best_fit_lin_reg_models(cls, regr_tree):
        return len(regr_tree.summary(only_leaves=True))

    async def get_matching_tl_data(self, symbol, date):

        tl_params = stock_data_dealer.models.top_list.TiTopListParams(
            dt.datetime.combine(date, dt.time(9, 25)),
            self.filter_type,
            biggest_first=True
        )
        lookup_key = tl_params

        if lookup_key in WquCapstoneResearchShared.tl_data_cache:
            all_tl_data = WquCapstoneResearchShared.tl_data_cache[lookup_key]
        else:
            all_tl_data = await self.top_list_controller.get_top_list_data(tl_params)
            WquCapstoneResearchShared.tl_data_cache[lookup_key] = all_tl_data

        matching_tl_data = next(tl for tl in all_tl_data if tl.symbol == symbol)

        return date, symbol, matching_tl_data

    @classmethod
    def build_endog_for_OLS(cls, ts):
        X = np.array(range(len(ts)))
        # add column of 1s for intercept
        X = sm.add_constant(X)

        return X

    @classmethod
    def fit_OLS(cls, ts):
        if len(ts) < 2:
            print('length of ts is less than 2!')
            return None

        if isinstance(ts, pd.DataFrame) or isinstance(ts, pd.Series):
            ts = ts.values

        X = cls.build_endog_for_OLS(ts)

        return sm.OLS(ts, X).fit()

    @classmethod
    def extract_ols_slope_r2(cls, ols_model):
        slope = ols_model.params[1]
        return slope, ols_model.rsquared

    @classmethod
    def calc_ols_slope_r2(cls, ts):
        model = cls.fit_OLS(ts)
        return cls.extract_ols_slope_r2(model)

    @classmethod
    def calc_momentum_adj_by_r2(cls, ts):
        slope, r2 = cls.calc_ols_slope_r2(ts)
        return slope * r2

    @classmethod
    def get_fundamental_data(cls, tl_data):
        return {
            'stock_price_925': tl_data.c_Price,
            'market_cap': tl_data.c_MCap,
            'short_float_pct': tl_data.c_SFloat,
            'float': tl_data.c_Float,
            'shares_outstanding': tl_data.c_ShOut,
            'cash_$': tl_data.c_Cash,
            'cash_to_debt_ratio': tl_data.c_CashDebt,
            'revenue_$_per_year': tl_data.c_Revenue,
            'dividend_$': tl_data.c_Dividend,
            'interest_income_$': tl_data.c_Interest,
            'current_assets_$': tl_data.c_Assets,
            'enterprise_value_$': tl_data.c_Value
        }

    @classmethod
    def get_market_open_end_intervals(cls, until=dt.time(10, 0), freq='5T'):
        if until < dt.time(9, 35):
            raise Exception(f'until {until} must proceed 9:35')

        #     first_boundary = pd.Timestamp(dt.datetime.combine(dt.date(2000,1,1), dt.time(9,30))) + pd.Timedelta(freq)
        first_boundary = dt.datetime.combine(dt.date(2000, 1, 1), dt.time(9, 35))
        last_boundary = dt.datetime.combine(dt.date(2000, 1, 1), until)
        return pd.date_range(
            first_boundary,
            end=last_boundary,
            freq=freq
        ).time

    @classmethod
    def get_target_reg_fits(cls, symbol, date, end_intervals=None):
        if end_intervals is None:
            end_intervals = cls.get_market_open_end_intervals(freq='1T')

        prev_trade_day = cls.MARKET_DAY_CHECKER.get_trading_day_offset(date, -1)

        ts = stock_data_dealer.iqfeed_io.IqfeedSmartReader.get_ticks_df_for_symbol_date(symbol, date)

        ts_after_open_prices = ts[ts.index.time >= dt.time(9, 30)].Last \
            .resample('1S').last() \
            .interpolate()

        return [
            (boundary, *cls.calc_ols_slope_r2(ts_after_open_prices[ts_after_open_prices.index.time <= boundary]))
            for boundary in end_intervals
        ]

    @classmethod
    def generate_predictor_var_symbol_date(cls, symbol, date: dt.date, tl_data, daily_ohlc_cache):
        try:

            hist_closes_252 = DailyOhlcRetriever.get_prev_daily_closes(
                symbol, date, daily_ohlc_cache, lookback=252
            )

            hist_closes_252_standardized = np.log(hist_closes_252 / hist_closes_252[0])

            hist_closes_10 = DailyOhlcRetriever.get_prev_daily_closes(
                symbol, date, daily_ohlc_cache, lookback=10
            )

            prev_trade_day = cls.MARKET_DAY_CHECKER.get_trading_day_offset(date, -1)

            ts = stock_data_dealer.iqfeed_io.IqfeedSmartReader.get_ticks_df_for_symbol_date(symbol, date)
            ts_prev = stock_data_dealer.iqfeed_io.IqfeedSmartReader.get_ticks_df_for_symbol_date(symbol, prev_trade_day)

            curr_day_pre_market = ts[ts.index.time < dt.time(9, 30)]
            prev_day_after_hours = ts_prev[ts_prev.index.time > dt.time(16, 0)]

            after_hours_prices_raw1 = pd.concat(
                (prev_day_after_hours.Last.resample('1S').last(),
                 curr_day_pre_market.Last.resample('1S').last())
            )
            after_hours_volume_raw1 = pd.concat(
                (prev_day_after_hours.LastSize.resample('1S').sum(),
                 curr_day_pre_market.LastSize.resample('1S').sum())
            )
            after_hours_volume_raw1[after_hours_volume_raw1.isna()] = 0

            thresh = 20
            if len(after_hours_prices_raw1.dropna()) < thresh:
                print(f'Total count of after hours trades for {symbol} on {date.isoformat()} is less than {thresh}.')
                return

            after_hours_total_ts = after_hours_prices_raw1.interpolate()
            after_hours_total_ts_volume = after_hours_volume_raw1

            after_hours_total_no_outliers = \
                cls.rolling_outlier_remove(after_hours_total_ts)

            # first time when price crosses above 80% range of
            # (previous day reg hours last trade price) - (after hours high trade price)
            prev_day_last_reg_trade_price = ts_prev.Last[ts_prev.index.time < dt.time(16, 0)][-1]
            first_above_80_pct_after_hours_slot = \
                cls.get_first_price_cross_after_hours_slot(after_hours_total_no_outliers,
                                                           prev_day_last_reg_trade_price)

            after_hours_ts_reg_tree = cls.fit_lin_reg_tree(after_hours_total_no_outliers)
            num_fitted_reg_models = \
                cls.get_count_of_best_fit_lin_reg_models(after_hours_ts_reg_tree)

            num_overnight_trades = \
                np.sum(prev_day_after_hours.LastSize) + \
                np.sum(curr_day_pre_market.LastSize)

            overnight_dollar_volume = \
                (after_hours_total_ts * after_hours_total_ts_volume).cumsum()[-1]

            max_min_price_pct_change = \
                np.max(after_hours_total_no_outliers) / np.min(after_hours_total_no_outliers) - 1

            if len(after_hours_total_no_outliers) < thresh:
                print(
                    f'Total count of after hours trades with outliers removed for '
                    f'{symbol} on {date.isoformat()} is less than {thresh}.')
                return

            overnight_momentum_adj_by_r2 = \
                cls.calc_momentum_adj_by_r2(after_hours_total_no_outliers)

            max_after_hours_price_velocity = \
                after_hours_total_no_outliers.diff().max()

            last_premarket_price_ratio_to_premarket_high = \
                after_hours_total_no_outliers[-1] / after_hours_total_no_outliers.max()

            last_premarket_price_ratio_to_10_day_high = \
                after_hours_total_no_outliers[-1] / np.max(hist_closes_10)

            last_premarket_price_ratio_to_10_day_low = \
                after_hours_total_no_outliers[-1] / np.min(hist_closes_10)

            ten_day_volume = DailyOhlcRetriever.get_hist_volume(symbol, date, daily_ohlc_cache, lookback=10)
            ten_day_avg_volume = np.mean(ten_day_volume)
            ten_day_price_std_dev = np.std(ten_day_volume)

            overnight_volume_to_10_day_avg_volume = num_overnight_trades / ten_day_avg_volume

            year_daily_closes_reg_slope_and_fit = \
                cls.get_hist_daily_closes_reg_slope_and_fit(
                    hist_closes_252_standardized, min_days=200
                )

            df = pd.DataFrame({
                **cls.get_fundamental_data(tl_data),
                'first_above_80_pct_after_hours_slot': first_above_80_pct_after_hours_slot,
                'num_fitted_reg_models': num_fitted_reg_models,
                'num_overnight_trades': num_overnight_trades,
                'overnight_dollar_volume': overnight_dollar_volume,
                'max_min_price_pct_change': max_min_price_pct_change,
                'overnight_momentum_adj_by_r2': overnight_momentum_adj_by_r2,
                'max_after_hours_price_velocity': max_after_hours_price_velocity,
                'last_premarket_price_ratio_to_premarket_high': last_premarket_price_ratio_to_premarket_high,
                'last_premarket_price_ratio_to_10_day_high': last_premarket_price_ratio_to_10_day_high,
                'last_premarket_price_ratio_to_10_day_low': last_premarket_price_ratio_to_10_day_low,
                '10_day_avg_volume': ten_day_avg_volume,
                '10_day_price_std_dev': ten_day_price_std_dev,
                'overnight_volume_to_10_day_avg_volume': overnight_volume_to_10_day_avg_volume,
                **year_daily_closes_reg_slope_and_fit,
            }, index=cls.build_df_multi_index(date, symbol))

            return df.astype(float)

        except Exception as e:
            print(f'Caught exception for {symbol} {date.isoformat()}: {e}')

    @classmethod
    def load_pickle_from_mmap(cls, name):
        sm = shared_memory.SharedMemory(name=name)
        unpickled = pickle.loads(bytes(sm.buf))
        return unpickled

    @classmethod
    def generate_predictor_var_per_date(
            cls,
            date,
            symbols,
            tl_lookup_mmap_dict_name: str,
            daily_ohlc_cache_mmap_name: str
    ):

        tl_lookup_dict = cls.load_pickle_from_mmap(tl_lookup_mmap_dict_name)
        tl_data = tl_lookup_dict[date]

        daily_ohlc_cache = cls.load_pickle_from_mmap(daily_ohlc_cache_mmap_name)

        all_res = []

        for symbol in symbols:
            tl_data_for_symbol = tl_data[symbol]
            res = cls.generate_predictor_var_symbol_date(symbol, date, tl_data_for_symbol, daily_ohlc_cache)
            if res is None:
                continue
            all_res.append(res)

        return pd.concat(all_res)

    @classmethod
    def get_highest_correlation_pairs(cls, df: pd.DataFrame):
        """
        source: https://stackoverflow.com/a/43073761
        """
        corr_matrix = df.corr().abs()

        corr_pairs = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                      .stack()
                      .sort_values(ascending=False))

        corr_pairs.index.names = ['predict_var_1', 'predict_var_2']
        corr_pairs.name = 'corr'

        return corr_pairs

    @classmethod
    def transform_5_minute_interval_to_time_since_close(cls, interal):
        time_calc = dt.datetime.combine(dt.date(2020, 1, 1), dt.time(16, 0)) + dt.timedelta(minutes=interal * 5)
        time_delta = time_calc - dt.datetime.combine(time_calc.date(), dt.time(20))
        if time_delta.seconds > 0:
            return dt.datetime.combine(dt.date(2020, 1, 1), dt.time(4, 0)) + time_delta
        return time_calc

    @classmethod
    def get_num_minute_size_interval_since_base_time(
            cls, until_time: pd.Timestamp, base_time=dt.time(9, 30), bin_size_minutes=1):
        return (until_time - dt.datetime.combine(until_time.date(), base_time)) \
                   .seconds / (60 * bin_size_minutes)

    @classmethod
    def get_hist_daily_closes_reg_slope_and_fit(cls, hist_daily_closes, min_days=200):

        def build_dict_to_return(reg_slope, reg_fit):
            return {'year_daily_closes_reg_slope': reg_slope,
                    'year_daily_closes_reg_fit': reg_fit}

        if hist_daily_closes is None or len(hist_daily_closes) < min_days:
            return build_dict_to_return(None, None)

        hist_daily_closes_reg = cls.calc_ols_slope_r2(hist_daily_closes)

        if hist_daily_closes_reg is None:
            return build_dict_to_return(None, None)

        return build_dict_to_return(*hist_daily_closes_reg)

    @classmethod
    def build_df_multi_index(cls, date, symbol):
        return pd.MultiIndex.from_tuples([(date, symbol)], names=['date', 'symbol'])

    @classmethod
    def get_returns_for_end_time_boundaries(cls, ts_after_open_prices, last_end_boundary=dt.time(10, 0)):
        end_intervals = \
            cls.get_market_open_end_intervals(freq='5T', until=last_end_boundary)

        returns = []
        for boundary in end_intervals:
            trimmed_ts = ts_after_open_prices[ts_after_open_prices.index.time <= boundary]
            if len(trimmed_ts.unique()) <= 1:
                ret = None
            else:
                ret = trimmed_ts[-1] / trimmed_ts[0] - 1
            returns.append((boundary, ret))

        return {f"{d.strftime('%H:%M')}_ret": ret for d, ret in returns}

    @classmethod
    def get_ols_slope_and_fits_for_end_time_boundaries(cls, ts_after_open_prices, last_end_boundary=dt.time(11, 0)):
        end_intervals = \
            cls.get_market_open_end_intervals(freq='5T', until=last_end_boundary)

        reg_fits = []
        for boundary in end_intervals:
            trimmed_ts = ts_after_open_prices[ts_after_open_prices.index.time <= boundary]
            if len(trimmed_ts.unique()) <= 1:
                slope_r2 = (None, None)
            else:
                slope_r2 = cls.calc_ols_slope_r2(trimmed_ts)
            reg_fits.append((boundary, *slope_r2))

        d1 = {d.strftime('%H:%M'): (slope, fit) for d, slope, fit in reg_fits}
        d2 = [{f'{k}_slope': v[0], f'{k}_fit': v[1]} for k, v in d1.items()]
        d3 = {}
        for d in d2:
            d3.update(d)

        return d3

    @classmethod
    def get_target_var_df_for_sym_date(cls, symbol: str, date: dt.date):

        try:

            ts = stock_data_dealer.iqfeed_io.IqfeedSmartReader.get_ticks_df_for_symbol_date(symbol, date)

            trades_after_open = ts[ts.index.time >= dt.time(9, 30)].Last

            if len(trades_after_open) < 10:
                print(f'len of trades after open for {symbol} {date} is less than 10!')
                return

            ts_after_open_prices: pd.Series = trades_after_open \
                .resample('1S').last() \
                .interpolate()

            thresh_time = dt.time(10, 0)

            ts_after_open_prices_until_thresh = ts_after_open_prices[ts_after_open_prices.index.time <= thresh_time]

            first_after_open_trade = ts_after_open_prices[0]

            prices_until_thresh_time = \
                ts_after_open_prices[ts_after_open_prices.index.time <= thresh_time]

            max_pos_price_pct_change_since_open = \
                prices_until_thresh_time.max() / first_after_open_trade - 1
            max_neg_price_pct_change_since_open = \
                prices_until_thresh_time.min() / first_after_open_trade - 1

            time_max_price_after_open = cls.get_num_minute_size_interval_since_base_time(
                ts_after_open_prices_until_thresh.index[np.argmax(ts_after_open_prices_until_thresh)]
            )
            time_min_price_after_open = cls.get_num_minute_size_interval_since_base_time(
                ts_after_open_prices_until_thresh.index[np.argmin(ts_after_open_prices_until_thresh)]
            )

            returns_for_end_time_boundaries = \
                cls.get_returns_for_end_time_boundaries(ts_after_open_prices)

            ols_slope_and_fits_for_end_time_boundaries = \
                cls.get_ols_slope_and_fits_for_end_time_boundaries(ts_after_open_prices)

            return pd.DataFrame({
                **returns_for_end_time_boundaries,
                'max_pos_price_pct_change_since_open': max_pos_price_pct_change_since_open,
                'max_neg_price_pct_change_since_open': max_neg_price_pct_change_since_open,
                'time_max_price_after_open_until_thresh_time': time_max_price_after_open,
                'time_min_price_after_open_until_thresh_time': time_min_price_after_open,
                **ols_slope_and_fits_for_end_time_boundaries,
            }, index=cls.build_df_multi_index(date, symbol))

        except Exception as e:
            print(f'Caught exception for {symbol} {date.isoformat()}: {e}')

    def generate_target_df(
            self,
            symbols_per_day: typing.List[typing.List[typing.Any]]
    ):

        args = []
        for date, symbols in symbols_per_day:
            for symbol in symbols:
                args.append((symbol, date))

        with Pool() as pool:
            map_res_dfs = pool.starmap(
                self.get_target_var_df_for_sym_date,
                args
            )

        target_vars_df = pd.concat(map_res_dfs)

        cache_filename = 'target_var_df.pickle'
        self.cache.save_df_to_cache(
            target_vars_df,
            cache_filename
        )

        return target_vars_df

    async def build_tl_lookup_dict(self, symbols_per_day_to_keep_filtered):

        matching_tl_per_day_res_list = []
        for d, sym_list in symbols_per_day_to_keep_filtered:
            matching_tl_per_day_res = await asyncio.gather(*[self.get_matching_tl_data(s, d) for s in sym_list])
            matching_tl_per_day_res_list.extend(matching_tl_per_day_res)

        tl_lookup_dict = defaultdict(dict)
        for d, s, ti_data in matching_tl_per_day_res_list:
            tl_lookup_dict[d][s] = ti_data

        cache_filename = 'tl_lookup_dict.pickle'
        self.cache.pickle_dump(
            tl_lookup_dict,
            cache_filename
        )

        return tl_lookup_dict

    @classmethod
    def dump_pickle_to_mmap(cls, obj):
        dump = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        dump_sm = shared_memory.SharedMemory(
            create=True,
            name=None,
            size=len(dump)
        )
        dump_sm.buf[:len(dump)] = dump
        return dump_sm

    def generate_predictor_df(
            self,
            symbols_per_day: typing.List[typing.List[typing.Any]],
            tl_lookup_dict,
            daily_ohlc_cache,
    ):
        tl_lookup_dict_shared_memory = \
            self.dump_pickle_to_mmap(
                tl_lookup_dict,
            )

        daily_ohlc_cache_sm = \
            self.dump_pickle_to_mmap(
                daily_ohlc_cache,
            )

        # use new pool per chunk to prevent RAM overflow
        df_chunks = []
        for i, symbols_per_day_chunk in enumerate(self.chunks(symbols_per_day, 512)):
            print(f'building predictor df for chunk #{i + 1}')
            with Pool(24, maxtasksperchild=4) as pool:
                res = pool.starmap(
                    self.generate_predictor_var_per_date,
                    (
                        (date, symbols, tl_lookup_dict_shared_memory.name, daily_ohlc_cache_sm.name)
                        for date, symbols, in symbols_per_day_chunk
                    )
                )
                df_chunks.extend(res)

        tl_lookup_dict_shared_memory.close()
        tl_lookup_dict_shared_memory.unlink()

        daily_ohlc_cache_sm.close()
        daily_ohlc_cache_sm.unlink()

        pred_vars_all_stocks_df = \
            pd.concat(df_chunks)

        cache_filename = f'pred_vars_all_stocks_df.pickle'
        self.cache.save_df_to_cache(pred_vars_all_stocks_df, cache_filename)

        return pred_vars_all_stocks_df

    @classmethod
    def chunks(cls, iterable, size):
        from itertools import chain, islice
        iterator = iter(iterable)
        for first in iterator:
            yield list(chain([first], islice(iterator, size - 1)))

    class Timer:
        def __enter__(self):
            self.start = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.stop = time.perf_counter()
            print(f"Elapsed time: {self.stop - self.start} seconds.")

    @classmethod
    def get_train_test(cls, df, test_start_date):
        train_df = df[df.index.get_level_values(0) < test_start_date]
        test_df = df[df.index.get_level_values(0) >= test_start_date]
        return train_df, test_df

    @classmethod
    def build_trade_stats_df(cls, trade_stats, trade_date_symbol_indices):
        return pd.DataFrame(
            trade_stats,
            columns=['ret', 'trade_ret', 'predicted_ret', 'mae', 'direction',
                     'pos_size', 'pos_amount', 'ending_position_capital', 'slippage_and_comm'],
            index=pd.MultiIndex.from_tuples(trade_date_symbol_indices)
        )

    @classmethod
    def backtest_strategy(
            cls,
            target_vars_df,
            ret_df,
            exit_time=dt.time(10, 0),
            min_pred_thresh_for_trade=0.01,
            init_capital=100_000,
            slippage_long=0.01,
            slippage_short=0.01,
            avg_locate_fee_per_short_per_share=0.05,  # 5 cents
    ):

        capital = init_capital
        # count = 50
        count = None
        exit = False
        trade_stats = []
        pnl_per_date = [capital]
        trade_date_symbol_indices = []
        positions_per_day = []

        for date in ret_df.index.get_level_values(0).unique():

            ending_position_capitals = []

            symbols_to_trade_curr_date = [
                s for s in ret_df.loc[date].index
                if abs(ret_df.loc[(date, s)]['predicted_ret']) >= min_pred_thresh_for_trade
            ]
            # if less than 4 symbols to trade, set allocation to each symbol = 1/4
            max_allocation_per_symbol = max(len(symbols_to_trade_curr_date), 4)

            for symbol in symbols_to_trade_curr_date:

                try:
                    key = (date, symbol)
                    ret, predicted_ret = ret_df.loc[key]
                    origin_target_variables = target_vars_df.loc[key]

                    ts = stock_data_dealer.iqfeed_io.IqfeedSmartReader.get_ticks_df_for_symbol_date(symbol, date).Last
                    ts_after_market_open_until_exit_time = ts[
                        (ts.index.time > dt.time(9, 30)) & (ts.index.time <= exit_time)]
                    first_trade_price = ts_after_market_open_until_exit_time[0]
                    if first_trade_price is None:
                        continue
                    ts_exit_time_price = ts_after_market_open_until_exit_time[-1]

                    ret = origin_target_variables['10:00_ret']
                    direction = np.sign(predicted_ret)
                    ret = ret * direction
                    mae = cls.get_mae(predicted_ret, origin_target_variables)
                    slippage_and_comm = 0
                    if direction < 0:
                        mae *= (1 + slippage_short)
                        trade_entry_price = first_trade_price * (1 + slippage_short)
                        slippage_and_comm += first_trade_price * slippage_short
                    else:
                        mae *= (1 + slippage_long)
                        trade_entry_price = first_trade_price * (1 + slippage_long)
                        slippage_and_comm += first_trade_price * slippage_long
                    trade_ret = ts_exit_time_price / trade_entry_price - 1

                    # can be negative
                    pos_size = cls.get_position_size(
                        capital,
                        max_allocation_per_symbol,
                        trade_entry_price,
                        direction
                    )

                    pos_amount = abs(pos_size * first_trade_price)
                    capital -= pos_amount
                    ending_position_capital = pos_amount * (1 + trade_ret)
                    if direction < 0:
                        trade_locate_fee = abs(pos_size) * avg_locate_fee_per_short_per_share
                        ending_position_capital -= trade_locate_fee
                        slippage_and_comm += trade_locate_fee
                    ending_position_capitals.append(ending_position_capital)

                    trade_stats.append((ret, trade_ret, predicted_ret, mae, direction, pos_size,
                                        pos_amount, ending_position_capital, slippage_and_comm))

                    trade_date_symbol_indices.append(key)

                    if count is not None:
                        count -= 1
                        if count < 0:
                            exit = True
                            break

                except Exception as e:
                    print(f'Caught exception while processing {symbol} {date.isoformat()}:', e)

            capital += sum(ending_position_capitals)
            positions_per_day.append(len(symbols_to_trade_curr_date))
            pnl_per_date.append(capital)

            if exit:
                break

        print('min_pred_thresh_for_trade:', min_pred_thresh_for_trade)
        print('starting capital:', init_capital)
        print('ending capital:', capital)

        if len(trade_date_symbol_indices) == 0:
            print('No trades executed')
            return

        trade_stats_df = cls.build_trade_stats_df(trade_stats, trade_date_symbol_indices)

        strat_df = pd.DataFrame({
            'init_capital': init_capital,
            'ending_capital': capital,
            'net_return': capital / init_capital - 1,
            'avg_positions_per_day': np.mean(positions_per_day),
            'slippage_long': slippage_long,
            'slippage_short': slippage_short,
            'avg_locate_fee_per_short_per_share': avg_locate_fee_per_short_per_share,
            'sharpe': cls.calc_sharpe(pnl_per_date),
            'max_drawdown': cls.calc_drawdown_series(pnl_per_date).min(),
            'slippage_and_comm': sum(trade_stats_df.slippage_and_comm),
            'exit_time': exit_time
        }, index=[min_pred_thresh_for_trade]).T

        return strat_df, trade_stats_df, pnl_per_date

    @classmethod
    def get_position_size(cls, capital, max_allocation, price, direction):
        max_capital_per_position = capital / max_allocation
        position_size = int(max_capital_per_position // price * direction)
        return position_size

    @classmethod
    def calc_sharpe(cls, pnl_per_date):
        return (((pnl_per_date[-1] / pnl_per_date[0] - 1) * 1) \
                / pd.Series(pnl_per_date).pct_change().std()) / np.sqrt(len(pnl_per_date))

    @classmethod
    def calc_drawdown_series(cls, pnl):
        pnl_standardized = np.array(pnl) / pnl[0]
        return pnl_standardized - np.maximum.accumulate(pnl_standardized)

    @classmethod
    def get_mae(cls, pred, target_df):
        if pred > 0:
            return -1 * target_df.max_neg_price_pct_change_since_open
        elif pred < 0:
            return target_df.max_pos_price_pct_change_since_open
        else:
            return 0

    @classmethod
    def build_return_df(cls, actual: pd.Series, predicted):
        return pd.DataFrame({
            'ret': actual,
            'predicted_ret': predicted
        }, index=actual.index)

    @classmethod
    def backtest_model(cls, df, ret_time: dt.time, model_per_ret_time, X):
        col_name = f'{ret_time.strftime("%H:%M")}_ret'
        model = model_per_ret_time[col_name]
        ret_df = cls.build_return_df(
            df[col_name],
            model.predict(X)
        )

        return cls.backtest_across_min_predicted_thresh(df, ret_df, ret_time)

    @classmethod
    def backtest_across_min_predicted_thresh(cls, df, ret_df, exit_time):

        # exit at 9:35 with different param min thresh
        args = [
            (df, ret_df, exit_time, min_trade_thresh)
            for min_trade_thresh in np.arange(0, 0.21, 0.01)
        ]

        with Pool(32, maxtasksperchild=4) as pool:
            backtest_results_per_min_trade_thresh_935 = pool.starmap(
                cls.backtest_strategy,
                args
            )

        return backtest_results_per_min_trade_thresh_935

    @classmethod
    def get_test_start_date_objs(cls, df):
        dates_under_study = \
            df.index.get_level_values(0).unique()

        idx_of_test_start = int(len(dates_under_study) * 0.70)
        test_start_date = dates_under_study[idx_of_test_start]

        return dates_under_study, idx_of_test_start, test_start_date

    @classmethod
    def train_pred_models_per_returns(cls, X, train_df, return_columns, reg_estimator):
        model_per_ret_time = {}

        est = sklearn.base.clone(reg_estimator)

        for ret_column in return_columns:
            X1, y1 = \
                cls.drop_na_from_Xy_pair(X, train_df[ret_column])
            est.fit(X1, y1)
            model_per_ret_time[ret_column] = est

        return model_per_ret_time

    @classmethod
    def drop_na_from_Xy_pair(cls, X, y):
        y_filter_idx = ~y.isna()
        return X[y_filter_idx], y[y_filter_idx]

    @classmethod
    def plot_underwater(cls, pnl):
        import pylab as plt

        drawdown_series = cls.calc_drawdown_series(pnl)
        mean_drawdown = np.mean(drawdown_series)
        plt.plot(drawdown_series, label='drawdown')
        plt.fill_between(np.arange(len(drawdown_series)), drawdown_series, np.zeros_like(drawdown_series), alpha=0.5)
        plt.axhline(mean_drawdown, label=f'mean drawdown {mean_drawdown.round(3)}', color='grey')
        plt.legend()


async def main():
    multiprocessing.set_start_method('spawn')

    # cache_dir = '/home/trade/Dev/wqu_capstone/cache'
    # wqu_capstone_research_obj = WquCapstoneResearchShared(
    #     cache_dir=cache_dir
    # )

    # symbols_per_day_unfiltered = await wqu_capstone_research_obj.get_symbols_per_day_unfiltered()
    # symbols_per_day_filtered = \
    #     wqu_capstone_research_obj.get_symbols_per_day_filtered(
    #     symbols_per_day_unfiltered
    #     )
    # symbols_per_day_filtered = wqu_capstone_research_obj.cache.pickle_load('symbols_per_day_filtered.pickle')

    # tl_lookup_dict = await wqu_capstone_research_obj.build_tl_lookup_dict(symbols_per_day_filtered)
    # tl_lookup_dict = wqu_capstone_research_obj.cache.pickle_load('tl_lookup_dict.pickle')

    # daily_ohlc_retriever = DailyOhlcRetriever(cache=wqu_capstone_research_obj.cache)
    # daily_ohlc_cache = await daily_ohlc_retriever.build_daily_ohlc_cache(
    #     symbols_per_day_filtered
    # )

    # wqu_capstone_research_obj.generate_predictor_df(
    #     symbols_per_day_filtered,
    #     tl_lookup_dict,
    #     daily_ohlc_cache
    # )

    # wqu_capstone_research_obj.generate_target_df(
    #     symbols_per_day_filtered
    # )


if __name__ == '__main__':
    stock_data_dealer.utils.run_async_func(main())
