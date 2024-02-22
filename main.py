import numpy as np
from loguru import logger
import time
import random
import pandas_ta as ta
import ccxt
import pandas as pd
from multiprocessing import Process

class CryptoBot:
    def __init__(self) -> None:
        self.m_symbols = []
        self.m_macd_config = {
            'fast': 12,
            'slow': 26,
            'signal': 9
        }
        self.m_history_market_date_config = {
            'exchange': 'binance',
            'period': '1d',
            'limit': 1000,
        }
        self.m_exchange = ccxt.binance({
            'options': {
                'defaultType': 'swap', 
            }
        })

    def get_symbols(self):
        self.m_symbols = pd.DataFrame.from_dict(self.m_exchange.fetch_tickers()).transpose()['symbol'].to_list()
        self.m_symbols = [symbol for symbol in self.m_symbols if symbol.split(':')[1] == 'USDT']

    def __dkx_cross_startegy(self, symbol: str, period: str):
        time.sleep(random.choice([0.2, 0.3, 0.4, 0.5]))
        df = pd.DataFrame(self.m_exchange.fetch_ohlcv(symbol, period, self.m_history_market_date_config['limit']), columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # filter coin which history candles less 100
        if df.shape[0] < 100:
            return
        
        df = df.iloc[::-1]
        df.reset_index(drop=True, inplace=True)

        # calculate dkx
        dkx = np.ones(df.shape[0])
        for idx in range(0, df.shape[0]):
            sum = 0
            count = 0
            if df.iloc[idx: idx + 20].shape[0] == 20:
                for _, row in df.iloc[idx: idx + 20].iterrows():
                    sum += (20 - count) * ((3 * row['close'] + row['low'] + row['open'] + row['high']) / 6)
                    count += 1
            sum /= 210
            dkx[idx] = sum

        df['dkx'] = dkx.tolist()

        # calculate dkx_sma
        dkx_sma = np.ones(df.shape[0])
        for idx in range(0, df.shape[0]):
            sum = 0
            if df.iloc[idx: idx + 10].shape[0] == 10:
                for _, row in df.iloc[idx: idx + 10].iterrows():
                    sum += row['dkx']
            sum /= 10
            dkx_sma[idx] = sum
        df['dkx_sma'] = dkx_sma.tolist()

        df = df.iloc[::-1]
        df.reset_index(drop=True, inplace=True)

        # check dkx crossing up dkx_ma
        if df.iloc[-2]['dkx'] < df.iloc[-2]['dkx_sma'] and df.iloc[-1]['dkx'] > df.iloc[-1]['dkx_sma']:           
            # check macd and signal
            macd_df = ta.macd(df['close'], self.m_macd_config['fast'], self.m_macd_config['slow'], self.m_macd_config['signal'])
            if (macd_df.iloc[-1]['MACD_12_26_9'] > 0 and macd_df.iloc[-1]['MACDs_12_26_9'] > 0) and (macd_df.iloc[-1]['MACD_12_26_9'] > macd_df.iloc[-1]['MACDs_12_26_9']):
                logger.success('{0} Cross Up !'.format(symbol))

    def dkx_cross_strategy(self):
        logger.info('start checking...')
        for symbol in self.m_symbols:
            try:
                process = Process(target=self.__dkx_cross_startegy, args=(symbol, self.m_history_market_date_config['period']))
                process.start()
            except:
                logger.error('check {0} failed'.format(symbol))
        logger.info('check finished')


if __name__ == '__main__':
    coin_bot = CryptoBot()
    coin_bot.get_symbols()
    coin_bot.dkx_cross_strategy()


