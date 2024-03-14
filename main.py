import os
import ccxt
import time
import warnings
import argparse
import numpy as np
import pandas as pd
import pandas_ta as ta
from loguru import logger
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait


class CoinBot:
    def __init__(self) -> None:
        self.__init_arg_parser()
        self.m_symbols = []
        self.m_macd_config = {"fast": 12, "slow": 26, "signal": 9}
        self.m_history_market_date_config = {}
        self.m_exchange = None

    def __init_arg_parser(self) -> None:
        self.arg_parser = argparse.ArgumentParser()
        self.arg_parser.add_argument(
            "-e", "--exchange", nargs="?", type=str, default="binance"
        )
        self.arg_parser.add_argument(
            "-p", "--period", nargs="?", type=str, default="1d"
        )
        self.arg_parser.add_argument("-l", "--limit", nargs="?", type=int, default=1000)

    def get_symbols(self):
        self.m_symbols = (
            pd.DataFrame.from_dict(self.m_exchange.fetch_tickers())
            .transpose()["symbol"]
            .to_list()
        )
        self.m_symbols = [
            symbol for symbol in self.m_symbols if symbol.split(":")[1] == "USDT"
        ]

    def init_exchange(self):
        args = self.arg_parser.parse_args()
        self.m_history_market_date_config = {
            "exchange": args.exchange,
            "period": args.period,
            "limit": args.limit,
        }
        self.m_exchange = ccxt.binance(
            {
                "options": {
                    "defaultType": "swap",
                }
            }
        )

    def __clear_log():
        with open("result.log", "r+") as f:
            f.seek(0)
            f.truncate()

    def dkx_cross_strategy(self):
        CoinBot.__clear_log()
        logger.add("result.log", level="SUCCESS")
        logger.info(
            "start checking with config: {0}...".format(
                self.m_history_market_date_config
            )
        )
        max_workers = min(32, os.cpu_count() + 4)
        split_symbols = [
            self.m_symbols[i : i + int(len(self.m_symbols) / max_workers)]
            for i in range(
                0, len(self.m_symbols), int(len(self.m_symbols) / max_workers)
            )
        ]
        with ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="coin_bot_"
        ) as pool:
            all_task = [
                pool.submit(
                    CoinBot.__dkx_cross_strategy,
                    self,
                    split_symbols[i],
                )
                for i in range(0, len(split_symbols))
            ]
            wait(all_task, return_when=ALL_COMPLETED)
            logger.info("check finished")

    def __dkx_cross_strategy(self, symbols: list):
        for _, symbol in enumerate(symbols):
            try:
                time.sleep(0.5)
                logger.info("checking {0}".format(symbol))

                if self.m_history_market_date_config["period"] == "5d":
                    df = pd.DataFrame(
                        self.m_exchange.fetch_ohlcv(
                            symbol,
                            "1d",
                            self.m_history_market_date_config["limit"],
                        ),
                        columns=["timestamp", "open", "high", "low", "close", "volume"],
                    )
                else:
                    df = pd.DataFrame(
                        self.m_exchange.fetch_ohlcv(
                            symbol,
                            self.m_history_market_date_config["period"],
                            self.m_history_market_date_config["limit"],
                        ),
                        columns=["timestamp", "open", "high", "low", "close", "volume"],
                    )

                # filter coin which history candles less 100
                if df.shape[0] < 100:
                    continue

                if self.m_history_market_date_config["period"] == "5d":
                    blocks = -1
                    if df.shape[0] % 5 == 0:
                        blocks = df.shape[0] / 5
                    else:
                        blocks = df.shape[0] / 5 + 1
                    split_dfs = np.array_split(df.copy(), blocks)
                    df = pd.DataFrame(columns=["open", "close", "high", "low"])
                    for split_df in split_dfs:
                        df.loc[len(df)] = [
                            split_df[:1]["open"].to_list()[0],
                            split_df[-1:]["close"].to_list()[0],
                            split_df["high"].max(),
                            split_df["low"].min(),
                        ]

                df = df.iloc[::-1]
                df.reset_index(drop=True, inplace=True)

                # calculate dkx
                dkx = np.ones(df.shape[0])
                for idx in range(0, df.shape[0]):
                    sum = 0
                    count = 0
                    if df.iloc[idx : idx + 20].shape[0] == 20:
                        for _, row in df.iloc[idx : idx + 20].iterrows():
                            sum += (20 - count) * (
                                (
                                    3 * row["close"]
                                    + row["low"]
                                    + row["open"]
                                    + row["high"]
                                )
                                / 6
                            )
                            count += 1
                    sum /= 210
                    dkx[idx] = sum

                df["dkx"] = dkx.tolist()

                # calculate dkx_sma
                dkx_sma = np.ones(df.shape[0])
                for idx in range(0, df.shape[0]):
                    sum = 0
                    if df.iloc[idx : idx + 10].shape[0] == 10:
                        for _, row in df.iloc[idx : idx + 10].iterrows():
                            sum += row["dkx"]
                    sum /= 10
                    dkx_sma[idx] = sum
                df["dkx_sma"] = dkx_sma.tolist()

                df = df.iloc[::-1]
                df.reset_index(drop=True, inplace=True)

                # check dkx crossing up dkx_sma
                if (
                    df.iloc[-2]["dkx"] < df.iloc[-2]["dkx_sma"]
                    and df.iloc[-1]["dkx"] > df.iloc[-1]["dkx_sma"]
                ):
                    # check macd and signal
                    macd_df = ta.macd(
                        df["close"],
                        self.m_macd_config["fast"],
                        self.m_macd_config["slow"],
                        self.m_macd_config["signal"],
                    )
                    if (
                        macd_df.iloc[-1]["MACD_12_26_9"] > 0
                        and macd_df.iloc[-1]["MACDs_12_26_9"] > 0
                    ) and (
                        macd_df.iloc[-1]["MACD_12_26_9"]
                        > macd_df.iloc[-1]["MACDs_12_26_9"]
                    ):
                        logger.success("{0} Cross Up !".format(symbol))
            except:
                # usually network error
                logger.error("check {0} failed".format(symbol))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    coin_bot = CoinBot()
    coin_bot.init_exchange()
    coin_bot.get_symbols()
    coin_bot.dkx_cross_strategy()
