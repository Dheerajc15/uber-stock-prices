import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot


def plot_price_series(df, save_path):
    plt.figure(figsize=(12, 5))
    plt.plot(df["Date"], df["price"])
    plt.title("Uber Adjusted Close Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Adjusted Close Price")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_return_distribution(df, save_path):
    r = df["log_return"].dropna()
    plt.figure(figsize=(10, 5))
    sns.histplot(r, kde=True, bins=50)
    plt.title("Distribution of Uber Daily Log Returns")
    plt.xlabel("Log Return")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_return_qq(df, save_path):
    r = df["log_return"].dropna()
    fig = qqplot(r, line="s")
    plt.title("Q-Q Plot of Uber Daily Log Returns")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_rolling_volatility(df, save_path):
    plt.figure(figsize=(12, 5))
    plt.plot(df["Date"], df["vol_21d"], label="21D Annualized Volatility")
    plt.plot(df["Date"], df["vol_63d"], label="63D Annualized Volatility")
    plt.title("Rolling Volatility (Annualized)")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_drawdown(df, save_path):
    plt.figure(figsize=(12, 4))
    plt.plot(df["Date"], df["drawdown"])
    plt.title("Drawdown Curve")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()