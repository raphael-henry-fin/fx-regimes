
#=============================================================================
# Automated financial analysis system
# Asset analysed: DEXUSEU (USD/EUR exchange rate)
#
# Required libraries:
# pip install pandas numpy matplotlib scipy statsmodels xlsxwriter jinja2
#=============================================================================

import base64
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from scipy import stats         # Shapiro-Wilk normality test
import statsmodels.api as sm    # OLS regression

import xlsxwriter
from jinja2 import Template

# =============================================================================
# FILE PATHS
# =============================================================================
# Paths are resolved relative to the script itself, so the folder can be moved
# or opened on another machine without editing a single line.

# The try/except covers Jupyter, where __file__ is not always defined.
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

# The JSON holds the analysed series, the CSV the context series including the
# SP500 benchmark.
CSV_PATH  = os.path.join(BASE_DIR, "market_data_context_DEXUSEU.csv")
JSON_PATH = os.path.join(BASE_DIR, "market_data_DEXUSEU.json")

OUTPUT_DIR = BASE_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Checked upfront so a missing file gives an actionable message instead of a
# raw pandas error in the middle of processing.
for _path in (CSV_PATH, JSON_PATH):
    if not os.path.exists(_path):
        raise FileNotFoundError(
            f"\n\nData file not found:\n  {_path}\n\n"
            f"Place the following two files in the same folder as this script:\n"
            f"  - market_data_context_DEXUSEU.csv\n"
            f"  - market_data_DEXUSEU.json\n"
            f"\nExpected folder: {BASE_DIR}\n"
        )


# =============================================================================
# PART 1: DATA LOADING AND CLEANING
# =============================================================================

def load_and_clean_data(csv_path, json_path):
    """
    Load the CSV and JSON sources, clean them and merge them on the date.

    Returns a DataFrame sorted by date, holding DEXUSEU, SP500 and NASDAQCOM.

    Cleaning happens before merging: merging first would align empty rows
    against empty rows and inflate the table for nothing.
    """

    print("=" * 60)
    print("ÉTAPE 1 : Chargement et nettoyage des données")
    print("=" * 60)

    df_csv = pd.read_csv(csv_path)
    print(f"CSV chargé : {len(df_csv)} lignes, {len(df_csv.columns)} colonnes")

    # As text, "2021-10-01" < "2021-9-01" : a real date type is needed for the
    # comparisons and the sort below to be correct.
    df_csv["Date"] = pd.to_datetime(df_csv["Date"])

    with open(json_path, "r") as f:
        json_data = json.load(f)

    df_json = pd.DataFrame(json_data)
    df_json["Date"] = pd.to_datetime(df_json["Date"])
    print(f"JSON chargé : {len(df_json)} lignes")

    # The FRED file starts in 1990 but DEXUSEU is only populated from 2021, and
    # weekends carry no quotation. These rows are dropped rather than filled:
    # inventing a price on a closed day would distort returns and volatility.
    # subset= limits the test to that column, so a gap in SP500 costs nothing.
    before = len(df_json)
    df_json = df_json.dropna(subset=["DEXUSEU"])
    after = len(df_json)
    print(f"JSON : {before - after} lignes nulles supprimées → {after} lignes restantes")

    # how="left" keeps every DEXUSEU trading day even when the CSV has no match
    # that day. how="inner" would silently drop days of the analysed series.
    df = pd.merge(df_json, df_csv[["Date", "SP500", "NASDAQCOM"]], on="Date", how="left")

    # pct_change() and cummax() both read rows in order, so the chronological
    # sort underpins every metric computed later.
    df = df.sort_values("Date").reset_index(drop=True)

    # A high count on SP500 here would signal a calendar mismatch between the
    # two sources, to be fixed before running the regression.
    print("\nValeurs manquantes par colonne après fusion :")
    print(df.isnull().sum())

    print(f"\nDonnées finales : {len(df)} lignes")
    print(f"Période : du {df['Date'].min().date()} au {df['Date'].max().date()}")

    return df


# =============================================================================
# PART 2: OBJECT-ORIENTED ARCHITECTURE (Classes)
# =============================================================================
# A class keeps the data and the returns in memory once, so every metric reads
# them through self instead of recomputing them. Part 3 then reuses the same
# class on each market regime without duplicating a single formula.
#
#   PortfolioAnalyzer      -> base class, the financial calculations
#   PeriodAwareAnalyzer    -> subclass, adds the split by market regime
# =============================================================================

class PortfolioAnalyzer:
    """Core financial calculations over a price series."""

    def __init__(self, df, name="Portfolio"):
        # .copy() keeps this analyser from sharing a table with the caller: an
        # in-place change would otherwise contaminate the main program's data
        # and every other analyser built on it.
        self.df = df.copy()
        self.name = name

        # Computed once at construction: three of the four metrics depend on it.
        self.returns = self._calculate_returns()

    def _calculate_returns(self):
        """Daily returns: (today's price - yesterday's price) / yesterday's price."""
        # The first row has no previous day, so pct_change() yields NaN there.
        # It must go: mean() and std() would otherwise return NaN.
        returns = self.df["DEXUSEU"].pct_change().dropna()
        return returns

    def annualized_return(self):
        """Average daily return projected over a full trading year."""
        # 252 = trading sessions in a year, the market convention. Geometric
        # compounding would be more exact but the gap stays small on daily FX.
        daily_mean = self.returns.mean()
        annualized = daily_mean * 252
        return annualized

    def annualized_volatility(self):
        """Standard deviation of daily returns, scaled to one year."""
        daily_std = self.returns.std()

        # Square root of 252, not 252: variance scales with time, standard
        # deviation with its square root. Scaling by 252 would inflate the
        # figure roughly sixteenfold.
        annualized = daily_std * np.sqrt(252)
        return annualized

    def sharpe_ratio(self, risk_free_rate=0.02):
        """
        Sharpe ratio: (return - risk-free rate) / volatility.

        Measures how much each unit of risk taken pays. Above 1 is considered
        good, above 2 excellent.
        """
        excess_return = self.annualized_return() - risk_free_rate
        sharpe = excess_return / self.annualized_volatility()
        return sharpe

    def max_drawdown(self):
        """
        Largest loss suffered from a historical peak of the series.

        Complements volatility: two series with identical volatility can show
        very different drawdowns depending on whether declines cluster together
        or alternate with rallies.
        """
        prices = self.df["DEXUSEU"]

        # Running maximum: the highest level reached so far. It never falls
        # back, which is what makes the drop from the peak measurable.
        rolling_max = prices.cummax()
        drawdown = (prices - rolling_max) / rolling_max

        # Zero at the peak, negative everywhere else, so the worst loss is the
        # minimum — min() here, not max().
        return drawdown.min()

    def print_summary(self):
        """Print the four metrics to the console."""
        print(f"\n--- Résumé : {self.name} ---")
        print(f"  Rendement annualisé  : {self.annualized_return():.2%}")
        print(f"  Volatilité annualisée: {self.annualized_volatility():.2%}")
        print(f"  Ratio de Sharpe      : {self.sharpe_ratio():.4f}")
        print(f"  Maximum Drawdown     : {self.max_drawdown():.2%}")


class PeriodAwareAnalyzer(PortfolioAnalyzer):
    """
    Adds the split into market regimes on top of PortfolioAnalyzer.

    Regimes read off the FRED chart of the series:
      - Bearish  2021-06-01 -> 2022-09-26: the dollar strengthens sharply, the
        euro falls to about 0.95
      - Bullish  2024-09-01 -> 2025-12-31: the euro rebounds from about 1.05
        to 1.17
    """

    # Keyed by name rather than position, so a third regime can be added
    # without touching any of the code below.
    PERIODS = {
        "bearish": ("2021-06-01", "2022-09-26"),   # strong dollar
        "bullish": ("2024-09-01", "2025-12-31"),   # strong euro
    }

    def __init__(self, df):
        super().__init__(df, name="PeriodAwareAnalyzer")

    def get_period_data(self, period_name):
        """Return only the rows falling inside the requested date window."""
        # Raised here rather than as a bare KeyError several frames deeper.
        if period_name not in self.PERIODS:
            raise ValueError(f"Période inconnue : {period_name}. Choisis 'bearish' ou 'bullish'.")

        start, end = self.PERIODS[period_name]

        # & and not "and": & compares the two series element by element. The
        # parentheses are mandatory, & binding tighter than >=.
        mask = (self.df["Date"] >= start) & (self.df["Date"] <= end)

        # .copy() detaches the extract, otherwise pandas returns a view and
        # raises SettingWithCopyWarning on any later change.
        return self.df[mask].copy()

    def analyze_periods(self):
        """
        Compute the four metrics separately on each market regime.

        Returns {"bearish": {"return": ..., "sharpe": ...}, "bullish": {...}}
        """
        results = {}

        for period_name in self.PERIODS:
            period_df = self.get_period_data(period_name)

            # A fresh analyser on the period extract alone. It recomputes its
            # own returns on those rows, so each regime is treated as an
            # independent sample rather than a slice of the global metrics.
            sub_analyzer = PortfolioAnalyzer(period_df, name=period_name)

            results[period_name] = {
                "return"     : sub_analyzer.annualized_return(),
                "volatility" : sub_analyzer.annualized_volatility(),
                "sharpe"     : sub_analyzer.sharpe_ratio(),
                "drawdown"   : sub_analyzer.max_drawdown(),
                # Needed to judge reliability: a Sharpe over 30 days is not
                # worth a Sharpe over 300.
                "n_points"   : len(period_df),
            }

            sub_analyzer.print_summary()

        return results


# =============================================================================
# PART 3: STATISTICAL ANALYSIS
# =============================================================================

def run_statistical_analysis(df, period_analyzer):
    """
    Runs an OLS regression of DEXUSEU on SP500, then a Shapiro-Wilk test on the
    residuals to check whether the regression's significance tests can be
    trusted.

    Returns (model, residuals).
    """

    print("\n" + "=" * 60)
    print("ÉTAPE 3 : Analyse statistique")
    print("=" * 60)

    # A regression needs complete pairs: a DEXUSEU observation with no matching
    # SP500 value is unusable.
    reg_df = df[["DEXUSEU", "SP500"]].dropna()

    # Returns are regressed, not price levels. Two series each drifting upward
    # look correlated when all they share is a trend — the spurious regression
    # trap. Daily changes remove that common trend.
    y = reg_df["DEXUSEU"].pct_change().dropna()   # dependent variable
    x = reg_df["SP500"].pct_change().dropna()      # explanatory variable

    # Both series were cleaned independently, so their dates can differ.
    y, x = y.align(x, join="inner")

    # Adds the column of ones producing the intercept alpha. Without it the
    # line is forced through zero, assuming a flat SP500 implies a flat DEXUSEU.
    X = sm.add_constant(x)

    model = sm.OLS(y, X).fit()
    print("\n--- Résultats de la régression OLS (DEXUSEU ~ SP500) ---")
    print(model.summary())

    # The p-values printed above assume normally distributed residuals. If that
    # fails, the coefficients stay correctly estimated but the significance
    # tests around them become unreliable.
    #   p-value > 0.05  -> normality plausible
    #   p-value <= 0.05 -> residuals are not normal
    residuals = model.resid

    # Shapiro-Wilk becomes unreliable beyond 5000 observations, hence the cap.
    # random_state fixes the draw so the figure is reproducible across runs.
    sample = residuals.sample(min(5000, len(residuals)), random_state=42)
    stat_sw, p_sw = stats.shapiro(sample)

    print(f"\n--- Test de Shapiro-Wilk sur les résidus ---")
    print(f"  Statistique W : {stat_sw:.6f}")
    print(f"  p-value       : {p_sw:.6f}")
    if p_sw > 0.05:
        print("  → Les résidus semblent normalement distribués (p > 0.05)")
    else:
        print("  → Les résidus NE suivent PAS une distribution normale (p ≤ 0.05)")

    return model, residuals


# =============================================================================
# PART 4: CHARTS
# =============================================================================

def create_charts(df, period_analyzer, output_dir):
    """
    Build and save the two charts, returning a dict of their file paths for the
    report builder to embed.
    """

    print("\n" + "=" * 60)
    print("ÉTAPE 4 : Création des graphiques")
    print("=" * 60)

    charts = {}

    # --- Chart 1: price series with both regimes shaded ----------------------
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["Date"], df["DEXUSEU"], color="steelblue", linewidth=1, label="DEXUSEU")

    # Boundaries read back from the class, so the shaded areas can never drift
    # out of sync with the periods actually computed.
    p1_start, p1_end = pd.to_datetime(period_analyzer.PERIODS["bearish"])

    # axvspan spans the full height of the chart between two x positions.
    # alpha keeps it transparent enough to read the curve underneath.
    ax.axvspan(p1_start, p1_end, color="red", alpha=0.15, label="Période 1 : Baissière")

    p2_start, p2_end = pd.to_datetime(period_analyzer.PERIODS["bullish"])
    ax.axvspan(p2_start, p2_end, color="green", alpha=0.15, label="Période 2 : Haussière")

    ax.set_title("Taux de change USD/EUR (DEXUSEU) avec périodes analysées", fontsize=14)
    ax.set_xlabel("Date")
    # DEXUSEU quotes dollars per euro, not the reverse — worth stating.
    ax.set_ylabel("USD pour 1 EUR")
    ax.legend()

    # Year only, otherwise full dates overlap along the axis.
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)

    # Without tight_layout the tilted labels get clipped on save.
    plt.tight_layout()

    path1 = os.path.join(output_dir, "chart_dexuseu_periods.png")
    fig.savefig(path1, dpi=150)
    plt.close(fig)

    charts["price_chart"] = path1
    print(f"  Graphique 1 sauvegardé : {path1}")

    # --- Chart 2: distribution of daily returns ------------------------------
    # Returns come from the class rather than a second pct_change(), so the
    # chart cannot disagree with the tables.
    analyzer = PortfolioAnalyzer(df)
    returns = analyzer.returns

    fig2, ax2 = plt.subplots(figsize=(10, 5))

    # 60 bins: fine enough to show the thickness of the tails without
    # descending into the noise of an over-sliced histogram.
    ax2.hist(returns, bins=60, color="steelblue", edgecolor="white", alpha=0.8)
    ax2.set_title("Distribution des rendements journaliers (DEXUSEU)", fontsize=13)
    ax2.set_xlabel("Rendement journalier")
    ax2.set_ylabel("Fréquence")
    plt.tight_layout()

    path2 = os.path.join(output_dir, "chart_returns_dist.png")
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)
    charts["returns_dist"] = path2
    print(f"  Graphique 2 sauvegardé : {path2}")

    return charts


# =============================================================================
# PART 5: REPORT GENERATION (ReportBuilder class)
# =============================================================================

class ReportBuilder:
    """
    Class responsible for generating every output file automatically:
    - the multi-sheet Excel workbook
    - the HTML report
    - the justification text files
    """

    def __init__(self, df, analysis_results, model, residuals, charts, output_dir):
        self.df              = df                 # cleaned and merged data
        self.results         = analysis_results   # metrics for both regimes
        self.model           = model              # fitted OLS model
        self.residuals       = residuals          # residuals, for the normality test
        self.charts          = charts             # paths of the PNGs to embed
        self.output_dir      = output_dir         # where to write the outputs

        # analysis_results only holds the two regimes, so the global metrics are
        # recomputed here.
        self.analyzer        = PortfolioAnalyzer(df, "Global")

    def save_justification_files(self):
        """
        Write the three justification text files.

        Must run before build_html_report(), which reads them back to embed
        their content in the document.
        """

        # --- Justification of the chosen periods ---
        period_text = """JUSTIFICATION DES PÉRIODES CHOISIES
=====================================

Source visuelle : Graphique FRED - U.S. Dollars to Euro Spot Exchange Rate (DEXUSEU)
URL : https://fred.stlouisfed.org/series/DEXUSEU

PÉRIODE 1 : Bearish (baissière pour l'euro)
  Dates    : 01/06/2021 → 26/09/2022
  Contexte : Durant cette période, la Réserve Fédérale américaine (Fed) a amorcé
             un cycle de hausse des taux d'intérêt agressif pour combattre l'inflation.
             Des taux américains plus élevés rendent le dollar plus attractif.
             En parallèle, la crise énergétique en Europe (guerre en Ukraine) a
             fragilisé l'euro. Le taux USD/EUR est tombé sous la parité (< 1.00)
             le 26 septembre 2022, un niveau historiquement bas.
  Raison de sélection : Forte tendance directionnelle baissière visible clairement
                        sur le graphique, idéale pour tester une stratégie défensive.

PÉRIODE 2 : Bullish (haussière pour l'euro)
  Dates    : 01/09/2024 → 31/12/2025
  Contexte : La Fed a commencé à baisser ses taux à partir de septembre 2024,
             réduisant l'attrait du dollar. L'économie européenne s'est stabilisée.
             Le taux EUR/USD est remonté de ~1.05 à ~1.17.
  Raison de sélection : Tendance haussière claire et soutenue, représentant
                        un retournement structurel par rapport à la période 1.
"""
        path = os.path.join(self.output_dir, "period_justification.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(period_text)
        print(f"  Fichier sauvegardé : {path}")

        # --- Justification of the benchmark ---
        benchmark_text = """JUSTIFICATION DU BENCHMARK CHOISI
===================================

Benchmark retenu : S&P 500 (SP500)

Raison du choix :
  1. Le S&P 500 est l'indice boursier américain le plus représentatif du marché
     actions mondial. Il est universellement utilisé comme benchmark de référence.

  2. Le taux de change USD/EUR est directement influencé par la santé économique
     américaine, que le S&P 500 reflète bien. Quand les actions américaines montent,
     la demande de dollars augmente, ce qui tend à apprécier le dollar.

  3. Contrairement à un benchmark purement monétaire (ex: DXY), le S&P 500 permet
     de mesurer la relation entre les marchés actions et le forex, ce qui est
     pertinent pour les stratégies multi-actifs.

Résultats de la régression OLS :
  - R² : indique quelle fraction de la variance de DEXUSEU est expliquée par SP500
  - Beta : sensibilité du taux de change aux mouvements du S&P 500
"""
        path2 = os.path.join(self.output_dir, "benchmark_justification.txt")
        with open(path2, "w", encoding="utf-8") as f:
            f.write(benchmark_text)
        print(f"  Fichier sauvegardé : {path2}")

        # --- Critique of the AI-generated code ---
        ai_critique_text = """CRITIQUE DU CODE IA - RATIO DE SHARPE
========================================

PROMPT UTILISÉ :
  "Write a Python function to calculate the annualized Sharpe ratio given
   a series of daily returns."

RÉPONSE TYPIQUE DE L'IA (ChatGPT) :
--------------------------------------
def sharpe_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
--------------------------------------

TROIS FAILLES SUBSTANTIELLES IDENTIFIÉES :
-------------------------------------------

FAILLE 1 - Taux sans risque mal appliqué
  Problème IA : L'IA soustrait le taux sans risque JOURNALIER directement
                des rendements journaliers. Mais risk_free_rate=0.0 par défaut,
                ce qui revient à ne pas l'utiliser du tout.
  Notre solution : On annualise d'abord le rendement moyen (mean * 252), puis
                   on soustrait le taux annualisé (ex: 2%). Cela respecte la
                   formule officielle du ratio de Sharpe.

FAILLE 2 - Écart-type biaisé (population vs échantillon)
  Problème IA : np.std() sans argument utilise ddof=0 (diviseur N),
                soit l'écart-type de la POPULATION.
                Pour un échantillon financier, on doit utiliser ddof=1 (diviseur N-1).
  Notre solution : self.returns.std() utilise ddof=1 par défaut dans pandas,
                   ce qui est statistiquement correct pour un échantillon.

FAILLE 3 - Absence de gestion des valeurs manquantes
  Problème IA : La fonction ne vérifie pas si la série contient des NaN.
                Un seul NaN rend np.mean() et np.std() inutilisables (résultat NaN).
  Notre solution : On appelle .dropna() avant tout calcul (voir _calculate_returns()),
                   ce qui garantit des résultats fiables quelles que soient les données.
"""
        path3 = os.path.join(self.output_dir, "ai_critique.txt")
        with open(path3, "w", encoding="utf-8") as f:
            f.write(ai_critique_text)
        print(f"  Fichier sauvegardé : {path3}")

    def build_excel(self):
        """
        Build the multi-sheet Excel workbook with xlsxwriter.

        xlsxwriter writes but does not read: it cannot reopen an existing file
        to extend it, so the workbook is rebuilt from scratch on every run.
        """
        path = os.path.join(self.output_dir, "financial_report.xlsx")
        workbook = xlsxwriter.Workbook(path)

        # --- Cell styles -------------------------------------------------------
        # Palette: black, grey, white. Green and red are reserved for the sign
        # of the figures, the only place where colour carries information.

        fmt_title = workbook.add_format({
            "bold": True, "font_size": 13,
            "bg_color": "#000000", "font_color": "#FFFFFF",
            "align": "left", "valign": "vcenter",
        })

        # No fill, just a rule underneath, as in a printed table.
        fmt_header = workbook.add_format({
            "bold": True, "font_color": "#000000",
            "bottom": 2, "bottom_color": "#000000",
            "align": "left", "valign": "bottom",
        })

        # num_format controls the display only: "0.00%" shows 0.0327 as 3.27%
        # while the cell still holds 0.0327 for Excel formulas.
        fmt_number  = workbook.add_format({"num_format": "0.0000", "border": 1, "border_color": "#D9D9D9"})
        fmt_percent = workbook.add_format({"num_format": "0.00%",  "border": 1, "border_color": "#D9D9D9"})
        fmt_date    = workbook.add_format({"num_format": "yyyy-mm-dd", "border": 1, "border_color": "#D9D9D9"})
        fmt_text    = workbook.add_format({"border": 1, "border_color": "#D9D9D9"})

        # Coloured variants, applied only to quantities carrying a directional
        # meaning: return, Sharpe, drawdown.
        fmt_pct_pos = workbook.add_format({"num_format": "0.00%", "border": 1, "border_color": "#D9D9D9", "font_color": "#1A6B3C"})
        fmt_pct_neg = workbook.add_format({"num_format": "0.00%", "border": 1, "border_color": "#D9D9D9", "font_color": "#A32020"})
        fmt_num_pos = workbook.add_format({"num_format": "0.0000", "border": 1, "border_color": "#D9D9D9", "font_color": "#1A6B3C"})
        fmt_num_neg = workbook.add_format({"num_format": "0.0000", "border": 1, "border_color": "#D9D9D9", "font_color": "#A32020"})

        def signed(value, is_percent=True):
            """Pick the green or red format according to the sign of the value."""
            if value > 0:
                return fmt_pct_pos if is_percent else fmt_num_pos
            if value < 0:
                return fmt_pct_neg if is_percent else fmt_num_neg
            return fmt_percent if is_percent else fmt_number

        # ===== SHEET 1: Global summary ========================================
        ws1 = workbook.add_worksheet("Résumé Global")
        ws1.set_column("A:A", 30)
        ws1.set_column("B:B", 20)
        ws1.merge_range("A1:B1", "Analyse DEXUSEU (USD/EUR) — Résumé global", fmt_title)

        # Row 2 left empty to give the table some air below the title.
        ws1.write("A3", "Métrique", fmt_header)
        ws1.write("B3", "Valeur",   fmt_header)

        glob_ret    = self.analyzer.annualized_return()
        glob_vol    = self.analyzer.annualized_volatility()
        glob_sharpe = self.analyzer.sharpe_ratio()
        glob_dd     = self.analyzer.max_drawdown()

        # Volatility and the day count stay black: high volatility is neither
        # good nor bad in itself, so colouring it would mislead.
        metrics = [
            ("Rendement annualisé",   glob_ret,     signed(glob_ret)),
            ("Volatilité annualisée", glob_vol,     fmt_percent),
            ("Ratio de Sharpe",       glob_sharpe,  signed(glob_sharpe, is_percent=False)),
            ("Maximum Drawdown",      glob_dd,      signed(glob_dd)),
            ("Nombre de jours",       len(self.df), fmt_number),
        ]

        for i, (label, value, fmt) in enumerate(metrics):
            # xlsxwriter counts from 0, and the offset of 3 skips the title, the
            # blank row and the header.
            row = 3 + i
            ws1.write(row, 0, label, fmt_text)
            ws1.write(row, 1, value, fmt)

        # ===== SHEET 2: Comparison of the two regimes =========================
        ws2 = workbook.add_worksheet("Analyse par Période")
        ws2.set_column("A:E", 22)
        ws2.merge_range("A1:E1", "Métriques par régime de marché", fmt_title)

        headers = ["Métrique", "Période 1 (Baissière)", "Période 2 (Haussière)", "Dates P1", "Dates P2"]
        for col, h in enumerate(headers):
            ws2.write(2, col, h, fmt_header)

        # .get() rather than [] so a missing regime yields an empty dict and the
        # report is still produced.
        p1 = self.results.get("bearish", {})
        p2 = self.results.get("bullish", {})

        period_metrics = [
            ("Rendement annualisé",   p1.get("return", 0),     p2.get("return", 0),     True),
            ("Volatilité annualisée", p1.get("volatility", 0), p2.get("volatility", 0), None),
            ("Ratio de Sharpe",       p1.get("sharpe", 0),     p2.get("sharpe", 0),     False),
            ("Maximum Drawdown",      p1.get("drawdown", 0),   p2.get("drawdown", 0),   True),
            ("Nombre de points",      p1.get("n_points", 0),   p2.get("n_points", 0),   None),
        ]

        for i, (label, v1, v2, colored) in enumerate(period_metrics):
            row = 3 + i
            ws2.write(row, 0, label, fmt_text)

            # True = coloured percentage, False = coloured number,
            # None = neutral quantity left in black.
            if colored is None:
                f1 = f2 = fmt_percent if i == 1 else fmt_number
            else:
                f1, f2 = signed(v1, colored), signed(v2, colored)

            ws2.write(row, 1, v1, f1)
            ws2.write(row, 2, v2, f2)

        # Written once only: the boundaries hold for the whole column.
        ws2.write(3, 3, "2021-06-01 → 2022-09-26", fmt_text)
        ws2.write(3, 4, "2024-09-01 → 2025-12-31", fmt_text)

        # ===== SHEET 3: Raw data ==============================================
        # Makes the report auditable: a reader can recompute the metrics from
        # the displayed data.
        ws3 = workbook.add_worksheet("Données Brutes")
        ws3.set_column("A:B", 18)
        ws3.merge_range("A1:B1", "Données journalières DEXUSEU", fmt_title)
        ws3.write(1, 0, "Date",    fmt_header)
        ws3.write(1, 1, "DEXUSEU", fmt_header)

        for i, row_data in self.df[["Date", "DEXUSEU"]].iterrows():
            # write_datetime/write_number rather than write(): Excel then stores
            # genuine dates and numbers, on which formulas and sorting work.
            # to_pydatetime() converts the pandas timestamp into the standard
            # datetime xlsxwriter expects.
            ws3.write_datetime(i + 2, 0, row_data["Date"].to_pydatetime(), fmt_date)
            ws3.write_number(i + 2, 1, row_data["DEXUSEU"], fmt_number)

        # ===== SHEET 4: Regression output =====================================
        ws4 = workbook.add_worksheet("Régression OLS")
        ws4.set_column("A:B", 34)
        ws4.merge_range("A1:B1", "Régression OLS — DEXUSEU sur SP500", fmt_title)
        ws4.write(2, 0, "Paramètre", fmt_header)
        ws4.write(2, 1, "Valeur",    fmt_header)

        # .get() guards against the explanatory column carrying another name.
        # The .iloc[-1] fallback takes the last coefficient, always the
        # benchmark's, since add_constant prepends the constant.
        ols_metrics = [
            ("R² (coefficient de détermination)", self.model.rsquared),
            ("Alpha (constante)",                 self.model.params.get("const", float("nan"))),
            ("Beta (SP500)",                      self.model.params.get("SP500", self.model.params.iloc[-1])),
            ("p-value (Beta)",                    self.model.pvalues.iloc[-1]),
            ("AIC",                               self.model.aic),
            ("BIC",                               self.model.bic),
        ]

        # All in black: a negative beta is not bad news, it is information about
        # the direction of the relationship.
        for i, (label, value) in enumerate(ols_metrics):
            ws4.write(3 + i, 0, label, fmt_text)
            ws4.write(3 + i, 1, value, fmt_number)

        # close() is what actually writes the file to disk; everything above
        # lived in memory only.
        workbook.close()
        print(f"  Excel sauvegardé : {path}")
        return path

    def build_html_report(self):
        """
        Build the HTML report with Jinja2.

        The layout (HTML with {{ }} placeholders) is kept apart from the data
        (Python values), the two being combined at the end by render(). Changing
        the look of the report therefore never touches the calculation code.
        """

        def read_file(filename):
            """Read back one of the justification files produced earlier."""
            filepath = os.path.join(self.output_dir, filename)
            # A missing ancillary file inserts a placeholder rather than failing
            # the whole report.
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    return f.read()
            except FileNotFoundError:
                return "(Fichier non trouvé)"

        # Each {{ name }} below must match an argument of render() exactly — a
        # typo silently leaves the slot empty, with no error raised.
        #
        # Design: black, grey, white, serif typography, hairline rules. Green
        # and red appear only on the sign of the figures, through the .pos and
        # .neg classes computed on the Python side.
        html_template = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>DEXUSEU — Note d'analyse</title>
<style>
    @page { size: A4; margin: 20mm 18mm; }

    html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }

    body {
        font-family: Georgia, "Times New Roman", serif;
        font-size: 10.5pt;
        line-height: 1.55;
        color: #111;
        background: #fff;
        max-width: 172mm;
        margin: 24px auto;
        padding: 0 8px;
    }

    /* ---- Masthead ---------------------------------------------------- */
    .kicker {
        font-family: Helvetica, Arial, sans-serif;
        font-size: 7.5pt;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #666;
        margin-bottom: 6px;
    }
    h1 {
        font-size: 19pt;
        font-weight: normal;
        letter-spacing: -0.01em;
        margin: 0 0 4px 0;
    }
    .subtitle { font-size: 11pt; color: #444; font-style: italic; margin: 0 0 14px 0; }
    .rule-heavy { border: 0; border-top: 2px solid #111; margin: 0 0 10px 0; }
    .rule-light { border: 0; border-top: 1px solid #ccc; margin: 22px 0 18px 0; }

    /* ---- Identification block ----------------------------------------- */
    .ident { width: 100%; border-collapse: collapse; margin-bottom: 4px; }
    .ident td { padding: 2px 0; vertical-align: top; font-size: 9.5pt; }
    .ident td.k {
        width: 34mm; color: #666;
        font-family: Helvetica, Arial, sans-serif; font-size: 8.5pt;
        text-transform: uppercase; letter-spacing: 0.06em;
        padding-top: 4px;
    }

    /* ---- Sections ----------------------------------------------------- */
    h2 {
        font-family: Helvetica, Arial, sans-serif;
        font-size: 11pt;
        font-weight: bold;
        margin: 26px 0 10px 0;
        padding-bottom: 5px;
        border-bottom: 1px solid #111;
    }
    h2 .num { color: #888; font-weight: normal; margin-right: 10px; }
    h3 {
        font-family: Helvetica, Arial, sans-serif;
        font-size: 9pt;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #333;
        margin: 20px 0 8px 0;
    }

    p { margin: 0 0 10px 0; text-align: justify; }

    /* ---- Tables -------------------------------------------------------- */
    table.data {
        width: 100%;
        border-collapse: collapse;
        margin: 12px 0 6px 0;
        font-size: 9.5pt;
    }
    table.data th {
        font-family: Helvetica, Arial, sans-serif;
        font-size: 8pt;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #333;
        text-align: right;
        padding: 5px 8px 6px 8px;
        border-bottom: 1.5px solid #111;
        vertical-align: bottom;
    }
    table.data th:first-child { text-align: left; }
    table.data th .sub {
        display: block;
        font-weight: normal;
        text-transform: none;
        letter-spacing: 0;
        color: #777;
        font-size: 7.5pt;
        padding-top: 2px;
    }
    table.data td {
        padding: 5px 8px;
        border-bottom: 1px solid #e2e2e2;
        text-align: right;
        font-variant-numeric: tabular-nums;
        font-feature-settings: "tnum";
    }
    table.data td:first-child { text-align: left; }
    table.data tr:last-child td { border-bottom: 1px solid #111; }
    table.data td.lbl { color: #222; }
    table.data td.note { font-size: 8.5pt; color: #777; font-style: italic; }

    /* ---- Key figures ---------------------------------------------------- */
    table.kpi { width: 100%; border-collapse: collapse; margin: 14px 0 4px 0; }
    table.kpi td {
        width: 25%;
        padding: 10px 12px 10px 0;
        border-top: 1px solid #111;
        border-bottom: 1px solid #ccc;
        vertical-align: top;
    }
    .kpi-lbl {
        font-family: Helvetica, Arial, sans-serif;
        font-size: 7.5pt;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #666;
        margin-bottom: 5px;
    }
    .kpi-val {
        font-size: 16pt;
        font-variant-numeric: tabular-nums;
        font-feature-settings: "tnum";
        letter-spacing: -0.01em;
    }

    /* ---- Sign: the only two colours in the document --------------------- */
    .pos { color: #1a6b3c; }
    .neg { color: #a32020; }

    /* ---- Text blocks pulled in from the .txt files ---------------------- */
    .verbatim {
        white-space: pre-wrap;
        font-family: Georgia, "Times New Roman", serif;
        font-size: 9.5pt;
        line-height: 1.5;
        color: #222;
        border-left: 2px solid #ccc;
        padding: 2px 0 2px 14px;
        margin: 10px 0 14px 0;
    }

    /* ---- Figures -------------------------------------------------------- */
    figure { margin: 16px 0 20px 0; }
    figure img { width: 100%; display: block; border: 1px solid #ddd; }
    figcaption {
        font-family: Helvetica, Arial, sans-serif;
        font-size: 8pt;
        color: #666;
        padding-top: 6px;
        border-top: 1px solid #e2e2e2;
        margin-top: 6px;
    }
    figcaption b { color: #333; }

    /* ---- Misc ----------------------------------------------------------- */
    .lede { font-size: 11pt; color: #222; }
    .footnote {
        font-size: 8.5pt;
        color: #666;
        line-height: 1.45;
        margin-top: 6px;
    }
    .signoff {
        margin-top: 30px;
        padding-top: 8px;
        border-top: 1px solid #ccc;
        font-size: 8.5pt;
        color: #666;
        font-family: Helvetica, Arial, sans-serif;
    }
    .page-break { page-break-after: always; }
</style>
</head>
<body>

<div class="kicker">Note d'analyse &middot; Marché des changes</div>
<h1>Dollar américain contre euro — DEXUSEU</h1>
<p class="subtitle">Performance, régimes de marché et sensibilité au S&amp;P 500</p>
<hr class="rule-heavy">

<table class="ident">
  <tr><td class="k">Instrument</td><td>U.S. Dollars to Euro Spot Exchange Rate (DEXUSEU)</td></tr>
  <tr><td class="k">Source</td><td>Federal Reserve Economic Data (FRED), série quotidienne</td></tr>
  <tr><td class="k">Échantillon</td><td>{{ date_start }} au {{ date_end }} — {{ n_points }} séances</td></tr>
  <tr><td class="k">Benchmark</td><td>S&amp;P 500 (SP500)</td></tr>
  <tr><td class="k">Convention</td><td>Rendements arithmétiques quotidiens, annualisés sur 252 séances ;
      taux sans risque retenu 2,00&nbsp;%</td></tr>
</table>

<hr class="rule-light">

<p class="lede">Cette note évalue le comportement du taux de change USD/EUR sur l'ensemble
de la période disponible, puis isole deux régimes de marché contrastés afin de vérifier si
les caractéristiques de risque et de rendement de la paire sont stables dans le temps. Une
régression sur le S&amp;P 500 mesure ensuite dans quelle mesure le marché actions américain
explique les variations de la paire.</p>

<!-- ================================================================= -->
<h2><span class="num">1</span>Profil global de la période</h2>

<table class="kpi">
  <tr>
    <td>
      <div class="kpi-lbl">Rendement annualisé</div>
      <div class="kpi-val {{ cls_return }}">{{ annualized_return }}</div>
    </td>
    <td>
      <div class="kpi-lbl">Volatilité annualisée</div>
      <div class="kpi-val">{{ annualized_vol }}</div>
    </td>
    <td>
      <div class="kpi-lbl">Ratio de Sharpe</div>
      <div class="kpi-val {{ cls_sharpe }}">{{ sharpe }}</div>
    </td>
    <td>
      <div class="kpi-lbl">Drawdown maximal</div>
      <div class="kpi-val {{ cls_dd }}">{{ max_dd }}</div>
    </td>
  </tr>
</table>

<p class="footnote">Le drawdown maximal correspond au repli le plus profond enregistré
depuis un sommet historique de la série, sans reconstitution intermédiaire. Le ratio de
Sharpe est calculé net du taux sans risque annoncé ci-dessus.</p>

<figure>
  <img src="{{ chart_price }}" alt="Taux de change USD/EUR sur la période analysée">
  <figcaption><b>Figure 1.</b> Taux de change USD/EUR, séance par séance. Les deux plages
  grisées correspondent aux régimes isolés en section 2.</figcaption>
</figure>

<figure>
  <img src="{{ chart_returns }}" alt="Distribution des rendements quotidiens">
  <figcaption><b>Figure 2.</b> Distribution des rendements quotidiens. L'épaisseur des
  queues conditionne la validité des hypothèses testées en section 3.</figcaption>
</figure>

<div class="page-break"></div>

<!-- ================================================================= -->
<h2><span class="num">2</span>Comparaison des deux régimes de marché</h2>

<h3>Méthode de découpage</h3>
<div class="verbatim">{{ period_justification }}</div>

<h3>Métriques comparées</h3>
<table class="data">
  <thead>
    <tr>
      <th>Métrique</th>
      <th>Régime baissier<span class="sub">2021-06-01 → 2022-09-26</span></th>
      <th>Régime haussier<span class="sub">2024-09-01 → 2025-12-31</span></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="lbl">Rendement annualisé</td>
      <td class="{{ cls_p1_return }}">{{ p1_return }}</td>
      <td class="{{ cls_p2_return }}">{{ p2_return }}</td>
    </tr>
    <tr>
      <td class="lbl">Volatilité annualisée</td>
      <td>{{ p1_vol }}</td>
      <td>{{ p2_vol }}</td>
    </tr>
    <tr>
      <td class="lbl">Ratio de Sharpe</td>
      <td class="{{ cls_p1_sharpe }}">{{ p1_sharpe }}</td>
      <td class="{{ cls_p2_sharpe }}">{{ p2_sharpe }}</td>
    </tr>
    <tr>
      <td class="lbl">Drawdown maximal</td>
      <td class="{{ cls_p1_dd }}">{{ p1_drawdown }}</td>
      <td class="{{ cls_p2_dd }}">{{ p2_drawdown }}</td>
    </tr>
    <tr>
      <td class="lbl">Séances retenues</td>
      <td>{{ p1_npoints }}</td>
      <td>{{ p2_npoints }}</td>
    </tr>
  </tbody>
</table>

<p class="footnote">Chaque régime est traité comme un échantillon indépendant : les
métriques sont recalculées sur les seules séances de la fenêtre, et non extraites de la
série complète. Les deux fenêtres n'ayant pas la même longueur, la comparaison porte sur
des grandeurs annualisées et non sur des performances cumulées.</p>

<div class="page-break"></div>

<!-- ================================================================= -->
<h2><span class="num">3</span>Sensibilité au benchmark et validation statistique</h2>

<h3>Choix du benchmark</h3>
<div class="verbatim">{{ benchmark_justification }}</div>

<h3>Régression par moindres carrés — DEXUSEU sur S&amp;P 500</h3>
<table class="data">
  <thead>
    <tr><th>Paramètre</th><th>Estimation</th><th>Lecture</th></tr>
  </thead>
  <tbody>
    <tr>
      <td class="lbl">R&sup2;</td><td>{{ ols_r2 }}</td>
      <td class="note">Part de la variance expliquée par le benchmark</td>
    </tr>
    <tr>
      <td class="lbl">Alpha (constante)</td><td>{{ ols_alpha }}</td>
      <td class="note">Dérive résiduelle, benchmark neutralisé</td>
    </tr>
    <tr>
      <td class="lbl">Beta (S&amp;P 500)</td><td>{{ ols_beta }}</td>
      <td class="note">Réaction de la paire à un mouvement du benchmark</td>
    </tr>
    <tr>
      <td class="lbl">p-value du beta</td><td>{{ ols_pval }}</td>
      <td class="note">Significatif au seuil de 5&nbsp;% si inférieure à 0,05</td>
    </tr>
    <tr>
      <td class="lbl">Critère AIC</td><td>{{ ols_aic }}</td>
      <td class="note">Base de comparaison avec une spécification alternative</td>
    </tr>
  </tbody>
</table>

<h3>Normalité des résidus — test de Shapiro-Wilk</h3>
<p>{{ shapiro_result }}</p>

<p class="footnote">Le test est appliqué à un tirage aléatoire de 5&nbsp;000 résidus au
maximum, limite propre à la statistique de Shapiro-Wilk. Un rejet de la normalité
n'invalide pas les estimations ponctuelles du modèle, mais fragilise les intervalles de
confiance et les tests de significativité qui en découlent.</p>

<div class="page-break"></div>

<!-- ================================================================= -->
<h2><span class="num">4</span>Revue critique du code produit par assistance IA</h2>

<div class="verbatim">{{ ai_critique }}</div>

<div class="signoff">
  Note générée automatiquement à partir de la série DEXUSEU (FRED).
  Échantillon arrêté au {{ date_end }}. Document à usage pédagogique&nbsp;;
  ne constitue ni un conseil en investissement, ni une recommandation.
</div>

</body>
</html>
"""

        # --- Preparing the values to inject ------------------------------------
        p1 = self.results.get("bearish", {})
        p2 = self.results.get("bullish", {})

        # Re-run with the same seed as part 3, so the report cannot contradict
        # what was printed to the console.
        sample = self.residuals.sample(min(5000, len(self.residuals)), random_state=42)
        stat_sw, p_sw = stats.shapiro(sample)

        # Written in Python so the template stays free of conditional logic.
        shapiro_text = (
            f"Statistique W = {stat_sw:.6f}, p-value = {p_sw:.6f}. "
            + ("Les résidus semblent normalement distribués (p > 0.05)." if p_sw > 0.05
               else "Les résidus ne suivent pas une distribution normale (p ≤ 0.05).")
        )

        # Decided in Python rather than CSS so the colour rests on the actual
        # numeric value, before any text formatting.
        def sign_class(value):
            if value > 0:
                return "pos"
            if value < 0:
                return "neg"
            return ""

        def chart_src(key):
            """
            Return the chart as a base64 data URI rather than a file path.

            An <img src="chart.png"> would tie the report to its neighbouring
            files: moved or emailed on its own, the HTML would show empty
            frames. Embedding costs about a third in size and makes the report
            a single self-contained file.
            """
            full = self.charts.get(key, "")
            if not full or not os.path.exists(full):
                return ""

            # "rb": an image is not text, reading it in text mode would corrupt
            # the bytes.
            with open(full, "rb") as img:
                encoded = base64.b64encode(img.read()).decode("ascii")

            return f"data:image/png;base64,{encoded}"

        # Stored because each value is used twice: formatted for display, and
        # raw to determine its colour.
        glob_return = self.analyzer.annualized_return()
        glob_sharpe = self.analyzer.sharpe_ratio()
        glob_dd     = self.analyzer.max_drawdown()

        # Access by name, falling back to position — alpha first since
        # add_constant prepends the constant, beta last.
        try:
            alpha_val = self.model.params["const"]
            beta_val  = self.model.params.get("SP500", self.model.params.iloc[-1])
            pval_val  = self.model.pvalues.get("SP500", self.model.pvalues.iloc[-1])
        except Exception:
            alpha_val = self.model.params.iloc[0]
            beta_val  = self.model.params.iloc[1] if len(self.model.params) > 1 else 0
            pval_val  = self.model.pvalues.iloc[-1]

        # Formatting happens here rather than in the HTML, which keeps the
        # template free of presentation logic. :.6f on the p-value is needed to
        # read a figure that close to zero.
        template = Template(html_template)
        html_content = template.render(
            date_start            = str(self.df["Date"].min().date()),
            date_end              = str(self.df["Date"].max().date()),
            n_points              = len(self.df),
            annualized_return     = f"{glob_return:.2%}",
            annualized_vol        = f"{self.analyzer.annualized_volatility():.2%}",
            sharpe                = f"{glob_sharpe:.4f}",
            max_dd                = f"{glob_dd:.2%}",
            cls_return            = sign_class(glob_return),
            cls_sharpe            = sign_class(glob_sharpe),
            cls_dd                = sign_class(glob_dd),
            chart_price           = chart_src("price_chart"),
            chart_returns         = chart_src("returns_dist"),
            period_justification  = read_file("period_justification.txt"),
            benchmark_justification = read_file("benchmark_justification.txt"),
            ai_critique           = read_file("ai_critique.txt"),
            p1_return             = f"{p1.get('return', 0):.2%}",
            p1_vol                = f"{p1.get('volatility', 0):.2%}",
            p1_sharpe             = f"{p1.get('sharpe', 0):.4f}",
            p1_drawdown           = f"{p1.get('drawdown', 0):.2%}",
            p1_npoints            = p1.get("n_points", 0),
            cls_p1_return         = sign_class(p1.get("return", 0)),
            cls_p1_sharpe         = sign_class(p1.get("sharpe", 0)),
            cls_p1_dd             = sign_class(p1.get("drawdown", 0)),
            p2_return             = f"{p2.get('return', 0):.2%}",
            p2_vol                = f"{p2.get('volatility', 0):.2%}",
            p2_sharpe             = f"{p2.get('sharpe', 0):.4f}",
            p2_drawdown           = f"{p2.get('drawdown', 0):.2%}",
            p2_npoints            = p2.get("n_points", 0),
            cls_p2_return         = sign_class(p2.get("return", 0)),
            cls_p2_sharpe         = sign_class(p2.get("sharpe", 0)),
            cls_p2_dd             = sign_class(p2.get("drawdown", 0)),
            ols_r2                = f"{self.model.rsquared:.6f}",
            ols_alpha             = f"{alpha_val:.8f}",
            ols_beta              = f"{beta_val:.6f}",
            ols_pval              = f"{pval_val:.6f}",
            ols_aic               = f"{self.model.aic:.4f}",
            shapiro_result        = shapiro_text,
        )

        path = os.path.join(self.output_dir, "financial_report.html")

        # encoding="utf-8" must match the template's <meta charset>, otherwise
        # Windows writes in its local encoding and accents render as garbage.
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"  Rapport HTML sauvegardé : {path}")
        return path


# =============================================================================
# MAIN PROGRAM
# =============================================================================
# if __name__ == "__main__" means "only when this file is run directly", so
# importing the module elsewhere to reuse a class does not trigger the full
# analysis.
# =============================================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("  SYSTÈME D'ANALYSE FINANCIÈRE AUTOMATISÉ — DEXUSEU")
    print("=" * 60)

    # --- STEP 1: data ------------------------------------------------------
    df = load_and_clean_data(CSV_PATH, JSON_PATH)

    # --- STEP 2: metrics, through the classes ------------------------------
    print("\n" + "=" * 60)
    print("ÉTAPE 2 : Architecture orientée objet")
    print("=" * 60)

    # Full period: the reference against which the regimes are compared.
    global_analyzer = PortfolioAnalyzer(df, "Global (toutes périodes)")
    global_analyzer.print_summary()

    period_analyzer = PeriodAwareAnalyzer(df)
    period_results  = period_analyzer.analyze_periods()

    # --- STEP 3: statistics ------------------------------------------------
    model, residuals = run_statistical_analysis(df, period_analyzer)

    # --- STEP 4: charts ----------------------------------------------------
    # period_analyzer is passed in so the shaded areas read the very same
    # boundaries as those actually analysed.
    charts = create_charts(df, period_analyzer, OUTPUT_DIR)

    # --- STEP 5: reports ----------------------------------------------------
    print("\n" + "=" * 60)
    print("ÉTAPE 5 : Génération des rapports")
    print("=" * 60)

    # Named arguments: with six parameters, a silent positional swap would
    # happen sooner or later.
    builder = ReportBuilder(
        df               = df,
        analysis_results = period_results,
        model            = model,
        residuals        = residuals,
        charts           = charts,
        output_dir       = OUTPUT_DIR,
    )

    # Order matters: the text files must exist before the HTML report tries to
    # read them back to embed their content.
    builder.save_justification_files()
    builder.build_excel()
    builder.build_html_report()

    print("\n" + "=" * 60)
    print("  TERMINÉ ! Tous les fichiers sont dans :")
    print(f"  {OUTPUT_DIR}")
    print("=" * 60)
