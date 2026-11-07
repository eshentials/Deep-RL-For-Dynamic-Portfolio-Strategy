"""Risk-free parsing and accrual."""
import numpy as np
import pandas as pd
import pytest

from data.rf import daily_accrual, parse_rbi


def test_parses_export_with_title_rows(tmp_path):
    rows = [["Table 218 : Auctions of 91-Day Government of India Treasury Bills", "", ""],
            ["(Amount in Rs crore)", "", ""],
            ["Date of Auction", "Notified Amount", "Implicit Yield at Cut-off Price (per cent)"],
            [1, 2, 12],                                  # RBI's column-number row
            ["2024-25", "", ""],                         # fiscal-year label row
            ["03-04-2024", 9000, 6.8945],
            ["10-04-2024", 9000, 6.8621],
            ["17-04-2024", 9000, 6.8803]]
    p = tmp_path / "t218.xlsx"
    pd.DataFrame(rows).to_excel(p, header=False, index=False)
    s = parse_rbi(p)
    assert len(s) == 3
    assert s.index[0] == pd.Timestamp("2024-04-03")      # day-first dates
    assert s.iloc[1] == pytest.approx(0.068621)           # percent -> decimal


def test_prefers_the_91_day_column_in_a_wide_export(tmp_path):
    df = pd.DataFrame({"Date": ["2020-01-01", "2020-01-08"],
                       "182 Day T-Bill Cut-off Yield": [5.3, 5.31],
                       "91 Day T-Bill Cut-off Yield": [5.1, 5.12]})
    p = tmp_path / "dbie.csv"
    df.to_csv(p, index=False)
    s = parse_rbi(p)
    assert s.iloc[0] == pytest.approx(0.051)


def test_accrual_compounds_to_the_annual_yield():
    sessions = pd.bdate_range("2021-01-01", "2021-12-31")
    y = pd.Series(0.065, index=sessions[:1])
    acc = daily_accrual(y, sessions)["accrual"].to_numpy()
    days = (sessions[-1] - sessions[0]).days
    assert np.prod(1 + acc[1:]) == pytest.approx(1.065 ** (days / 365), rel=1e-12)


def test_yield_carried_forward_only():
    sessions = pd.bdate_range("2022-01-03", periods=10)
    y = pd.Series([0.04, 0.05], index=[sessions[0], sessions[5]])
    out = daily_accrual(y, sessions)["yield"]
    assert (out.iloc[:5] == 0.04).all() and (out.iloc[5:] == 0.05).all()


def test_never_takes_the_cut_off_price_as_the_yield(tmp_path):
    rows = [["Date of Auction", "Cut-off Price (Rs)", "Implicit Yield at Cut-off Price (percent)"],
            ["2025-07-02", 98.68, 5.3699], ["2025-07-09", 98.67, 5.3872]]
    p = tmp_path / "t.xlsx"
    pd.DataFrame(rows).to_excel(p, header=False, index=False)
    s = parse_rbi(p)
    assert s.iloc[0] == pytest.approx(0.053699)


def test_coverage_check_rejects_short_and_stale_series():
    from data.rf import coverage_problems
    freeze = pd.Timestamp("2026-11-06")
    one_year = pd.Series(0.055, index=pd.date_range("2025-07-02", "2026-06-24", freq="7D"))
    probs = coverage_problems(one_year, freeze)
    assert any("start" in p for p in probs) and any("end" in p for p in probs)
    full = pd.Series(0.06, index=pd.date_range("2012-06-27", "2026-11-04", freq="7D"))
    assert coverage_problems(full, freeze) == []
    assert coverage_problems(pd.Series(dtype=float), freeze) == ["no auction yields"]


def test_files_are_combined_and_conflicts_rejected(tmp_path):
    from data.rf import load_yields
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    pd.DataFrame({"Date": ["2020-01-01", "2020-01-08"], "Yield": [5.0, 5.1]}).to_csv(a, index=False)
    pd.DataFrame({"Date": ["2020-01-08", "2020-01-15"], "Yield": [5.1, 5.2]}).to_csv(b, index=False)
    y = load_yields([a, b])
    assert len(y) == 3
    pd.DataFrame({"Date": ["2020-01-08"], "Yield": [6.0]}).to_csv(b, index=False)
    with pytest.raises(ValueError):
        load_yields([a, b])
