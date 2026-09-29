"""
Unit tests for enhanced solar flare cataloging and multi-instrument association logic.
"""

from __future__ import annotations

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from pipeline.catalog import (
    merge_catalogs,
    upgrade_catalog_schema,
    append_to_master,
    catalog_summary,
    load_master_catalog,
    CATALOG_COLUMNS,
)
from pipeline.detector import FlareEvent


class TestCatalogAssociation(unittest.TestCase):

    def test_basic_exact_match(self):
        t0 = pd.Timestamp("2024-05-10 12:00:00")
        t_peak = pd.Timestamp("2024-05-10 12:10:00")
        t_end = pd.Timestamp("2024-05-10 12:25:00")

        sxr_ev = [
            FlareEvent(
                start_time=t0,
                peak_time=t_peak,
                end_time=t_end,
                peak_flux=1.5e-5,
                flare_class="M1.5",
                instrument="solexs",
                channel="flux_long",
                peak_excess=1.2e-5,
                rise_rate=1e-7,
                peak_flux_wm2=1.5e-5,
            )
        ]

        hxr_ev = [
            FlareEvent(
                start_time=t0 + pd.Timedelta(minutes=2),
                peak_time=t_peak,
                end_time=t_end - pd.Timedelta(minutes=5),
                peak_flux=3500.0,
                flare_class="HXR",
                instrument="helios",
                channel="hxr_total",
            )
        ]

        merged = merge_catalogs(sxr_ev, hxr_ev)
        self.assertEqual(len(merged), 1)
        row = merged.iloc[0]
        self.assertTrue(row["confirmed_multi_instrument"])
        self.assertEqual(row["flare_class"], "M1.5")
        self.assertEqual(row["helios_peak_flux"], 3500.0)
        self.assertEqual(row["neupert_lag_sec"], 0.0)
        self.assertEqual(row["hxr_burst_count"], 1)
        self.assertEqual(row["duration_sec"], 1500.0)  # 25 min
        self.assertEqual(row["rise_time_sec"], 600.0)   # 10 min
        self.assertEqual(row["decay_time_sec"], 900.0)  # 15 min

    def test_neupert_rise_phase_match(self):
        """HXR peak precedes SXR peak by 6 minutes during the impulsive rise phase."""
        t_start = pd.Timestamp("2024-05-10 10:00:00")
        t_sxr_peak = pd.Timestamp("2024-05-10 10:15:00")
        t_end = pd.Timestamp("2024-05-10 10:40:00")

        t_hxr_peak = pd.Timestamp("2024-05-10 10:09:00")  # 6 min before SXR peak

        sxr_ev = [
            FlareEvent(
                start_time=t_start,
                peak_time=t_sxr_peak,
                end_time=t_end,
                peak_flux=2.4e-4,
                flare_class="X2.4",
                instrument="solexs",
                channel="flux_long",
                peak_flux_wm2=2.4e-4,
            )
        ]

        hxr_ev = [
            FlareEvent(
                start_time=t_start + pd.Timedelta(minutes=1),
                peak_time=t_hxr_peak,
                end_time=t_sxr_peak,
                peak_flux=12000.0,
                flare_class="HXR",
                instrument="helios",
                channel="hxr_total",
            )
        ]

        merged = merge_catalogs(sxr_ev, hxr_ev)
        self.assertEqual(len(merged), 1)
        row = merged.iloc[0]
        self.assertTrue(row["confirmed_multi_instrument"])
        self.assertEqual(row["helios_peak_flux"], 12000.0)
        # SXR peak (10:15) - HXR peak (10:09) = 360 seconds
        self.assertEqual(row["neupert_lag_sec"], 360.0)
        self.assertEqual(row["hxr_burst_count"], 1)

    def test_multi_burst_aggregation(self):
        """A major SXR flare accompanied by 2 successive HXR sub-bursts."""
        t_start = pd.Timestamp("2024-05-10 14:00:00")
        t_sxr_peak = pd.Timestamp("2024-05-10 14:20:00")
        t_end = pd.Timestamp("2024-05-10 14:50:00")

        sxr_ev = [
            FlareEvent(
                start_time=t_start,
                peak_time=t_sxr_peak,
                end_time=t_end,
                peak_flux=5.0e-5,
                flare_class="M5.0",
                instrument="solexs",
                channel="flux_long",
                peak_flux_wm2=5.0e-5,
            )
        ]

        hxr_burst1 = FlareEvent(
            start_time=t_start + pd.Timedelta(minutes=2),
            peak_time=t_start + pd.Timedelta(minutes=6),
            end_time=t_start + pd.Timedelta(minutes=9),
            peak_flux=4000.0,
            flare_class="HXR",
            instrument="helios",
            channel="hxr_total",
        )

        hxr_burst2 = FlareEvent(
            start_time=t_start + pd.Timedelta(minutes=10),
            peak_time=t_start + pd.Timedelta(minutes=15),
            end_time=t_start + pd.Timedelta(minutes=19),
            peak_flux=8500.0,  # Dominant burst
            flare_class="HXR",
            instrument="helios",
            channel="hxr_total",
        )

        merged = merge_catalogs(sxr_ev, [hxr_burst1, hxr_burst2])
        self.assertEqual(len(merged), 1)
        row = merged.iloc[0]
        self.assertTrue(row["confirmed_multi_instrument"])
        self.assertEqual(row["hxr_burst_count"], 2)
        self.assertEqual(row["helios_peak_flux"], 8500.0)
        # Dominant burst peak was at 14:15, SXR peak at 14:20 -> lag = 5 min = 300s
        self.assertEqual(row["neupert_lag_sec"], 300.0)

    def test_open_ended_missing_end_time(self):
        """Flare still active when satellite pass ended (end_time is None)."""
        t_start = pd.Timestamp("2024-05-10 23:45:00")
        t_sxr_peak = pd.Timestamp("2024-05-10 23:55:00")

        sxr_ev = [
            FlareEvent(
                start_time=t_start,
                peak_time=t_sxr_peak,
                end_time=None,
                peak_flux=3.0e-5,
                flare_class="M3.0",
                instrument="solexs",
                channel="flux_long",
                peak_flux_wm2=3.0e-5,
            )
        ]

        hxr_ev = [
            FlareEvent(
                start_time=t_start + pd.Timedelta(minutes=2),
                peak_time=t_sxr_peak - pd.Timedelta(minutes=1),
                end_time=None,
                peak_flux=1500.0,
                flare_class="HXR",
                instrument="helios",
                channel="hxr_total",
            )
        ]

        merged = merge_catalogs(sxr_ev, hxr_ev)
        self.assertEqual(len(merged), 1)
        row = merged.iloc[0]
        self.assertTrue(pd.isna(row["end_time"]))
        self.assertTrue(pd.isna(row["duration_sec"]))
        self.assertTrue(pd.isna(row["decay_time_sec"]))
        self.assertEqual(row["rise_time_sec"], 600.0)
        self.assertTrue(row["confirmed_multi_instrument"])
        self.assertEqual(row["neupert_lag_sec"], 60.0)

    def test_standalone_helios_events(self):
        """HEL1OS burst far outside any SoLEXS window should be kept as standalone HXR."""
        t_sxr = pd.Timestamp("2024-05-10 08:00:00")
        t_hxr = pd.Timestamp("2024-05-10 16:00:00")  # 8 hours later

        sxr_ev = [
            FlareEvent(
                start_time=t_sxr,
                peak_time=t_sxr + pd.Timedelta(minutes=5),
                end_time=t_sxr + pd.Timedelta(minutes=15),
                peak_flux=1.0e-6,
                flare_class="C1.0",
                instrument="solexs",
                channel="flux_long",
                peak_flux_wm2=1.0e-6,
            )
        ]

        hxr_ev = [
            FlareEvent(
                start_time=t_hxr,
                peak_time=t_hxr + pd.Timedelta(minutes=2),
                end_time=t_hxr + pd.Timedelta(minutes=10),
                peak_flux=900.0,
                flare_class="HXR",
                instrument="helios",
                channel="hxr_total",
            )
        ]

        merged = merge_catalogs(sxr_ev, hxr_ev)
        self.assertEqual(len(merged), 2)
        hxr_row = merged[merged["flare_class"] == "HXR"].iloc[0]
        self.assertFalse(hxr_row["confirmed_multi_instrument"])
        self.assertTrue(pd.isna(hxr_row["solexs_peak_flux"]))
        self.assertEqual(hxr_row["helios_peak_flux"], 900.0)
        self.assertEqual(hxr_row["hxr_burst_count"], 1)

    def test_upgrade_catalog_schema(self):
        """Verify backward compatibility and auto-population on legacy 8-column data."""
        legacy_df = pd.DataFrame([
            {
                "start_time": "2024-02-01 10:00:00",
                "peak_time": "2024-02-01 10:10:00",
                "end_time": "2024-02-01 10:30:00",
                "solexs_peak_flux": 2.5e-5,
                "helios_peak_flux": 1200.0,
                "flare_class": "M2.5",
                "hardness_ratio": 4.8e7,
                "confirmed_multi_instrument": True,
            }
        ])

        upgraded = upgrade_catalog_schema(legacy_df)
        for col in CATALOG_COLUMNS:
            self.assertIn(col, upgraded.columns)

        row = upgraded.iloc[0]
        self.assertEqual(row["duration_sec"], 1800.0)
        self.assertEqual(row["rise_time_sec"], 600.0)
        self.assertEqual(row["decay_time_sec"], 1200.0)
        self.assertEqual(row["hxr_burst_count"], 1)

    def test_append_and_deduplicate(self):
        """Verify append_to_master deduplicates events within 1 min, favoring confirmed events."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / "test_master.parquet"

            # Initial event (unconfirmed)
            ev1 = pd.DataFrame([{
                "start_time": pd.Timestamp("2024-03-01 12:00:00"),
                "peak_time": pd.Timestamp("2024-03-01 12:10:00"),
                "end_time": pd.Timestamp("2024-03-01 12:20:00"),
                "flare_class": "C2.0",
                "confirmed_multi_instrument": False,
            }])
            append_to_master(ev1, test_path)

            # Second event: same peak within 30s, but confirmed multi-instrument
            ev2 = pd.DataFrame([{
                "start_time": pd.Timestamp("2024-03-01 12:00:00"),
                "peak_time": pd.Timestamp("2024-03-01 12:10:30"),
                "end_time": pd.Timestamp("2024-03-01 12:20:00"),
                "flare_class": "C2.5",
                "confirmed_multi_instrument": True,
                "neupert_lag_sec": 120.0,
                "hxr_burst_count": 1,
            }])
            res = append_to_master(ev2, test_path)

            self.assertEqual(len(res), 1)
            self.assertTrue(res.iloc[0]["confirmed_multi_instrument"])
            self.assertEqual(res.iloc[0]["neupert_lag_sec"], 120.0)

    def test_catalog_summary(self):
        """Verify summary string includes multi-instrument, Neupert lag, and multi-burst stats."""
        df = pd.DataFrame([
            {
                "start_time": pd.Timestamp("2024-03-01 12:00:00"),
                "peak_time": pd.Timestamp("2024-03-01 12:10:00"),
                "end_time": pd.Timestamp("2024-03-01 12:20:00"),
                "flare_class": "M1.0",
                "confirmed_multi_instrument": True,
                "neupert_lag_sec": 95.0,
                "hxr_burst_count": 2,
            }
        ])
        summary = catalog_summary(df)
        self.assertIn("Total: 1 events", summary)
        self.assertIn("Multi-instrument: 1", summary)
        self.assertIn("Median Neupert lag: +95.0s", summary)
        self.assertIn("Multi-burst events: 1", summary)


if __name__ == "__main__":
    unittest.main()

