from __future__ import annotations

import unittest

from frequency_sweep import MockClient, ScanConfig, SpectrumScanner


def _one_point_config(waveform: str) -> ScanConfig:
    return ScanConfig(
        sweep_mode="frequency",
        awg_resource="MOCK_AWG",
        scope_resource="MOCK_SCOPE",
        awg_vpp=2.7,
        awg_offset_v=0.0,
        fixed_frequency_hz=100_000.0,
        total_start_hz=100_000.0,
        total_stop_hz=100_000.0,
        subwindow_span_hz=10_000.0,
        step_size_hz=1_000.0,
        amp_start_vpp=1.0,
        amp_stop_vpp=1.0,
        amp_step_vpp=0.1,
        offset_start_v=0.0,
        offset_stop_v=0.0,
        offset_step_v=0.1,
        rbw_hz=1_000.0,
        iterations=1,
        avg_count=1,
        dwell_s=0.0,
        save_csv=False,
        csv_path="",
        use_mock=True,
        awg_waveform=waveform,
    )


class StandardSweepRegressionTests(unittest.TestCase):
    def test_existing_standard_waveforms_still_complete_a_scan(self) -> None:
        for waveform in ("SINusoid", "SQUare", "RAMP"):
            with self.subTest(waveform=waveform):
                client = MockClient()
                points = SpectrumScanner(client).run(_one_point_config(waveform))
                self.assertEqual(len(points), 1)
                self.assertEqual(client.current_awg_waveform, waveform)
                self.assertEqual(client.current_awg_hz, 100_000.0)
                self.assertEqual(client.current_awg_vpp, 2.7)
                self.assertEqual(client.current_awg_offset, 0.0)

    def test_afg1062_standard_waveform_frequency_limits(self) -> None:
        for waveform, maximum_hz in (
            ("SINusoid", 60_000_000.0),
            ("SQUare", 30_000_000.0),
            ("RAMP", 2_000_000.0),
        ):
            with self.subTest(waveform=waveform, boundary="accepted"):
                config = _one_point_config(waveform)
                config.total_start_hz = maximum_hz
                config.total_stop_hz = maximum_hz
                SpectrumScanner._validate_config(config)

            with self.subTest(waveform=waveform, boundary="rejected"):
                config.total_stop_hz = maximum_hz + 1.0
                with self.assertRaisesRegex(ValueError, "AFG1062"):
                    SpectrumScanner._validate_config(config)


if __name__ == "__main__":
    unittest.main()
