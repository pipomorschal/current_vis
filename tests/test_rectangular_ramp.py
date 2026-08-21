from __future__ import annotations

import unittest

import numpy as np

from frequency_sweep import (
    AWG_WAVEFORM_OPTIONS,
    RECTANGULAR_RAMP_WAVEFORM,
    TektronixVisaClient,
    normalize_awg_waveform,
)
from rectangular_ramp import (
    AFG1062_DAC_MAX_CODE,
    AFG1062_MAX_ARB_POINTS,
    AFG1062_MAX_SAMPLE_RATE_SPS,
    RectangularRampSettings,
    generate_rectangular_ramp,
)


class RectangularRampGenerationTests(unittest.TestCase):
    def test_default_grid_obeys_afg1062_limits_and_frequency_relationship(self) -> None:
        waveform = generate_rectangular_ramp(RectangularRampSettings())

        self.assertEqual(AFG1062_MAX_ARB_POINTS, 1_048_576)
        self.assertEqual(waveform.samples_per_period, 756)
        self.assertEqual(waveform.point_count, 7560)
        self.assertEqual(waveform.arb_repetition_hz, 39_500.0)
        self.assertEqual(waveform.effective_sample_rate_sps, 298_620_000.0)
        self.assertLessEqual(waveform.point_count, AFG1062_MAX_ARB_POINTS)
        self.assertLessEqual(
            waveform.effective_sample_rate_sps, AFG1062_MAX_SAMPLE_RATE_SPS
        )
        self.assertAlmostEqual(
            waveform.arb_repetition_hz * waveform.settings.periods,
            waveform.settings.modulation_frequency_hz,
        )

    def test_square_is_exactly_half_duty_and_baseline_changes_per_period(self) -> None:
        settings = RectangularRampSettings(ramp_slope_mv_per_period=3.25)
        waveform = generate_rectangular_ramp(settings)
        index = np.arange(waveform.point_count, dtype=float)
        baseline = 3.25e-3 * index / waveform.samples_per_period
        rectangular = waveform.voltage_v - baseline

        np.testing.assert_allclose(
            np.abs(rectangular), settings.rectangular_vpp / 2.0, atol=1e-12
        )
        self.assertEqual(np.count_nonzero(rectangular > 0.0), waveform.point_count // 2)
        np.testing.assert_allclose(
            waveform.voltage_v[waveform.samples_per_period :]
            - waveform.voltage_v[: -waveform.samples_per_period],
            3.25e-3,
            atol=1e-12,
        )

    def test_positive_and_negative_ramp_reset_directions(self) -> None:
        positive = generate_rectangular_ramp(
            RectangularRampSettings(ramp_slope_mv_per_period=2.0, periods=10)
        )
        negative = generate_rectangular_ramp(
            RectangularRampSettings(ramp_slope_mv_per_period=-2.0, periods=10)
        )

        self.assertAlmostEqual(positive.baseline_change_v, 0.020)
        self.assertAlmostEqual(positive.baseline_reset_jump_v, -0.020)
        self.assertAlmostEqual(positive.record_wrap_jump_v, 2.380)
        self.assertAlmostEqual(negative.baseline_change_v, -0.020)
        self.assertAlmostEqual(negative.baseline_reset_jump_v, 0.020)
        self.assertAlmostEqual(negative.record_wrap_jump_v, 2.420)

    def test_afg_amplitude_and_offset_reconstruct_the_requested_samples(self) -> None:
        waveform = generate_rectangular_ramp(
            RectangularRampSettings(ramp_slope_mv_per_period=-7.5)
        )
        reconstructed = waveform.afg_offset_v + waveform.total_waveform_vpp * (
            waveform.dac_codes.astype(float) / AFG1062_DAC_MAX_CODE - 0.5
        )
        maximum_quantization_error = (
            waveform.total_waveform_vpp / AFG1062_DAC_MAX_CODE / 2.0
        )

        np.testing.assert_allclose(
            reconstructed,
            waveform.voltage_v,
            atol=maximum_quantization_error + 1e-12,
        )
        self.assertAlmostEqual(
            waveform.total_waveform_vpp,
            float(np.ptp(waveform.voltage_v)),
        )
        self.assertAlmostEqual(
            waveform.afg_offset_v,
            0.5 * (float(np.min(waveform.voltage_v)) + float(np.max(waveform.voltage_v))),
        )

    def test_unrepresentable_frequency_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "300 MS/s"):
            generate_rectangular_ramp(
                RectangularRampSettings(modulation_frequency_hz=80_000_000.0)
            )


class _FakeAwg:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.binary_call = None
        self.point_count = 0
        self.pending_point_count = None
        self.function = "SIN"
        self.amplitude_vpp = 1.0
        self.offset_v = 0.0
        self.frequency_hz = 1_000.0
        self.output_enabled = False

    def write(self, command: str) -> None:
        self.commands.append(command)
        if command.startswith("DATA:POINts EMEMory,"):
            self.pending_point_count = int(command.rsplit(",", 1)[1])
        elif command == "SOURce1:FUNCtion EMEMory":
            self.function = "EMEM"
        elif command.startswith("SOURce1:VOLTage:LEVel:IMMediate:AMPLitude "):
            self.amplitude_vpp = float(command.split()[-2])
        elif command.startswith("SOURce1:VOLTage:LEVel:IMMediate:OFFSet "):
            self.offset_v = float(command.split()[-1])
        elif command.startswith("SOURce1:FREQuency:FIXed "):
            self.frequency_hz = float(command.split()[-1])
        elif command == "OUTPut1:STATe ON":
            self.output_enabled = True
        elif command == "OUTPut1:STATe OFF":
            self.output_enabled = False

    def write_binary_values(self, command, values, **kwargs) -> None:
        if self.pending_point_count is not None:
            raise AssertionError("Binary transfer started before point allocation completed")
        self.binary_call = (command, values, kwargs)

    def query_binary_values(self, command, **kwargs):
        self.commands.append(command)
        if command != "DATA:DATA? EMEMory":
            raise AssertionError(f"Unexpected binary query: {command}")
        self.binary_query_kwargs = kwargs
        return list(self.binary_call[1])

    def query(self, command: str) -> str:
        if command == "DATA:POINts? EMEMory":
            return str(self.point_count)
        if command == "SOURce1:FUNCtion?":
            return self.function
        if command == "SOURce1:VOLTage:LEVel:IMMediate:AMPLitude?":
            return str(self.amplitude_vpp)
        if command == "SOURce1:VOLTage:LEVel:IMMediate:OFFSet?":
            return str(self.offset_v)
        if command == "SOURce1:FREQuency:FIXed?":
            return str(self.frequency_hz)
        if command == "OUTPut1:STATe?":
            return "1" if self.output_enabled else "0"
        if command.startswith("DATA:DATA:VALue? EMEMory,"):
            point = int(command.rsplit(",", 1)[1])
            return str(self.binary_call[1][point - 1])
        if command == "SYSTem:ERRor:NEXT?":
            return '0,"No error"'
        if command == "*OPC?":
            if self.pending_point_count is not None:
                self.point_count = self.pending_point_count
                self.pending_point_count = None
            return "1"
        raise AssertionError(f"Unexpected query: {command}")


class _TenKPointAwg(_FakeAwg):
    """Simulate an instrument that keeps its existing 10 kpoint edit record."""

    def __init__(self) -> None:
        super().__init__()
        self.point_count = 10_000

    def query(self, command: str) -> str:
        if command == "*OPC?":
            return "1"
        return super().query(command)


class _BinaryLocalWarningAwg(_FakeAwg):
    """Queue the AFG1000 firmware's spurious -201 after binary transfer."""

    def __init__(self) -> None:
        super().__init__()
        self.local_warning_pending = False

    def write_binary_values(self, command, values, **kwargs) -> None:
        super().write_binary_values(command, values, **kwargs)
        self.local_warning_pending = True

    def query(self, command: str) -> str:
        if command == "SYSTem:ERRor:NEXT?" and self.local_warning_pending:
            self.local_warning_pending = False
            return '-201,"Invalid while in local"'
        return super().query(command)


class _RejectedNegativeOffsetAwg(_FakeAwg):
    """Simulate a genuine local-mode rejection of the negative offset write."""

    def __init__(self) -> None:
        super().__init__()
        self.local_warning_pending = False

    def write(self, command: str) -> None:
        if command.startswith("SOURce1:VOLTage:LEVel:IMMediate:OFFSet "):
            requested_offset = float(command.split()[-1])
            if requested_offset < 0.0:
                self.commands.append(command)
                self.local_warning_pending = True
                return
        super().write(command)

    def query(self, command: str) -> str:
        if command == "SYSTem:ERRor:NEXT?" and self.local_warning_pending:
            self.local_warning_pending = False
            return '-201,"Invalid while in local"'
        return super().query(command)


class AfgUploadTests(unittest.TestCase):
    def test_upload_uses_big_endian_14_bit_block_and_applies_scaling(self) -> None:
        waveform = generate_rectangular_ramp(
            RectangularRampSettings(ramp_slope_mv_per_period=1.0)
        )
        fake = _FakeAwg()
        client = TektronixVisaClient()
        client.awg = fake

        client.upload_arbitrary_waveform(waveform)

        self.assertIsNotNone(fake.binary_call)
        command, values, kwargs = fake.binary_call
        self.assertEqual(command, "DATA EMEMory,")
        self.assertEqual(values, waveform.dac_codes.tolist())
        self.assertEqual(kwargs["datatype"], "H")
        self.assertTrue(kwargs["is_big_endian"])
        self.assertIn("DATA:DATA? EMEMory", fake.commands)
        self.assertIn("SOURce1:ASKey:STATe OFF", fake.commands)
        self.assertIn("SOURce1:PSKey:STATe OFF", fake.commands)
        self.assertIn("SOURce1:PWM:STATe OFF", fake.commands)
        self.assertEqual(fake.binary_query_kwargs["datatype"], "H")
        self.assertTrue(fake.binary_query_kwargs["is_big_endian"])
        self.assertIn("SOURce1:FUNCtion EMEMory", fake.commands)
        self.assertIn(
            f"SOURce1:FREQuency:FIXed {waveform.arb_repetition_hz}", fake.commands
        )
        self.assertIn(
            "SOURce1:VOLTage:LEVel:IMMediate:AMPLitude "
            f"{waveform.total_waveform_vpp} VPP",
            fake.commands,
        )
        self.assertIn(
            "SOURce1:VOLTage:LEVel:IMMediate:OFFSet "
            f"{waveform.afg_offset_v}",
            fake.commands,
        )
        self.assertEqual(fake.commands[-1], "OUTPut1:STATe ON")

    def test_verified_local_warning_allows_negative_slope(self) -> None:
        waveform = generate_rectangular_ramp(
            RectangularRampSettings(ramp_slope_mv_per_period=-2.0)
        )
        fake = _BinaryLocalWarningAwg()
        client = TektronixVisaClient()
        client.awg = fake

        warnings = client.upload_arbitrary_waveform(waveform)

        self.assertLess(waveform.afg_offset_v, 0.0)
        self.assertAlmostEqual(fake.offset_v, waveform.afg_offset_v)
        self.assertTrue(fake.output_enabled)
        self.assertEqual(len(warnings), 1)
        self.assertIn("binary transfer", warnings[0])
        self.assertIn("-201", warnings[0])

    def test_genuinely_rejected_negative_offset_remains_a_failure(self) -> None:
        waveform = generate_rectangular_ramp(
            RectangularRampSettings(ramp_slope_mv_per_period=-2.0)
        )
        fake = _RejectedNegativeOffsetAwg()
        client = TektronixVisaClient()
        client.awg = fake

        with self.assertRaisesRegex(
            RuntimeError,
            r"setup readback did not match: offset=0 V, expected -",
        ):
            client.upload_arbitrary_waveform(waveform)

        self.assertFalse(fake.output_enabled)

    def test_unaccepted_point_allocation_stops_before_binary_transfer(self) -> None:
        waveform = generate_rectangular_ramp(RectangularRampSettings())
        fake = _TenKPointAwg()
        client = TektronixVisaClient()
        client.awg = fake

        with self.assertRaisesRegex(
            RuntimeError,
            r"before upload \(10000 points reported, 7560 requested\)",
        ):
            client.upload_arbitrary_waveform(waveform)

        self.assertIsNone(fake.binary_call)
        self.assertNotIn("OUTPut1:STATe ON", fake.commands)

    def test_standard_waveform_normalization_is_unchanged(self) -> None:
        self.assertEqual(normalize_awg_waveform("sinusoid"), "SINusoid")
        self.assertEqual(normalize_awg_waveform("square"), "SQUare")
        self.assertEqual(normalize_awg_waveform("ramp"), "RAMP")
        self.assertIn(("Rectangular + Ramp", RECTANGULAR_RAMP_WAVEFORM), AWG_WAVEFORM_OPTIONS)


if __name__ == "__main__":
    unittest.main()
