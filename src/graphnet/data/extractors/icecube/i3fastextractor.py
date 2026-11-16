"""I3Extractor class(es) for extracting specific, reconstructed features."""

from typing import TYPE_CHECKING, Any, Dict, List
from .i3extractor import I3Extractor
from graphnet.data.extractors.icecube.utilities.frames import (
    get_om_keys_and_pulseseries,
    put_frame_key,
)
from graphnet.utilities.imports import has_icecube_package

if has_icecube_package() or TYPE_CHECKING:
    from icecube import icetray  # pyright: reportMissingImports=false
    from icecube import phys_services

import numpy as np
class I3FastExtractor(I3Extractor):
    """Base class for extracting specific, reconstructed features."""

    def __init__(self, 
        pulsemap: str, 
        exclude: list = [None],
        training_data: bool = False,
        #exclude_saturation: bool = False,
        #exclude_calibration_errata: bool = False,
        charge_windows: tuple = (5, 10, 25, 50, 100, 250, 500, 1000),
        percentiles: np.ndarray = np.asarray([1, 10, 25, 50, 75, 90, 95]),
        saturation_exclusions: bool = False,
        errata_exclusions: bool = False,
        percentile_time_cap: float = 1000.0,
    ) -> None:
        """Construct I3FastExtractor.

        Args:
            pulsemap: Name of the pulse (series) map for which to extract
                reconstructed features.
            exclude: List of keys to exclude from the extracted data.
            training_data: Whether to also extract MC-truth based features
                (`dom_r`, `dom_time_residual`). Requires MC keys in frame.
            percentile_time_cap: Time window [ns] after the first pulse
                within which the `time_after_<p>` charge percentiles are
                computed.
        """
        # Member variable(s)
        self._pulsemap = pulsemap
        #self._exclude_saturation = exclude_saturation
        #self._exclude_calibration_errata = exclude_calibration_errata
        self._training_data = training_data
        self._percentile_time_cap = percentile_time_cap

        # Must match the `charge_after_<w>` / `time_after_<p>` output keys
        self._charge_windows = charge_windows
        self._percentiles = percentiles
        self._saturation_exclusions = saturation_exclusions
        self._errata_exclusions = errata_exclusions
        # Base class constructor
        super().__init__(pulsemap, exclude=exclude)


class I3FastExtractorIceCube86(I3FastExtractor):
    """Class for extracting reconstructed features for IceCube-86."""

    def __call__(self, frame: "icetray.I3Frame") -> Dict[str, List[Any]]:
        """Extract reconstructed features from `frame`.

        Args:
            frame: Physics (P) I3-frame from which to extract reconstructed
                features.

        Returns:
            Dictionary of reconstructed features for all pulses in `pulsemap`,
                in pure-python format.
        """
        padding_value: float = -1.0
        output: Dict[str, List[Any]] = {
            "dom_time": [],
            "width": [],
            "dom_x": [],
            "dom_y": [],
            "dom_z": [],
            "pmt_area": [],
            "rde": [],
            "is_bright_dom": [],
            "is_saturated_dom": [],
            "is_errata_dom": [],
            #"event_time": [],
            "hlc": [],
            "awtd": [],
            "string": [],
            #"pmt_number": [],
            "dom_number": [],
            #"dom_type": [],
            "dom_qtot": [],
            "dom_qtot_no_sat": [],
            "dom_qtot_no_sat_no_errata": [],
        }

        output.update({f'charge_after_{w}': [] for w in self._charge_windows})
        output.update({f'time_after_{p}': [] for p in self._percentiles})
        
        
        if self._saturation_exclusions:
            output.update({f'charge_after_{w}_sat': [] for w in self._charge_windows})
            output.update({f'time_after_{p}_sat': [] for p in self._percentiles})

        if self._errata_exclusions:
            output.update({f'charge_after_{w}_sat_no_errata': [] for w in self._charge_windows})
            output.update({f'time_after_{p}_sat_no_errata': [] for p in self._percentiles})

        if self._training_data:
            output["dom_r"] = []
            output["dom_time_residual"] = []


        # Get OM data
        if self._pulsemap in frame:
            om_keys, data = get_om_keys_and_pulseseries(
                frame,
                self._pulsemap,
                self._calibration,
            )
        else:
            self.warning_once(f"Pulsemap {self._pulsemap} not found in frame.")
            return output

        # Added these :
        bright_doms = None
        bad_doms = None
        saturation_windows = None
        calibration_errata = None

        if "BrightDOMs" in frame:
            bright_doms = frame.Get("BrightDOMs")

        if "BadDomsList" in frame:
            bad_doms = set(frame.Get("BadDomsList"))

        if "SaturationWindows" in frame:
            saturation_windows = frame.Get("SaturationWindows")

        if "CalibrationErrata" in frame:
            calibration_errata = frame.Get("CalibrationErrata")

        hqtot = frame["HQTOT"].value if "HQTOT" in frame else None
        total_charge = frame["Total_Charge"].value if "Total_Charge" in frame else None

        event_time = frame["I3EventHeader"].start_time.mod_julian_day_double

        if self._training_data:
            leading_primary, leading = self._get_leading_charged_particles(
                frame,
            )
        for om_key in om_keys:

            # DOM flags
            if bright_doms is not None:
                is_bright_dom = 1 if om_key in bright_doms else 0
            else:
                is_bright_dom = int(padding_value)


            if bad_doms is not None:
                is_bad_dom = 1 if om_key in bad_doms else 0
            else:
                is_bad_dom = int(padding_value)


            if is_bad_dom == 1:
                continue  # Skip bad DOMs
            
            
            # Common values for each OM
            x = self._gcd_dict[om_key].position.x
            y = self._gcd_dict[om_key].position.y
            z = self._gcd_dict[om_key].position.z
            area = self._gcd_dict[om_key].area
            rde = self._get_relative_dom_efficiency(
                frame, om_key, padding_value
            )

            string = om_key[0]
            dom_number = om_key[1]
            pmt_number = om_key[2]
            dom_type = self._gcd_dict[om_key].omtype
            
            #output["is_bright_dom"].append(is_bright_dom)
            #output["is_bad_dom"].append(is_bad_dom)

            if saturation_windows is not None:
                is_saturated_dom = 1 if om_key in saturation_windows else 0
            else:
                is_saturated_dom = int(padding_value)

            if calibration_errata is not None:
                is_errata_dom = 1 if om_key in calibration_errata else 0
            else:
                is_errata_dom = int(padding_value)

            # Loop over pulses for each OM
            pulses = data[om_key]

            min_pulse_time = getattr(pulses[0], "time", padding_value)
            output["width"].append(getattr(pulses[0], "width", padding_value))

            output['dom_time'].append(min_pulse_time)

            if self._training_data:
                r = phys_services.I3Calculator.closest_approach_distance(leading, self._gcd_dict[om_key].position)
                t_res = phys_services.I3Calculator.time_residual(leading, self._gcd_dict[om_key].position, min_pulse_time)
                output['dom_r'].append(r)
                output['dom_time_residual'].append(t_res)
            output['dom_x'].append(x)
            output['dom_y'].append(y)
            output['dom_z'].append(z)
            output["pmt_area"].append(area)
            output["rde"].append(rde)
            output["is_saturated_dom"].append(is_saturated_dom)
            output["is_errata_dom"].append(is_errata_dom)

            flags = getattr(pulses[0], "flags", padding_value)
            if flags == padding_value:
                output["hlc"].append(padding_value)
                output["awtd"].append(padding_value)
            else:
                output["hlc"].append((pulses[0].flags >> 0) & 0x1)  # bit 0
                output["awtd"].append(self._parse_awtd_flag(pulses[0]))

            # Pulse arrays (pulse series are time-ordered)
            n_pulses = len(pulses)
            times = np.fromiter((p.time for p in pulses), dtype=np.float64, count=n_pulses)
            charges = np.fromiter((p.charge for p in pulses), dtype=np.float64, count=n_pulses)
            dt = times - min_pulse_time

            # Pulses inside any saturation / calibration-errata window
            is_saturated_pulse = np.zeros(n_pulses, dtype=bool)
            if is_saturated_dom == 1:
                for window in saturation_windows[om_key]:
                    is_saturated_pulse |= (times >= window.start) & (times <= window.stop)
            is_errata_pulse = np.zeros(n_pulses, dtype=bool)
            if is_errata_dom == 1:
                for window in calibration_errata[om_key]:
                    is_errata_pulse |= (times >= window.start) & (times <= window.stop)

            cq = np.cumsum(charges)
            cq_no_sat = np.cumsum(np.where(is_saturated_pulse, 0.0, charges))
            cq_no_sat_no_errata = np.cumsum(
                np.where(is_saturated_pulse | is_errata_pulse, 0.0, charges)
            )

            # charge_after_<w>: cumulative charge up to the last pulse with dt <= w
            last = np.searchsorted(dt, self._charge_windows, side="right") - 1  # dt[0] == 0
            for w, i in zip(self._charge_windows, last):
                output[f"charge_after_{w}"].append(float(cq[i]))
                if self._saturation_exclusions:
                    output[f"charge_after_{w}_sat"].append(float(cq_no_sat[i]))
                if self._errata_exclusions:
                    output[f"charge_after_{w}_sat_no_errata"].append(float(cq_no_sat_no_errata[i]))

            # time_after_<p>: time since first pulse at which p% of the charge
            # within `percentile_time_cap` has been collected
            n_cap = np.searchsorted(dt, self._percentile_time_cap, side="right")

            # Only if saturation / errata exclusions are enabled, compute the percentiles for each case
            cq_variants = [("", cq)]
            if self._saturation_exclusions:
                cq_variants.append(("_sat", cq_no_sat))
            if self._errata_exclusions:
                cq_variants.append(("_sat_no_errata", cq_no_sat_no_errata))

            for suffix, cq_x in cq_variants:
                t_pct = self._percentile_times(dt[:n_cap], cq_x[:n_cap], padding_value)
                for p, t in zip(self._percentiles, t_pct):
                    output[f"time_after_{p}{suffix}"].append(float(t))

            dom_qtot = float(cq[-1])
            output['dom_qtot'].append(dom_qtot)
            output['dom_qtot_no_sat'].append(float(cq_no_sat[-1]))
            output['dom_qtot_no_sat_no_errata'].append(float(cq_no_sat_no_errata[-1]))
            #output["event_time"].append(event_time)
            output["string"].append(string)
            #output["pmt_number"].append(pmt_number)
            output["dom_number"].append(dom_number)
            #output["dom_type"].append(dom_type)

            if total_charge is None:
                output['is_bright_dom'].append(int(padding_value))
            elif dom_qtot >= total_charge/10:
                output['is_bright_dom'].append(1)
            else:
                output['is_bright_dom'].append(0)

        if self._training_data:
            put_frame_key(frame, 'NumberStrings', icetray.I3Int(len(set(output["string"]))))
            put_frame_key(frame, 'NumberDOMs', icetray.I3Int(len(output["dom_number"])))
            put_frame_key(frame, 'NumberStringsHLC', icetray.I3Int(len(set([s for s, h in zip(output["string"], output["hlc"]) if h == 1]))))
            put_frame_key(frame, 'NumberDOMsHLC', icetray.I3Int(len([h for h in output["hlc"] if h == 1])))

        return output

    def _percentile_times(
        self, dt: np.ndarray, cq: np.ndarray, padding_value: float
    ) -> np.ndarray:
        """Get times at which `cq` first reaches each of `self._percentiles`.

        Args:
            dt: Pulse times relative to the first pulse, time-ordered.
            cq: Cumulative charge at each pulse (excluded pulses contribute
                zero, so `cq` is flat across them).
            padding_value: Returned for all percentiles if `cq` holds no
                charge (e.g. all pulses saturated / in errata).

        Returns:
            Array of times, one per percentile.
        """
        if cq[-1] <= 0:
            return np.full(len(self._percentiles), padding_value)
        targets = cq[-1] * (self._percentiles / 100.0)
        idx = np.minimum(np.searchsorted(cq, targets, side="left"), len(cq) - 1)
        return dt[idx]

    def _get_relative_dom_efficiency(
        self, frame: "icetray.I3Frame", om_key: int, padding_value: float
    ) -> float:
        if (
            "I3Calibration" in frame
        ):  # Not available for e.g. mDOMs in IceCube Upgrade
            rde = frame["I3Calibration"].dom_cal[om_key].relative_dom_eff
        else:
            try:
                assert self._calibration is not None
                rde = self._calibration.dom_cal[om_key].relative_dom_eff
            except:  # noqa: E722
                rde = padding_value
        return rde

    def _parse_awtd_flag(
        self, pulse: Any, fadc_min_width_ns: float = 6.0
    ) -> bool:
        """Parse awtd flag from pulse width.

        Returns True if the pulse was readout using the awtd digitizer.

        Method by Tom Stuttard.

        Notes from Tom:
        Function to read the bits of the pulse flags and unpack them into
        meaningful info Using pulse width rather than flags to separate FADC vs
        ATWD pulses, due to a known issue.
        https://github.com/icecube/icetray/issues/2721 Note that the issue
        states to use 8ns, but I have found that actually 6ns is correct.
        """
        # Use pulse width to check whether a pulse is
        # (a) FADC-only, or
        # includes ATWD (and probably also FADC)
        return pulse.width < (fadc_min_width_ns * icetray.I3Units.ns)
    
    def _get_leading_charged_particles(
        self,
        frame,
    ):

        primary = frame["PolyplopiaPrimary"]
        pdg = frame["PolyplopiaPrimary"].pdg_encoding
        mctree = frame["I3MCTree_preMuonProp"]
        if np.abs(pdg) in [12, 14, 16]:
            current = mctree[1]
            while mctree.number_of_children(current) > 0:
                current = mctree.first_child(current)

            primary.pos.x = current.pos.x
            primary.pos.y = current.pos.y
            primary.pos.z = current.pos.z
        else:
            current = mctree[frame["PolyplopiaPrimary"]]
            highest_energy = -1
            bundle_particles = mctree.get_daughters(current)
            for particle in bundle_particles:
                if (
                    particle.type_string in ["MuPlus", "MuMinus"]
                    and particle.location_type_string == "InIce"
                ):
                    if particle.energy > highest_energy:
                        highest_energy = particle.energy
                        current = particle

            tracklist = frame["MMCTrackList"]

            e_initial = 0
            for track in tracklist:
                if mctree.is_in_subtree(primary, track.particle) == True: # Cleaning Coincidence Hits
                    if track.Ei > e_initial:
                        e_initial = track.Ei
                        current = track.particle

        return primary, current
