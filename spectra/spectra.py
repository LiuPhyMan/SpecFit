#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 0:31 2018/4/5

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   test
@IDE:       PyCharm
"""
import os
import math
import numpy as np
from scipy import sparse as spr
from scipy.optimize import curve_fit
import pandas as pd
from BasicFunc import constants as const
from BasicFunc import gaussian, lorentzian, pseudo_voigt
from BasicFunc import tracer
from copy import deepcopy
from matplotlib import pyplot as plt
from lmfit import Model


# from .voigt import voigt_pseudo


def convolute_to_voigt(*, fwhm_G, fwhm_L):
    return 0.5346 * fwhm_L + np.sqrt(0.2166 * fwhm_L ** 2 + fwhm_G ** 2)


class MoleculeState(object):

    def __init__(self, state):
        super().__init__()
        self.state = state
        if state == 'OH(A)':
            # LIFBASE manual.pdf
            self.Te = 32683.7
        if state == 'OH(X)':
            self.Te = 0
            self.we = 3737.76
            self.wexe = 84.881
            self.Be = 18.910
            self.ae = 0.7242
            self.De = 19.38e-4
            self.be = 0
        if state == 'CO(B)':  # from nist
            self.Te = 86928
            self.we = 2112.7
            self.wexe = 15.22
            self.Be = 1.9612
            self.ae = 0.026
            self.De = 7.10e-6
            self.be = 0
        if state == 'CO(A)':  # from nist
            self.Te = 65074.6
            self.we = 1518.2
            self.wexe = 19.4
            self.Be = 1.6115
            self.ae = 0.02325
            self.De = 7.33e-6
            self.be = 0

    def Ge_term(self, v):
        if self.state == 'OH(A)':
            # LIFBASE manual.pdf
            const_array = np.array([3178.3554, -92.68141, -1.77305, .307923, -3.5494e-2])
            return const_array.dot(np.power(v + .5, [1, 2, 3, 4, 5]))
        elif self.state == 'CO(B)':
            # v numbers : 0-2
            v_term = [1072, 3154, 5154]
            return v_term[v]
        elif self.state == 'CO(A)':
            # v numbers : 0-6
            # 0-6
            v_term = [753.49, 2242.3, 3685.2, 5097.9, 6476.1, 7818.2, 9125.0]
            return v_term[v]
        elif self.state == "N2(C)":
            # v numbers : 0-4
            # Ref :     "Improved Fits for the Vibrational and Rotational Constants of Many States
            #           of Nitrogen and Oxygen"
            vib_const = np.array([2047.18, -28.445, 2.0883, -5.350e-1])
            return vib_const.dot(np.power(v + .5, [1, 2, 3, 4]))
        elif self.state == "N2p(B)":
            # v numbers : 0-4
            # Ref :     "Improved Fits for the Vibrational and Rotational Constants of Many States
            #           of Nitrogen and Oxygen"
            vib_const = np.array([2420.83, -23.851, -0.3587, -6.192e-2])
            return vib_const.dot(np.power(v + .5, [1, 2, 3, 4]))
        else:
            raise Exception(f"vib constants of {self.state} is not supported.")

    def Fev_term(self, v, J):
        if self.state in ('CO(B)',):
            Bv = self.Be - self.ae * (v + .5)
            Dv = self.De + self.be * (v + .5)
            Fev = Bv * J * (J + 1) - Dv * J ** 2 * (J + 1) ** 2
            return Fev
        elif self.state in ('CO(A)',):
            Bv = self.Be - self.ae * (v + .5)
            Dv = self.De + self.be * (v + .5)
            L = 1  # total orbital momentum in the direction of the axis
            Fev = Bv * (J * (J + 1) - L) - Dv * J ** 2 * (J + 1) ** 2
            return Fev
        elif self.state == 'N2p(B)':
            return None
        else:
            return

    def term(self, v, J):
        return self.Te + self.Ge_term(v) + self.Fev_term(v, J)


class UppwerState(object):

    def __init__(self):
        super().__init__()
        self.distribution = None
        self.num_branch = None
        self.gv = None
        self.Ge = None
        self.gJ = None
        self.Fev = None
        self.size = None
        self.shape = None

    def energy_term(self):
        return self.Te + self.Ge + self.Fev

    def ravel_distribution(self):
        return self.distribution.ravel()

    def reduced_distribution(self):
        return np.divide(self.distribution, self.gJ,
                         out=np.zeros_like(self.distribution),
                         where=self.gJ != 0)

    def set_distribution(self, _distribution):
        if len(_distribution.shape) == 1:
            assert _distribution.size == self.size
            self.distribution = _distribution.reshape((self.num_branch, -1))
        else:
            assert _distribution.shape == self.shape
            self.distribution = _distribution

    def set_distribution_error(self, _distri_error):
        if len(_distri_error.shape) == 1:
            assert _distri_error.size == self.size
            self.distribution_error = _distri_error.reshape((self.num_branch, -1))
        else:
            assert _distri_error.shape == self.shape
            self.distribution_error = _distri_error

    def set_maxwell_distribution(self, *, Tvib, Trot):
        vib_distribution = self.gv * np.exp(-self.Ge * const.WNcm2K / Tvib)
        rot_distribution = self.gJ * np.exp(-self.Fev * const.WNcm2K / Trot)
        self.distribution = vib_distribution * rot_distribution

    def set_double_maxwell_distribution(self, *, Tvib, Trot_cold, Trot_hot, hot_ratio):
        hot_part = self.gJ * np.exp(-self.Fev * const.WNcm2K / Trot_hot)
        cold_part = self.gJ * np.exp(-self.Fev * const.WNcm2K / Trot_cold)
        hot_part = hot_part / hot_part.sum()
        cold_part = cold_part / cold_part.sum()
        rot_distribution = hot_ratio * hot_part + (1 - hot_ratio) * cold_part
        vib_distribution = self.gv * np.exp(-self.Ge * const.WNcm2K / Tvib)
        self.distribution = vib_distribution * rot_distribution


class OHState(UppwerState):

    def __init__(self, *, state, v_upper):
        super().__init__()
        self.state = state
        self.v_upper = v_upper
        if v_upper == 0:
            self.N_max = 41
        if v_upper == 1:
            self.N_max = 39
        if v_upper == 2:
            self.N_max = 35
        if v_upper == 3:
            self.N_max = 32
        if self.state == "A":
            self.Te = 32684.1  # from NIST in unit of cm-1
            self.Te_eV = self.Te * const.WNcm2eV
            self.num_branch = 2
            self._N = np.vstack((np.arange(self.N_max + 1),  # F1 branch
                                 np.arange(self.N_max + 1)))  # F2 branch
            self._J = np.vstack((np.arange(self.N_max + 1) + 1 / 2,
                                 np.arange(self.N_max + 1) - 1 / 2))
            self.gJ = 2 * self._J + 1
            self.gv = 1
            self.distribution = np.ones_like(self._J)
            self.distribution_error = np.zeros_like(self._J)
        self.shape = self._N.shape
        self.size = self._N.size
        self._set_Ge()
        self._set_Fev()

    def _dataFrame(self):
        _df = pd.DataFrame(index=range((self.N_max + 1) * 2),
                           columns=["state", "v", "branch", "N", "J",
                                    "gJ", "energy",
                                    "distri.", "reduced_distri.", "distri._error"])
        _df["state"] = "OH(A)"
        _df["v"] = self.v_upper
        _df["branch"] = ["F1"] * (self.N_max + 1) + ["F2"] * (self.N_max + 1)
        _df["N"] = self._N.ravel()
        _df["J"] = self._J.ravel()
        _df["gJ"] = self.gJ.ravel()
        _df["energy"] = self.energy_term().ravel()
        _df["distri."] = self.ravel_distribution()
        _df["reduced_distri."] = self.reduced_distribution().ravel()
        _df["distri._error"] = self.distribution_error.ravel()
        return _df

    def _set_Ge(self):
        self.Ge = MoleculeState('OH(A)').Ge_term(self.v_upper)
        self.Ge_eV = self.Ge * const.WNcm2eV

    def _set_Fev(self):
        # 1962 The ultraviolet bands of OH. Table 2
        OH_A_consts = [[16.961, 0.00204, 0.1122],
                       [16.129, 0.00203, 0.1056],
                       [15.287, 0.00208, 0.0997],
                       [14.422, 0.00206, 0.0980]]
        Bv, Dv, gama = OH_A_consts[self.v_upper]
        _N = np.arange(self.N_max + 1)
        _average_term = Bv * _N * (_N + 1) - Dv * _N ** 2 * (_N + 1) ** 2
        Fev_F1 = _average_term + gama * (_N + 1 / 2)
        Fev_F2 = _average_term - gama * (_N + 1 / 2)
        Fev_F2[0] = 0.0  # F2 branch works from N = 1, J = 1/2
        self.Fev = np.vstack((Fev_F1, Fev_F2))
        self.Fev_eV = self.Fev * const.WNcm2eV

    def plot_distribution(self, *, branch="both", new_figure=True):
        if new_figure is True:
            plt.figure()
        if branch in ("F1", "both"):
            plt.errorbar(self.energy_term()[0] * const.WNcm2eV,
                         self.distribution[0],
                         yerr=self.distribution_error[0],
                         marker='o', capsize=5, label='F1_branch')
        if branch in ("F2", "both"):
            plt.errorbar(self.energy_term()[1, 1:] * const.WNcm2eV,
                         self.distribution[1, 1:],
                         yerr=self.distribution_error[1, 1:],
                         marker='o', capsize=5, label='F2_branch')
        plt.xlabel("Energy (eV)")
        plt.ylabel("Distri.")
        plt.title("OH({state}), v={v}".format(state=self.state, v=self.v_upper))
        plt.grid(linestyle="-.", alpha=0.7)
        plt.legend()

    def plot_reduced_distribution(self, *, branch="both", new_figure=True):
        if new_figure is True:
            plt.figure()
        if branch in ("F1", "both"):
            plt.semilogy(self.energy_term()[0] * const.WNcm2eV,
                         self.reduced_distribution()[0],
                         marker='o', label='F1_branch')
        if branch in ("F2", "both"):
            plt.semilogy(self.energy_term()[1, 1:] * const.WNcm2eV,
                         self.reduced_distribution()[1, 1:],
                         marker='o', label='F2_branch')
        plt.xlabel("Energy (eV)")
        plt.ylabel("Reduced. Distri.")
        plt.title("OH({state}), v={v}".format(state=self.state, v=self.v_upper))
        plt.grid(linestyle="-.", alpha=0.7)
        plt.legend()


class N2State(UppwerState):

    def __init__(self, *, state, v_upper, J_max=90):
        super().__init__()
        self.state = state
        self.J_max = J_max
        self.v_upper = v_upper
        if self.state == "C":
            self.num_branch = 3
            self._J = np.vstack((np.arange(self.J_max),
                                 np.arange(self.J_max),
                                 np.arange(self.J_max)))
        self.gJ = 2 * self._J + 1
        self.gv = 1
        self.distribution = np.ones_like(self._J)
        self.distribution_error = np.zeros_like(self._J)
        self.shape = self._J.shape
        self.size = self._J.size
        self._set_Ge()
        self._set_Fev()

    def __str__(self):
        return f"""
State: N2({self.state}), v={self.v_upper}
Shape: {self.shape}
Ge:    {self.Ge} cm-1"""

    def _set_Ge(self):
        Ge_array = [1016.70635, 3011.108325, 4951.9, 6825.931175, 8607.21165]
        self.Ge = Ge_array[self.v_upper]
        self.Ge_eV = self.Ge * const.WNcm2eV

    def _set_Fev(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        _path = dir_path + r"\N2(C-B)\Fev_{v}.dat".format(v=self.v_upper)
        self.Fev = np.loadtxt(_path).transpose()[:, :self.J_max]
        self.Fev_eV = self.Fev * const.WNcm2eV


class N2pState(UppwerState):

    def __init__(self, *, state, v_upper):
        super().__init__()
        self.state = state
        self.v_upper = v_upper
        self.N_max = 86
        if self.state == 'B':
            self.num_branch = 2
            self._N = np.vstack((np.arange(self.N_max + 1),  # F1 branch
                                 np.arange(self.N_max + 1)))  # F2 branch
            self._J = np.vstack((np.arange(self.N_max + 1) + 1 / 2,
                                 np.arange(self.N_max + 1) - 1 / 2))
        self.gJ = 2 * self._J + 1
        self.gv = 1
        self.distribution = np.ones_like(self._J)
        self.distribution_error = np.zeros_like(self._J)
        self.shape = self._N.shape
        self.size = self._N.size
        self._set_Ge()
        self._set_Fev()

    def __str__(self):
        return f"""
State: N2+({self.state}), v={self.v_upper}
Shape: {self.shape}
Ge:    {self.Ge} cm-1"""

    def _set_Ge(self):
        if self.state == 'B':
            self.Ge = MoleculeState("N2p(B)").Ge_term(self.v_upper)
            self.Ge_eV = self.Ge * const.WNcm2eV
        else:
            raise Exception(f"The state {self.state} is not support")

    def _set_Fev(self):
        # The spin orbit split is ignored.
        Bv_array = np.array([2.07461, 2.05171, 2.02750, 2.00083, 1.97220, 1.9394, 1.90398])
        Dv_array = np.array([6.33, 6.53, 6.89, 7.12, 7.79, 7.8, 9.4]) * 1e-6
        Bv = Bv_array[self.v_upper]
        Dv = Dv_array[self.v_upper]
        self.Fev = Bv * self._N * (self._N + 1) - Dv * self._N ** 2 * (self._N + 1) ** 2


class Spectra(object):
    WNcm2K = 1.4387773538277204

    def __init__(self):
        r"""
        wave_number, wave_length, distribution, emission_coeffcients and intensity
            are all in the same shape.
        """
        super().__init__()
        self.wave_number = None
        self.wave_length = None
        self.emission_coefficients = None
        self.distribution = None
        self.intensity = None
        self.normalized_factor = None

    # ------------------------------------------------------------------------------------------- #
    def set_intensity(self):
        self.intensity = self.distribution * self.emission_coefficients * self.wave_number
        self.intensity = self.intensity * 1.9864458241717582e-25  # multiply hc

    # ------------------------------------------------------------------------------------------- #
    def _set_delta_wv_spr_matrix(self, *, wv_exp, fwhm, threshold=8):
        delta_wv = wv_exp[np.newaxis].transpose() - self.wave_length.ravel()
        # _fwhm = max(fwhm["Gaussian"], fwhm["Lorentzian"])
        _fwhm = 0.5346 * fwhm["Lorentzian"] + np.sqrt(0.2166 * fwhm["Lorentzian"] ** 2
                                                      + fwhm["Gaussian"] ** 2)
        delta_wv[delta_wv < -_fwhm * threshold] = 0
        delta_wv[delta_wv > +_fwhm * threshold] = 0
        self._delta_wv_spr_matrix = spr.csr_matrix(delta_wv)
        self._convolved_delta_wv = deepcopy(self._delta_wv_spr_matrix)

    def convolve_slit_func(self, fwhm, slit_func):
        _data = self._delta_wv_spr_matrix.data
        if slit_func == "Gaussian":
            self._convolved_delta_wv.data = gaussian(delta_wv=_data, fwhm=fwhm["Gaussian"])
        elif slit_func == "Lorentzian":
            self._convolved_delta_wv.data = lorentzian(delta_wv=_data, fwhm=fwhm["Lorentzian"])
        elif slit_func == "Voigt":
            self._convolved_delta_wv.data = pseudo_voigt(delta_wv=_data, fwhm=fwhm)
        else:
            raise Exception(f"The slit_function {slit_func} is error.")

    def get_intensity_exp(self, *, normalized=True):
        _ = self._convolved_delta_wv.dot(self.intensity.ravel())
        if normalized is True:
            if _.max() > 0:
                return _ / _.max()
        return _

    # def get_extended_wavelength(self, *, waveLength_exp, fwhm, slit_func,
    #                             normalized=False, threshold=3):
    #     r"""
    #     Convolve the slit function on the experiment wavelength.
    #     Threshold :
    #         Gaussian :
    #             threshold : 2.5
    #             exp(-4*log(2)*2.5**2) = 2.98e-08
    #         Lorentzian :
    #             threshold : 5e2
    #             1/(4*5e2**2+1) = 1.0e-06
    #     """
    # #   Format the matrix of the delta wavelength. Convolve the slit function.
    # #                               wavelength_in_range
    # #                               *       *       *
    # #   delta_wv:   wv_exp_in_range *       *       *
    # #                               *       *       *

    @staticmethod
    def wavelength_vac2air(wv0):
        k0 = 238.0185
        k1 = 5792105
        k2 = 57.362
        k3 = 167917
        wvnm = 1e3 / wv0
        n = (k1 / (k0 - wvnm ** 2) + k3 / (k2 - wvnm ** 2)) * 1e-8 + 1
        return wv0 / n


class MoleculeSpectra(Spectra):

    def __init__(self):
        super().__init__()
        self.gv_upper = None
        self.gJ_upper = None
        self.Ge_upper = None
        self.Fev_upper = None

    def _set_distribution_from_upper_state_distribution(self):
        pass

    def upper_state_dataframe(self):
        return self.upper_state._dataFrame()

    def upper_state_ravel_distribution(self):
        return self.upper_state.ravel_distribution()

    def set_upper_state_distribution(self, _distribution):
        self.upper_state.set_distribution(_distribution)
        self._set_distribution_from_upper_state_distribution()

    def set_upper_state_distribution_error(self, _distri_error):
        self.upper_state.set_distribution_error(_distri_error)

    def set_maxwell_upper_state_distribution(self, *, Tvib, Trot):
        self.upper_state.set_maxwell_distribution(Tvib=Tvib, Trot=Trot)
        self._set_distribution_from_upper_state_distribution()

    def set_double_maxwell_upper_state_distribution(self, *, Tvib, Trot_cold, Trot_hot, hot_ratio):
        self.upper_state.set_double_maxwell_distribution(Tvib=Tvib, Trot_cold=Trot_cold,
                                                         Trot_hot=Trot_hot, hot_ratio=hot_ratio)
        self._set_distribution_from_upper_state_distribution()

    # def set_distribution_by_upper_state(self):
    #     pass

    def fit_temperatures(self, *, wavelength_exp, intensity_exp,
                         init_value_dict, fit_kws=dict(ftol=1e-6, xtol=1e-6)):
        self._set_delta_wv_spr_matrix(wv_exp=wavelength_exp,
                                      fwhm=dict(Gaussian=init_value_dict["fwhm_g"],
                                                Lorentzian=init_value_dict["fwhm_l"]))

        def fit_func(x, Tvib, Trot, fwhm_g, fwhm_l):
            # print(f"{Tvib}, {Trot}, {fwhm_g}, {fwhm_l}")
            self.convolve_slit_func(fwhm=dict(Gaussian=fwhm_g, Lorentzian=fwhm_l),
                                    slit_func="Voigt")
            self.set_maxwell_upper_state_distribution(Tvib=Tvib, Trot=Trot)
            self.set_intensity()
            return self.get_intensity_exp(normalized=True)

        spectra_fit_model = Model(fit_func)
        # set params
        params = spectra_fit_model.make_params()
        value_range_dict = dict(Tvib=(300, 10000),
                                Trot=(300, 10000),
                                fwhm_g=(0, 1),
                                fwhm_l=(0, 1))
        for _key in ["Tvib", "Trot", "fwhm_g", "fwhm_l"]:
            params[_key].set(value=init_value_dict[_key])
            params[_key].set(min=value_range_dict[_key][0],
                             max=value_range_dict[_key][1])
            params[_key].set(vary=True)
        # To fit it.
        sim_result = spectra_fit_model.fit(intensity_exp,
                                           params=params,
                                           method="least_squares",
                                           fit_kws=fit_kws,
                                           x=wavelength_exp)
        return sim_result

    # def set_maxwell_distribution(self, *, Tvib, Trot):
    #     r"""
    #     Set the distribution to a maxwell one.
    #     Parameters
    #     ----------
    #     Tvib : K
    #     Trot : K
    #     """
    #     vib_distribution = self.gv_upper * np.exp(-self.Ge_upper * self.WNcm2K / Tvib)
    #     rot_distribution = self.gJ_upper * np.exp(-self.Fev_upper * self.WNcm2K / Trot)
    #     self.distribution = vib_distribution * rot_distribution

    # def set_double_temperature_distribution(self, *, Tvib, Trot_cold, Trot_hot, hot_ratio):
    #     warm_part = self.gJ_upper * np.exp(-self.Fev_upper * self.WNcm2K / Trot_hot)
    #     cold_part = self.gJ_upper * np.exp(-self.Fev_upper * self.WNcm2K / Trot_cold)
    #     warm_part = warm_part / warm_part.sum()
    #     cold_part = cold_part / cold_part.sum()
    #     rot_distribution = hot_ratio * warm_part + (1 - hot_ratio) * cold_part
    #     vib_distribution = self.gv_upper * np.exp(-self.Ge_upper * self.WNcm2K / Tvib)
    #     self.distribution = vib_distribution * rot_distribution

    @staticmethod
    def honl_london_factor(*, band, branch):
        if band in ('CO(B-A)',):
            if branch == 'R':
                return lambda J: J / 2
            elif branch == 'Q':
                return lambda J: (2 * J + 1) / 2
            elif branch == 'P':
                return lambda J: (J + 1) / 2
            else:
                raise Exception('branch is error')


class N2Spectra(MoleculeSpectra):
    r"""

    """

    def __init__(self, *, band, v_upper, v_lower, J_max=90):
        super().__init__()
        if band == 'C-B':
            self.upper_state = N2State(state='C', v_upper=v_upper)
        self.band = band
        self.v_upper = v_upper
        self.v_lower = v_lower
        self.J_max = J_max
        self._set_coefs()

    def _set_coefs(self):
        r""" wavelength, wavenumber, emission_coefs"""
        dir_path = os.path.dirname(os.path.realpath(__file__))
        assert self.band == 'C-B'
        wv_lg_path = dir_path + r"\N2(C-B)\{v0}-{v1}\wavelength_vac.dat".format(v0=self.v_upper,
                                                                                v1=self.v_lower)
        wv_nm_path = dir_path + r"\N2(C-B)\{v0}-{v1}\wavenumber.dat".format(v0=self.v_upper,
                                                                            v1=self.v_lower)
        self.wave_length = self.wavelength_vac2air(np.loadtxt(wv_lg_path))[:self.J_max, :]
        self.wave_number = np.loadtxt(wv_nm_path)[:self.J_max, :]
        self.wave_length = np.nan_to_num(self.wave_length)
        self.wave_number = np.nan_to_num(self.wave_number)
        self._set_emission_coefficients()

    def _set_emission_coefficients(self):
        gJ_upper = 2 * np.tile(np.arange(self.J_max), (27, 1)).transpose() + 1
        sjj = self.honl_london_factor()[:self.J_max, :]
        fc_factor = self.frank_condon_factor()
        self.emission_coefficients = self.wave_number ** 3 * fc_factor * sjj / gJ_upper

    def _set_distribution_from_upper_state_distribution(self):
        self.distribution = np.zeros((self.J_max, 27))
        branch_list = [0, 0, 0, 1, 1, 1, 2, 2, 2,
                       0, 0, 0, 1, 1, 1, 2, 2, 2,
                       0, 0, 0, 1, 1, 1, 2, 2, 2]
        for i, _branch in enumerate(branch_list):
            self.distribution[:, i] = self.upper_state.distribution[_branch]

    def honl_london_factor(self):
        if self.band == "C-B":
            dir_path = os.path.dirname(os.path.realpath(__file__))
            _path = dir_path + r"\N2(C-B)\honl_factor.dat"
            return np.loadtxt(_path)
        else:
            raise Exception("The band is not supported.")

    def frank_condon_factor(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        _path = dir_path + r"\N2(C-B)\frank_condon_factor.dat"
        factor_matrix = np.loadtxt(_path)
        return factor_matrix[self.v_lower, self.v_upper]


class N2pSpectra(MoleculeSpectra):
    r"""
    (N", J")    lower state
    (N', J')    upper state
    The first line:
    branch :    P1,     P2,     R1,     R2,     P_Q12,  R_Q21
        J" :    0.5,    0.5,    0.5,    0.5,    0.5,    0.5
        N" :    0,      1,      0,      1,      1,      0
        J' :    -0.5,   -0.5,   1.5,    1.5,    0.5,    0.5
        N' :    -1,     0,      1,      2,      0,      1
    """

    def __init__(self, *, band, v_upper, v_lower):
        super().__init__()
        self.upper_state = N2pState(state='B', v_upper=v_upper)
        self._set_coefs(band=band, v_upper=v_upper, v_lower=v_lower)

    # def _set_distribution_from_upper_state_distribution(self):

    def _set_coefs(self, *, band, v_upper, v_lower):
        assert band == "B-X"
        _sign = "{v0}-{v1}".format(v0=v_upper, v1=v_lower)

        def read_coefficients_from_csv(file_name):
            return np.genfromtxt(file_name, delimiter=',', skip_header=5)[:, 1:]

        dir_path = os.path.dirname(os.path.realpath(__file__))
        wvlg_path = dir_path + r"\N2+(B-X)\{v2v}\line_position_air.csv".format(v2v=_sign)
        wvnm_path = dir_path + r"\N2+(B-X)\{v2v}\line_position_cm-1.csv".format(v2v=_sign)
        ec_path = dir_path + r"\N2+(B-X)\{v2v}\emission_coefficients.csv".format(v2v=_sign)
        _chosen_branch = [0, 1, 4, 5, 8, 9]  # not all data from lifbase are validated.
        self.wave_length = read_coefficients_from_csv(wvlg_path)[:, _chosen_branch] / 10
        self.wave_number = read_coefficients_from_csv(wvnm_path)[:, _chosen_branch]
        self.emission_coefficients = read_coefficients_from_csv(ec_path)[:, _chosen_branch]

    def _set_distribution_from_upper_state_distribution(self):
        self.distribution = np.zeros((85, 6))
        self.distribution[1:, 0] = self.upper_state.distribution[0, 0:84]  # P1
        self.distribution[1:, 1] = self.upper_state.distribution[1, 1:85]  # P2
        self.distribution[0:, 2] = self.upper_state.distribution[0, 1:86]  # R1
        self.distribution[0:, 3] = self.upper_state.distribution[1, 2:87]  # R2
        self.distribution[0:, 4] = self.upper_state.distribution[0, 0:85]  # P_Q12
        self.distribution[0:, 5] = self.upper_state.distribution[1, 1:86]  # R_Q21


class OHSpectra(MoleculeSpectra):
    r"""
    branch : P1, P2, Q1, Q2, R1, R2, O12, Q12, P12, R21, Q21, S21
              1,  2,  1,  1,  1,  1,   2,   1,   1,   1,   1,   1
    F1 : J = N + 1/2
    F2 : J = N - 1/2
    branch
        P1 : dN = dJ = -1
        R1 : dN = dJ = 1
        Q1 : dN = dJ = 0
        P21 : dN = 0, dJ = -1
        Q21 : dN = 1, dJ = 0
        R21 : dN = 2, dJ = 1

    A-X_1/2
        P2 : dN = dJ = -1
        R2 : dN = dJ = 1
        Q2 : dN = dJ = 0
        R12 : dN =  0, dJ = 1
        Q12 : dN = -1, dJ = 0
        P12 : dN = -2, dJ = -1

    """
    _BRANCH_SEQ = ['P1', 'P2', 'Q1', 'Q2', 'R1', 'R2', 'O12', 'Q12', 'P12', 'R21', 'Q21', 'S21']

    def __init__(self, *, band, v_upper, v_lower):
        super().__init__()
        self.upper_state = OHState(state="A", v_upper=v_upper)
        self._set_coefs(band=band, v_upper=v_upper, v_lower=v_lower)

    def line_intensity(self, *, branch):
        assert branch in self._BRANCH_SEQ
        branch_index = [i for i, j in enumerate(self._BRANCH_SEQ) if j == branch][0]
        return self.wave_length[:, branch_index], self.intensity[:, branch_index]

    def _set_coefs(self, *, band, v_upper, v_lower):
        assert band == 'A-X'
        # assert (v_upper, v_lower) in ((0, 0), (1, 0), (1, 1))
        _sign = '{v0}-{v1}'.format(v0=v_upper, v1=v_lower)

        def read_coefficients_from_csv(file_name):
            return np.genfromtxt(file_name, delimiter=',', skip_header=4)[:, 1:]

        dir_path = os.path.dirname(os.path.realpath(__file__))
        wvlg_path = dir_path + r"\OH(A-X)\{v2v}\line_position_air.csv".format(v2v=_sign)
        wvnm_path = dir_path + r"\OH(A-X)\{v2v}\line_position_cm-1.csv".format(v2v=_sign)
        ec_path = dir_path + r"\OH(A-X)\{v2v}\emission_coefficients.csv".format(v2v=_sign)
        self.wave_length = read_coefficients_from_csv(wvlg_path) / 10
        self.wave_number = read_coefficients_from_csv(wvnm_path)
        self.emission_coefficients = read_coefficients_from_csv(ec_path)
        # self._set_Ge(v_upper)

    def _set_distribution_from_upper_state_distribution(self):
        N_max = self.upper_state.N_max
        self.distribution = np.zeros((N_max - 1, 12))
        self.distribution[:, 0] = self.upper_state.distribution[0, 0:N_max - 1]  # P1
        self.distribution[1:, 1] = self.upper_state.distribution[1, 1:N_max - 1]  # P2
        self.distribution[:, 2] = self.upper_state.distribution[0, 1:N_max]  # Q1
        self.distribution[:, 3] = self.upper_state.distribution[1, 1:N_max]  # Q2
        self.distribution[:, 4] = self.upper_state.distribution[0, 2:N_max + 1]  # R1
        self.distribution[:, 5] = self.upper_state.distribution[1, 2:N_max + 1]  # R2
        self.distribution[1:, 6] = self.upper_state.distribution[0, 0:N_max - 2]  # O12
        self.distribution[:, 7] = self.upper_state.distribution[0, 1:N_max]  # Q12
        self.distribution[:, 8] = self.upper_state.distribution[0, 0:N_max - 1]  # P12
        self.distribution[:, 9] = self.upper_state.distribution[1, 2:N_max + 1]  # R21
        self.distribution[:, 10] = self.upper_state.distribution[1, 1:N_max]  # Q21
        self.distribution[:-1, 11] = self.upper_state.distribution[1, 3:]  # S21


class COSpectra(MoleculeSpectra):
    __FRANK_CONDON_FACTORS = np.array([[0.08898, 0.18159, 0.21056, 0.18339, 0.13399, 0.08706],
                                       [0.25053, 0.17569, 0.03039, 0.00420, 0.05214, 0.09553]])

    def __init__(self, *, band, v_upper, v_lower):
        super().__init__()
        self.v_upper = v_upper
        self.v_lower = v_lower
        self._set_coefs(band=band)

    def _set_coefs(self, *, band):
        r"""
        Branches :
            P, Q, R
        """
        assert band == 'B-A'
        J_max = 40
        self.J_lower = np.tile(np.arange(1, J_max + 1), (3, 1)).transpose()
        self.J_upper = self.J_lower + np.tile([-1, 0, 1], (J_max, 1))
        self.gJ_upper = 2 * self.J_upper + 1
        self.gJ_lower = 2 * self.J_lower + 1
        self.wave_number = MoleculeState('CO(B)').term(self.v_upper, self.J_upper) - \
                           MoleculeState('CO(A)').term(self.v_lower, self.J_lower)
        self.wave_length = 1e7 / self.wave_number
        self._set_Ge()
        self._set_Fev()
        self._set_emission_coefficients()

    def _set_Ge(self):
        self.gv_upper = np.ones_like(self.wave_length)
        self.Ge_upper = np.ones_like(self.wave_length) * \
                        MoleculeState('CO(B)').Ge_term(v=self.v_upper)

    def _set_Fev(self):
        self.Fev_upper = MoleculeState('CO(B)').Fev_term(self.v_upper, self.J_upper)

    def _set_emission_coefficients(self):
        sjj = np.zeros_like(self.J_upper)
        sjj[:, 0] = self.honl_london_factor(band='CO(B-A)', branch='P')(self.J_upper[:, 0])
        sjj[:, 1] = self.honl_london_factor(band='CO(B-A)', branch='Q')(self.J_upper[:, 1])
        sjj[:, 2] = self.honl_london_factor(band='CO(B-A)', branch='R')(self.J_upper[:, 2])
        self.emission_coefficients = self.wave_number ** 3 * self.frank_condon_factor() * sjj / (
                2 * self.J_upper + 1)

    def frank_condon_factor(self):
        r"""
        from "The band spectrum of carbon monoxide" book.
         """
        return self.__FRANK_CONDON_FACTORS[self.v_upper, self.v_lower]


class AddSpectra(MoleculeSpectra):

    def __init__(self, spec0, spec1):
        super().__init__()
        if isinstance(spec0, AddSpectra):
            self.specs = spec0.specs + [spec1]
        else:
            self.specs = [spec0, spec1]
        self.bands_num = len(self.specs)

    def set_intensity(self):
        for spec in self.specs:
            spec.set_intensity()

    def upper_state_dataframe(self):
        _df = pd.DataFrame(columns=self.specs[0].upper_state_dataframe().columns)
        for spec in self.specs:
            _df = _df.append(spec.upper_state_dataframe())
        return _df

    def upper_state_ravel_distribution(self):
        ravel_distribution = np.empty(0)
        for spec in self.specs:
            ravel_distribution = np.hstack((ravel_distribution,
                                            spec.upper_state_ravel_distribution()))
        return ravel_distribution

    def set_upper_state_distribution(self, _distributions):
        _from = 0
        _to = 0
        for i, spec in enumerate(self.specs):
            _from = _to
            _to += spec.upper_state.size
            spec.set_upper_state_distribution(_distributions[_from:_to])

    def set_upper_state_distribution_error(self, _distri_error):
        _from = 0
        _to = 0
        for i, spec in enumerate(self.specs):
            _from = _to
            _to += spec.upper_state.size
            spec.set_upper_state_distribution_error(_distri_error[_from:_to])

    def set_maxwell_upper_state_distribution(self, *, Tvib, Trot):
        for spec in self.specs:
            spec.set_maxwell_upper_state_distribution(Tvib=Tvib, Trot=Trot)

    def set_double_maxwell_upper_state_distribution(self, *, Tvib, Trot_cold, Trot_hot, hot_ratio):
        for spec in self.specs:
            spec.set_double_maxwell_upper_state_distribution(Tvib=Tvib, Trot_cold=Trot_cold,
                                                             Trot_hot=Trot_hot,
                                                             hot_ratio=hot_ratio)

    def get_extended_wavelength(self, *, waveLength_exp, fwhm, slit_func,
                                normalized=False, threshold=3):
        intensity_sim_on_wv_exp = np.zeros_like(waveLength_exp)
        for spec in self.specs:
            _, _intensity = spec.get_extended_wavelength(waveLength_exp=waveLength_exp,
                                                         fwhm=fwhm, slit_func=slit_func,
                                                         normalized=False, threshold=threshold)
            intensity_sim_on_wv_exp = intensity_sim_on_wv_exp + _intensity

        if normalized:
            self.normalized_factor = intensity_sim_on_wv_exp.max()
            return waveLength_exp, intensity_sim_on_wv_exp / self.normalized_factor
        return waveLength_exp, intensity_sim_on_wv_exp


# ----------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    wv_exp = np.linspace(280, 300, num=1024)
    spectra = OHSpectra(band='A-X', v_upper=1, v_lower=0)
    # spectra = N2Spectra(band='C-B', v_upper=0, v_lower=0)
    fwhm = dict(Gaussian=0.05, Lorentzian=0.05)

    spectra._set_delta_wv_spr_matrix(wv_exp=wv_exp, fwhm=fwhm)


    def test():
        spectra.convolve_slit_func(fwhm=fwhm, slit_func="Voigt")
        # spectra.set_maxwell_upper_state_distribution(Tvib=3000, Trot=2345)
        spectra.set_double_maxwell_upper_state_distribution(Tvib=3000, Trot_cold=400,
                                                            Trot_hot=5000, hot_ratio=0.5)
        spectra.set_intensity()
        return spectra.get_intensity_exp()


    intensity_exp = test() + np.random.random(wv_exp.size) * 0.01


    # plt.plot(wv_exp, intensity_exp)
    # --------------------------------------------------------------------------------------- #
    # r"""
    def fit_it():
        return spectra.fit_temperatures(wavelength_exp=wv_exp,
                                        intensity_exp=intensity_exp,
                                        init_value_dict=dict(Tvib=2000, Trot=3000,
                                                             fwhm_g=0.03, fwhm_l=0.07),
                                        fit_kws=dict(ftol=1e-10, xtol=1e-10))


    # sim_result = fit_it()
    # sim_result.plot_fit()
    # print(sim_result.fit_report())
    # """
    # ------------------------------------------------------------------------------------------- #
    # curve fit
    @tracer
    def fit_func(x, Tvib, Trot, fwhm_g, fwhm_l):
        print(f"{Tvib}, {Trot}, {fwhm_g}, {fwhm_l}")
        spectra.convolve_slit_func(fwhm=dict(Gaussian=fwhm_g, Lorentzian=fwhm_l),
                                   slit_func="Voigt")
        spectra.set_maxwell_upper_state_distribution(Tvib=Tvib, Trot=Trot)
        spectra.set_intensity()
        return spectra.get_intensity_exp()


    def fit_it_by_curve_fit():
        paras_fitted, pcov = curve_fit(fit_func, wv_exp, intensity_exp, bounds=(0, np.inf),
                                       p0=[2000, 3000, 0.03, 0.07],
                                       ftol=1e-10, xtol=1e-10)
        perr = np.sqrt(np.diag(pcov))
        return paras_fitted, perr


    @tracer
    def fit_func_by_distribution(x, *_distri):
        _distri_array = np.array(_distri)
        spectra.set_upper_state_distribution(_distri_array)
        spectra.set_intensity()
        return spectra.get_intensity_exp()


    # def fit_distribution():
    spectra.convolve_slit_func(fwhm=dict(Gaussian=0.05, Lorentzian=0.05), slit_func="Voigt")
    spectra.set_maxwell_upper_state_distribution(Tvib=3000, Trot=2000)
    spectra.upper_state.plot_reduced_distribution(new_figure=True)
    distri_guess = spectra.upper_state_ravel_distribution()
    distri_guess = distri_guess / distri_guess[0]
    _bounds = (np.zeros_like(distri_guess), np.inf * np.ones_like(distri_guess))
    _bounds[0][0] = distri_guess[0] * 0.99
    _bounds[1][0] = distri_guess[0]
    distri_fitted, pcov = curve_fit(fit_func_by_distribution,
                                    wv_exp, intensity_exp,
                                    bounds=_bounds,
                                    p0=distri_guess,
                                    ftol=1e-10)
    perr = np.sqrt(np.diag(pcov))
    spectra.set_upper_state_distribution(distri_fitted)
    spectra.set_upper_state_distribution_error(perr)
    spectra.upper_state.plot_distribution()
    spectra.upper_state.plot_reduced_distribution()
    # return distri_fitted, perr
