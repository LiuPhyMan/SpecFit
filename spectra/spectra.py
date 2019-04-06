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
from copy import deepcopy
from matplotlib import pyplot as plt
from .voigt import voigt_pseudo


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
        else:
            return self.we * (v + .5) - self.wexe * (v + .5) ** 2

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
        else:
            return

    def term(self, v, J):
        return self.Te + self.Ge_term(v) + self.Fev_term(v, J)


class Spectra(object):
    WNcm2K = 1.4387773538277204

    def __init__(self):
        super().__init__()
        self.wave_number = None
        self.wave_length = None
        self.emission_coefficients = None
        self.distribution = None
        self.intensity = None
        self.normalized_factor = None

    # ------------------------------------------------------------------------------------------- #
    def set_distribution(self):
        pass

    def set_intensity(self):
        self.intensity = self.distribution * self.emission_coefficients * self.wave_number
        self.intensity = self.intensity * 1.9864458241717582e-25  # multiply hc

    # ------------------------------------------------------------------------------------------- #
    def get_extended_wavelength(self, *, waveLength_exp,
                                fwhm, slit_func, wavelength_range=None,
                                normalized=False, threshold=3):
        r"""
        Convolve the slit function on the experiment wavelength.
        Threshold :
            Gaussian :
                threshold : 2.5
                exp(-4*log(2)*2.5**2) = 2.98e-08
            Lorentzian :
                threshold : 5e2
                1/(4*5e2**2+1) = 1.0e-06
        """
        # --------------------------------------------------------------------------------------- #
        #   _chosen :
        #       the boolean to choose the absolute wavelength position in the range.
        #   wv_exp_chosen :
        #       the boolean to choose the exp_wavelength in the range.
        if wavelength_range is None:
            _chosen = np.ones_like(self.wave_length) > 0
            wv_exp_chosen = np.ones_like(waveLength_exp) > 0
        else:
            _range_added = 3 * (fwhm['Gaussian'] + fwhm['Lorentzian'])
            _chosen = np.logical_and((wavelength_range[0] - 3 * _range_added) < self.wave_length,
                                     self.wave_length < (wavelength_range[1] + 3 * _range_added))
            wv_exp_chosen = np.logical_and(wavelength_range[0] < waveLength_exp,
                                           wavelength_range[1] > waveLength_exp)
        # --------------------------------------------------------------------------------------- #
        #   wavelength_in_range :
        #       absolute wavelength in the range.
        #   intensity_in_range :
        #       absolute intensity in the range.
        wavelength_in_range = self.wave_length[_chosen]
        intensity_in_range = self.intensity[_chosen]
        wv_exp_in_range = waveLength_exp[wv_exp_chosen]
        #   If the absolute wavelength are all out of the range.
        if not _chosen.any():
            return wv_exp_in_range, np.zeros_like(wv_exp_in_range)
        # --------------------------------------------------------------------------------------- #
        #   Format the matrix of the delta wavelength. Convolve the slit function.
        delta_wv = (wv_exp_in_range[np.newaxis].transpose() - wavelength_in_range)
        normal_intens_extended = self.evolve_slit_func(delta_wv, fwhm, slit_func, threshold)
        intensity_on_wv_exp = normal_intens_extended.dot(intensity_in_range)
        #   Return the wavelength and the simulated intensity in the range.
        if normalized:
            self.normalized_factor = intensity_on_wv_exp.max()
            return wv_exp_in_range, intensity_on_wv_exp / intensity_on_wv_exp.max()
        return wv_exp_in_range, intensity_on_wv_exp

    # ------------------------------------------------------------------------------------------- #
    def evolve_slit_func(self, delta_wv, fwhm, slit_func, threshold):
        r"""
        Returns the intensity matrix based on the delta wavelength and the slit function.
        """
        if slit_func == 'Gaussian':
            _fwhm = fwhm['Gaussian']
            delta_x = delta_wv / _fwhm
            _where = np.logical_and(-threshold < delta_x, delta_x < threshold)
            if not _where.any():
                return np.zeros_like(delta_wv)
            intens_matrix = np.zeros_like(delta_wv)
            _ = np.exp(-4 * np.log(2) * delta_x ** 2, out=intens_matrix, where=_where)
            return intens_matrix

        elif slit_func == 'Lorentzian':
            _fwhm = fwhm['Lorentzian']
            delta_x = delta_wv / _fwhm
            _where = np.logical_and(-threshold < delta_x, delta_x < threshold)
            if not _where.any():
                return np.zeros_like(delta_wv)
            intens_matrix = 2 / (4 * delta_x ** 2 + 1)
            return intens_matrix

        elif slit_func == 'Voigt':
            fG = fwhm['Gaussian']
            fL = fwhm['Lorentzian']
            # if fwhm_G / fwhm_L < 1e-6:
            #     return self.evolve_slit_func(delta_wv, fwhm, 'Lorentzian', threshold)
            # fwhm_V = convolve_to_voigt(fwhm_G=fwhm_G, fwhm_L=fwhm_L)
            # # print(delta_x.size)
            # # print(_where[_where == True].size)
            # sigma = fwhm_G / 2 / math.sqrt(2 * math.log(2))
            # gamma = fwhm_L / 2
            # temp = 0j * np.zeros_like(delta_wv)
            # z = (delta_wv + 1j * gamma) / sigma / math.sqrt(2)
            # y = np.real(Y) / special.wofz(1j * gamma / sigma / math.sqrt(2)).real
            # return y
            # ----------------------------------------------------------------------------------- #
            # psedo_voigt
            # base on voigt defination on wiki.
            # fG, fL = fwhm['Gaussian'], fwhm['Lorentzian']
            # _fwhm = fG ** 5 + 2.69269 * fG ** 4 * fL + 2.42843 * fG ** 3 * fL ** 2 + \
            #         4.47163 * fG ** 2 * fL ** 3 + 0.07842 * fG * fL ** 4 + fL ** 5
            # _fwhm = _fwhm ** (1 / 5)
            #
            # delta_x = delta_wv / _fwhm
            # _where = np.logical_and(-threshold < delta_x, delta_x < threshold)
            # if not _where.any():
            #     return np.zeros_like(delta_wv)
            #
            # a = 1.36603 * (fL / _fwhm) - 0.47719 * (fL / _fwhm) ** 2 + 0.11116 * (fL / _fwhm)
            # ** 3
            # L_part = fL / 2 / math.pi / (delta_wv ** 2 + (fL / 2) ** 2)
            # sigma = fG / 2 / math.sqrt(2 * math.log(2))
            # temp = np.zeros_like(delta_wv)
            # _ = np.exp(-delta_wv**2/2/sigma**2, out=temp, where=_where)
            # G_part = temp / sigma / math.sqrt(2 * math.pi)
            # TODO Check the voigt pseudo function.
            return voigt_pseudo(delta_wv, fG, fL)
        else:
            raise Exception("The slit function '{s}' is error.".format(s=slit_func))

    # ------------------------------------------------------------------------------------------- #
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

    def set_state_distribution(self):
        pass

    def set_maxwell_distribution(self, *, Tvib, Trot):
        r"""
        Set the distribution to a maxwell one.
        Parameters
        ----------
        Tvib : K
        Trot : K
        """
        vib_distribution = self.gv_upper * np.exp(-self.Ge_upper * self.WNcm2K / Tvib)
        rot_distribution = self.gJ_upper * np.exp(-self.Fev_upper * self.WNcm2K / Trot)
        self.distribution = vib_distribution * rot_distribution

    def set_double_temperature_distribution(self, *, Tvib, Trot_cold, Trot_hot, hot_ratio):
        warm_part = self.gJ_upper * np.exp(-self.Fev_upper * self.WNcm2K / Trot_hot)
        cold_part = self.gJ_upper * np.exp(-self.Fev_upper * self.WNcm2K / Trot_cold)
        warm_part = warm_part / warm_part.sum()
        cold_part = cold_part / cold_part.sum()
        rot_distribution = hot_ratio * warm_part + (1 - hot_ratio) * cold_part
        vib_distribution = self.gv_upper * np.exp(-self.Ge_upper * self.WNcm2K / Tvib)
        self.distribution = vib_distribution * rot_distribution

    def ravel_coefficients(self):
        self.wave_number = self.wave_number.ravel()
        self.wave_length = self.wave_length.ravel()
        self.emission_coefficients = self.emission_coefficients.ravel()
        self.gv_upper = self.gv_upper.ravel()
        self.Ge_upper = self.Ge_upper.ravel()
        self.gJ_upper = self.gJ_upper.ravel()
        self.Fev_upper = self.Fev_upper.ravel()

    def narrow_range(self, *, _range):
        _chosen = np.logical_and(_range[0] < self.wave_length, self.wave_length < _range[1])
        assert _chosen.any()
        _spectra_copy = deepcopy(self)
        _spectra_copy.wave_number = self.wave_number[_chosen]
        _spectra_copy.wave_length = self.wave_length[_chosen]
        _spectra_copy.emission_coefficients = self.emission_coefficients[_chosen]
        _spectra_copy.distribution = self.distribution[_chosen]
        _spectra_copy.intensity = self.intensity[_chosen]
        ##
        _spectra_copy.gv_upper = self.gv_upper[_chosen]
        _spectra_copy.Ge_upper = self.Ge_upper[_chosen]
        _spectra_copy.gJ_upper = self.gJ_upper[_chosen]
        _spectra_copy.Fev_upper = self.Fev_upper[_chosen]
        return _spectra_copy

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

    def __init__(self, *, band, v_upper, v_lower):
        super().__init__()
        self.band = band
        self.v_upper = v_upper
        self.v_lower = v_lower
        self._set_coefs()

    def _set_coefs(self):
        r""" wavelength, wavenumber, emission_coefs"""
        dir_path = os.path.dirname(os.path.realpath(__file__))
        assert self.band == 'C-B'
        wv_path = dir_path + r"\N2(C-B)\{v0}-{v1}\wavelength_vac.dat".format(v0=self.v_upper,
                                                                             v1=self.v_lower)
        self.wave_length = np.loadtxt(wv_path)
        self.wave_number = np.divide(1e7, self.wave_length,
                                     out=np.zeros_like(self.wave_length),
                                     where=(self.wave_length > 1))

        self.J_upper = np.tile(np.arange(0, 171), (9, 1)).transpose()
        self.gJ_upper = 2 * self.J_upper + 1
        self._set_Ge()
        self._set_Fev()
        self._set_emission_coefficients()

    def _set_Ge(self):
        assert self.band == "C-B"
        self.gv_upper = np.ones_like(self.wave_length)
        self.Ge_upper = np.ones_like(self.wave_length) * MoleculeState("N2(C)").Ge_term(
            self.v_upper)

    def _set_Fev(self):
        # wvnm_path = dir_path + r"\OH(A-X)\{v2v}\line_position_cm-1.csv".format(v2v=_sign)
        # ec_path = dir_path + r"\OH(A-X)\{v2v}\emission_coefficients.csv".format(v2v=_sign)
        # if self.v_upper
        dir_path = os.path.dirname(os.path.realpath(__file__))
        _path = dir_path + r"\N2(C-B)\Fev_{v}.dat".format(v=self.v_upper)
        self.Fev_upper = np.loadtxt(_path)

    def _set_emission_coefficients(self):
        sjj = self.honl_london_factor()
        self.emission_coefficients = self.wave_number ** 3 * self.frank_condon_factor() * sjj / self.gJ_upper

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
        self._set_coefs(band=band, v_upper=v_upper, v_lower=v_lower)

    def line_intensity(self, *, branch):
        assert branch in self._BRANCH_SEQ
        branch_index = [i for i, j in enumerate(self._BRANCH_SEQ) if j == branch][0]
        return self.wave_length[:, branch_index], self.intensity[:, branch_index]

    def set_state_distribution(self, *, F1_distri, F2_distri):
        assert F1_distri.size == 42
        assert F2_distri.size == 42
        self.distribution = np.zeros((40, 12))
        self.distribution[:, 0] = F1_distri[0:40]  # P1
        self.distribution[:, 1] = F2_distri[0:40]  # P2
        self.distribution[:, 2] = F1_distri[1:41]  # Q1
        self.distribution[:, 3] = F2_distri[1:41]  # Q2
        self.distribution[:, 4] = F1_distri[2:42]  # R1
        self.distribution[:, 5] = F2_distri[2:42]  # R2
        self.distribution[1:, 6] = F1_distri[0:39]  # O12
        self.distribution[:, 7] = F1_distri[1:41]  # Q12
        self.distribution[:, 8] = F1_distri[0:40]  # P12
        self.distribution[:, 9] = F2_distri[2:42]  # R21
        self.distribution[:, 10] = F2_distri[1:41]  # Q21
        self.distribution[:-1, 11] = F2_distri[3:]  # S21

    def set_distribution(self, _distribution):
        self.distribution = _distribution

    def get_level_params(self, params):
        F1_part = np.hstack((params[:, 0],
                             params[-1, 2],
                             params[-1, 4]))
        F2_part = np.hstack((params[:, 1],
                             params[-1, 3],
                             params[-1, 5]))
        return F1_part, F2_part

    def get_level_gJ_upper(self):
        F1_gJ = np.hstack((self.gJ_upper[:, 0],
                           self.gJ_upper[-1, 2],
                           self.gJ_upper[-1, 4]))
        F2_gJ = np.hstack((self.gJ_upper[:, 1],
                           self.gJ_upper[-1, 3],
                           self.gJ_upper[-1, 5]))
        return dict(F1_gJ_upper=F1_gJ,
                    F2_gJ_upper=F2_gJ)

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
        self._set_Ge(v_upper)
        self._set_Fev(v_upper)

    def _set_Ge(self, v_upper):
        self.gv_upper = np.ones_like(self.wave_length)
        self.Ge_upper = np.ones_like(self.wave_length) * MoleculeState('OH(A)').Ge_term(v_upper)

    def _set_Fev(self, v_upper):
        N_max = self.wave_length.shape[0]
        self.N_lower = np.tile(np.arange(1, N_max + 1), (12, 1)).transpose()
        self.J_lower = self.N_lower + .5 * np.tile([1, -1, 1, -1, 1, -1, -1, -1, -1, 1, 1, 1],
                                                   (N_max, 1))
        self.N_upper = self.N_lower + np.tile([-1, -1, 0, 0, 1, 1, -2, 0, -1, 1, 0, 2], (N_max, 1))
        self.J_upper = self.N_upper + .5 * np.tile([1, -1, 1, -1, 1, -1, 1, 1, 1, -1, -1, -1],
                                                   (N_max, 1))
        self.gJ_upper = 2 * self.J_upper + 1
        # 1962 The ultraviolet bands of OH. Table 2
        OH_A_consts = [[16.961, 0.00204, 0.1122],
                       [16.129, 0.00203, 0.1056],
                       [15.287, 0.00208, 0.0997],
                       [14.422, 0.00206, 0.0980]]
        Bv, Dv, gama = OH_A_consts[v_upper]
        ##
        self.Fev_upper = np.zeros_like(self.gJ_upper)
        for i_branch in range(12):
            _N = self.N_upper[:, i_branch]
            _average_term = Bv * _N * (_N + 1) - Dv * _N ** 2 * (_N + 1) ** 2
            if i_branch in (0, 2, 4, 6, 7, 8):  # F1
                temp = _average_term + gama * (_N + .5)
            else:  # F2
                temp = _average_term - gama * (_N + .5)
            self.Fev_upper[:, i_branch] = temp


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
        combine_coefs = lambda a, b: np.hstack((a.ravel(), b.ravel()))
        self.wave_number = combine_coefs(spec0.wave_number, spec1.wave_number)
        self.wave_length = combine_coefs(spec0.wave_length, spec1.wave_length)
        self.emission_coefficients = combine_coefs(spec0.emission_coefficients,
                                                   spec1.emission_coefficients)
        self.gv_upper = combine_coefs(spec0.gv_upper, spec1.gv_upper)
        self.gJ_upper = combine_coefs(spec0.gJ_upper, spec1.gJ_upper)
        self.Ge_upper = combine_coefs(spec0.Ge_upper, spec1.Ge_upper)
        self.Fev_upper = combine_coefs(spec0.Fev_upper, spec1.Fev_upper)


# ----------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    r"""
    oh_0 = OHSpectra(band='A-X', v_upper=0, v_lower=0)
    oh_1 = OHSpectra(band='A-X', v_upper=1, v_lower=0)
    oh_2 = OHSpectra(band='A-X', v_upper=1, v_lower=1)
    oh = AddSpectra(spec0=oh_0, spec1=oh_1)
    oh = AddSpectra(spec0=oh, spec1=oh_2)
    oh.set_maxwell_distribution(Tvib=5000, Trot=2000)
    oh.set_intensity()
    wv = np.linspace(300, 320, num=3000)
    wv_in_range, intensity = oh.get_extended_wavelength(wv_range=(250, 370),
                                                        waveLength_exp=wv,
                                                        slit_func='Gaussian',
                                                        fwhm={'Gaussian': 0.05,
                                                              'Lorentzian': 0.05},
                                                        normalized=True)
    # plt.plot(wv, intensity)
    # result.plot_fit() # xdata = wv
    # ydata = intensity
    #
    # popt, pcov = curve_fit(func, xdata, ydata,
    #                        p0=(6000, 3000, 0.1, 0.0, 0.1),
    #                        bounds=(np.array([4000, 2000, 0, 0, 0.3]),
    #                                np.array([8000, 3000, 1, 1, 0.3])),
    #                        method='trf',
    #                        xtol=1e-6)
    # print(popt)
    # plt.plot(xdata, ydata)
    # plt.plot(xdata, func(xdata, *popt))
    """
    oh = OHSpectra(band='A-X', v_upper=0, v_lower=0)


    def func(x, Tvib, Trot_hot, Trot_cold, hot_ratio, fwhm_g, fwhm_l):
        oh.set_double_temperature_distribution(Tvib=Tvib,
                                               Trot_cold=Trot_cold, Trot_hot=Trot_hot,
                                               hot_ratio=hot_ratio)
        oh.set_intensity()
        _, intens = oh.get_extended_wavelength(waveLength_exp=x,
                                               slit_func='Voigt',
                                               fwhm={'Gaussian': fwhm_g, 'Lorentzian': fwhm_l},
                                               normalized=True)
        return intens


    # oh.set_maxwell_distribution(Tvib=3000, Trot=3000)
    oh.set_double_temperature_distribution(Tvib=3000,
                                           Trot_cold=3000, Trot_hot=10000, hot_ratio=0.6)
    oh.set_intensity()
    wv = np.linspace(300, 320, num=5000)
    wv_in_range, intensity = oh.get_extended_wavelength(wavelength_range=(250, 370),
                                                        waveLength_exp=wv,
                                                        slit_func='Voigt',
                                                        fwhm={'Gaussian': 0.001,
                                                              'Lorentzian': 0.03},
                                                        normalized=True)

    intensity = intensity + 0.01 * np.random.rand(*intensity.shape)
    from lmfit import Model

    spectra_fit_model = Model(func)
    params = spectra_fit_model.make_params()
    params['Tvib'].set(value=6000, vary=True)
    params['Trot_cold'].set(value=3000, vary=True)
    params['Trot_hot'].set(value=3000, vary=True)
    params['hot_ratio'].set(value=0.5, vary=True)
    params['fwhm_g'].set(value=0.001, vary=True, min=0, max=1)
    params['fwhm_l'].set(value=0.03, vary=True, min=0, max=1)

    result = spectra_fit_model.fit(intensity, params=params,
                                   # method='least_squares',
                                   fit_kws=dict(ftol=1e-12),
                                   x=wv_in_range)
    print(result.fit_report())

    plt.plot(wv_in_range, intensity)
