from oggm.core.flowline import FlowlineModel
from oggm.core.flowline import k_calving_law
from oggm.core.flowline import (RectangularBedFlowline,
                                TrapezoidalBedFlowline,
                                MixedBedFlowline)
import numpy as np
import oggm.cfg as cfg
from oggm.exceptions import InvalidParamsError, InvalidWorkflowError
from oggm.cfg import G, GAUSSIAN_KERNEL
from oggm import utils
from scipy.linalg import solve_banded


class SemiImplicitModel(FlowlineModel):
    """Semi implicit flowline model.

    It solves the same equation as the FluxBasedModel, but the ice flux q is
    implemented as q^t = D^t * (ds/dx)^(t+1).

    It supports only a single flowline (no tributaries) with bed shapes
    rectangular, trapezoidal or a mixture of both.
    """

    def __init__(self, flowlines, mb_model=None, y0=0., glen_a=None, fs=0.,
                 inplace=False, fixed_dt=None, cfl_number=0.5, min_dt=None,
                 do_calving=None, calving_k=None, calving_law=k_calving_law,
                 calving_use_limiter=None, calving_limiter_frac=None,
                 water_level=1000,
                 **kwargs):
        """Instantiate the model.

        Parameters
        ----------
        flowlines : list
            the glacier flowlines
        mb_model : MassBalanceModel
            the mass balance model
        y0 : int
            initial year of the simulation
        glen_a : float
            Glen's creep parameter
        fs : float
            Oerlemans sliding parameter
        inplace : bool
            wether to make a copy of the flowline objects for the run
            setting to True implies that your objects will be modified at run
            time by the model (can help to spare memory)
        fixed_dt : float
            set to a value (in seconds) to prevent adaptive time-stepping.
        cfl_number : float
            For adaptive time stepping (the default), dt is chosen from the
            CFL criterion (dt = cfl_number * dx^2 / max(D/w)).
            Can be set to higher values compared to the FluxBasedModel.
            Default is 0.5, but need further investigation.
        min_dt : float
            Defaults to cfg.PARAMS['cfl_min_dt'].
            At high velocities, time steps can become very small and your
            model might run very slowly. In production, it might be useful to
            set a limit below which the model will just error.
        do_kcalving : bool
            switch on the k-calving parameterisation. Ignored if not a
            tidewater glacier. Use the option from PARAMS per default
        calving_law : func
             option to use another calving law. This is a temporary workaround
             to test other calving laws, and the system might be improved in
             future OGGM versions.
        calving_k : float
            the calving proportionality constant (units: yr-1). Use the
            one from PARAMS per default
        calving_use_limiter : bool
            whether to switch on the calving limiter on the parameterisation
            makes the calving fronts thicker but the model is more stable
        calving_limiter_frac : float
            limit the front slope to a fraction of the calving front.
            "3" means 1/3. Setting it to 0 limits the slope to sea-level.
        water_level : float
            the water level. It should be zero m a.s.l, but:
            - sometimes the frontal elevation is unrealistically high (or low).
            - lake terminating glaciers
            - other uncertainties
            The default is 0. For lake terminating glaciers,
            it is inferred from PARAMS['free_board_lake_terminating'].
            The best way to set the water level for real glaciers is to use
            the same as used for the inversion (this is what
            `flowline_model_run` does for you)
        kwargs : dict
            Further keyword arguments for FlowlineModel

        """

        super(SemiImplicitModel, self).__init__(flowlines, mb_model=mb_model,
                                                y0=y0, glen_a=glen_a, fs=fs,
                                                inplace=inplace, **kwargs)

        if len(self.fls) > 1:
            raise ValueError('Implicit model does not work with '
                             'tributaries.')

        # convert pure RectangularBedFlowline to TrapezoidalBedFlowline with
        # lambda = 0
        if isinstance(self.fls[0], RectangularBedFlowline):
            self.fls[0] = TrapezoidalBedFlowline(
                line=self.fls[-1].line, dx=self.fls[-1].dx,
                map_dx=self.fls[-1].map_dx, surface_h=self.fls[-1].surface_h,
                bed_h=self.fls[-1].bed_h, widths=self.fls[-1].widths,
                lambdas=0, rgi_id=self.fls[-1].rgi_id,
                water_level=self.fls[-1].water_level, gdir=None)

        if isinstance(self.fls[0], MixedBedFlowline):
            if ~np.all(self.fls[0].is_trapezoid):
                raise ValueError('Implicit model only works with a pure '
                                 'trapezoidal flowline! But different lambdas '
                                 'along the flowline possible (lambda=0 is'
                                 'rectangular).')
        elif not isinstance(self.fls[0], TrapezoidalBedFlowline):
            raise ValueError('Implicit model only works with a pure '
                             'trapezoidal flowline! But different lambdas '
                             'along the flowline possible (lambda=0 is'
                             'rectangular).')

        # if cfg.PARAMS['use_kcalving_for_run']:
        #     raise NotImplementedError("Calving is not implemented in the"
        #                               "SemiImplicitModel! Set "
        #                               "cfg.PARAMS['use_kcalving_for_run'] = "
        #                               "False or use a FluxBasedModel.")

        self.fixed_dt = fixed_dt
        if min_dt is None:
            min_dt = cfg.PARAMS['cfl_min_dt']
        self.min_dt = min_dt

        if cfl_number is None:
            cfl_number = cfg.PARAMS['cfl_number']
        if cfl_number < 0.1:
            raise InvalidParamsError("For the SemiImplicitModel you can use "
                                     "cfl numbers in the order of 0.1 - 0.5 "
                                     f"(you set {cfl_number}).")
        self.cfl_number = cfl_number

        # Calving params
        if do_calving is None:
            do_calving = cfg.PARAMS['use_kcalving_for_run']
        self.calving_law = calving_law
        self.do_calving = do_calving
        if calving_k is None:
            calving_k = cfg.PARAMS['calving_k']
        self.calving_k = calving_k / cfg.SEC_IN_YEAR
        if calving_use_limiter is None:
            calving_use_limiter = cfg.PARAMS['calving_use_limiter']
        self.calving_use_limiter = calving_use_limiter
        if calving_limiter_frac is None:
            calving_limiter_frac = cfg.PARAMS['calving_limiter_frac']
        if calving_limiter_frac > 0:
            raise NotImplementedError('calving limiter other than 0 not '
                                      'implemented yet')
        self.calving_limiter_frac = calving_limiter_frac

        # Storage variables for diagnostics
        self.calving_m3_since_y0 = 0
        self.calving_rate_myr = 0

        # Special output
        self._surf_vel_fac = (self.glen_n + 2) / (self.glen_n + 1)

        # optim
        nx = self.fls[-1].nx
        bed_h_exp = np.concatenate(([self.fls[-1].bed_h[0]],
                                    self.fls[-1].bed_h,
                                    [self.fls[-1].bed_h[-1]]))
        self.dbed_h_exp_dx = ((bed_h_exp[1:] - bed_h_exp[:-1]) /
                              self.fls[0].dx_meter)
        self.d_stag = [np.zeros(nx + 1)]
        self.d_matrix_banded = np.zeros((3, nx))
        w0 = self.fls[0]._w0_m
        self.w0_stag = (w0[0:-1] + w0[1:]) / 2
        self.rhog = (self.rho * G) ** self.glen_n

        # variables needed for the calculation of some diagnostics, this
        # calculations are done with @property, because they are not computed
        # on the fly during the dynamic run as in FluxBasedModel
        self._u_stag = [np.zeros(nx + 1)]
        self._flux_stag = [np.zeros(nx + 1)]
        self._slope_stag = [np.zeros(nx + 1)]
        self._thick_stag = [np.zeros(nx + 1)]
        self._section_stag = [np.zeros(nx + 1)]

    @property
    def slope_stag(self):
        slope_stag = self._slope_stag[0]

        surface_h = self.fls[0].surface_h
        dx = self.fls[0].dx_meter

        slope_stag[0] = 0
        slope_stag[1:-1] = (surface_h[0:-1] - surface_h[1:]) / dx
        slope_stag[-1] = slope_stag[-2]

        return [slope_stag]

    @property
    def thick_stag(self):
        thick_stag = self._thick_stag[0]

        thick = self.fls[0].thick

        thick_stag[1:-1] = (thick[0:-1] + thick[1:]) / 2.
        thick_stag[[0, -1]] = thick[[0, -1]]

        return [thick_stag]

    @property
    def section_stag(self):
        section_stag = self._section_stag[0]

        section = self.fls[0].section

        section_stag[1:-1] = (section[0:-1] + section[1:]) / 2.
        section_stag[[0, -1]] = section[[0, -1]]

        return [section_stag]

    @property
    def u_stag(self):
        u_stag = self._u_stag[0]

        slope_stag = self.slope_stag[0]
        thick_stag = self.thick_stag[0]
        N = self.glen_n
        rhog = self.rhog

        rhogh = rhog * slope_stag ** N

        u_stag[:] = ((thick_stag**(N+1)) * self._fd * rhogh +
                     (thick_stag**(N-1)) * self.fs * rhogh)

        return [u_stag]

    @property
    def flux_stag(self):
        flux_stag = self._flux_stag[0]

        section_stag = self.section_stag[0]
        u_stag = self.u_stag[0]

        flux_stag[:] = u_stag * section_stag

        return [flux_stag]

    def step(self, dt):
        """Advance one step."""

        # Just a check to avoid useless computations
        if dt <= 0:
            raise InvalidParamsError('dt needs to be strictly positive')

        # read out variables from current flowline
        fl = self.fls[0]
        dx = fl.dx_meter
        width = fl.widths_m
        thick = fl.thick
        surface_h = fl.surface_h

        # some variables needed later
        N = self.glen_n
        rhog = self.rhog

        # Not sure if we need this if statement, it was in the
        # Fluxgate model so I have implemented here:
        if self.do_calving and self.calving_use_limiter:
            # We lower the max possible ice deformation
            # by clipping the surface slope here. It is completely
            # arbitrary but reduces ice deformation at the calving front.
            # I think that in essence, it is also partly
            # a "calving process", because this ice deformation must
            # be less at the calving front. The result is that calving
            # front "free boards" are quite high.
            # Note that 0 is arbitrary, it could be any value below SL
            surface_h = utils.clip_min(surface_h, self.water_level)

        # calculate staggered variables for diffusivity
        width_stag = (width[0:-1] + width[1:]) / 2
        w0_stag = self.w0_stag
        thick_stag = (thick[0:-1] + thick[1:]) / 2.
        dsdx_stag = (surface_h[1:] - surface_h[0:-1]) / dx

        # calculate diffusivity
        # boundary condition d_stag_0 = d_stag_end = 0
        d_stag = self.d_stag[0]
        d_stag[1:-1] = ((self._fd * thick_stag ** (N + 2) +
                         self.fs * thick_stag ** N) * rhog *
                        (w0_stag + width_stag) / 2 *
                        np.abs(dsdx_stag) ** (N - 1))

        # Impose zero-flux boundary condition at glacier terminus (only for land-terminating)
        if self.do_calving:
            d_stag[-1] = 0

        # Time step
        if self.fixed_dt:
            # change only if step dt is larger than the chosen dt
            if self.fixed_dt < dt:
                dt = self.fixed_dt
        else:
            # use stability criterion dt <= dx^2 / max(D/w) * cfl_number
            divisor = np.max(np.abs(d_stag[1:-1] / width_stag))
            if divisor > cfg.FLOAT_EPS:
                cfl_dt = self.cfl_number * dx ** 2 / divisor
            else:
                cfl_dt = dt

            if cfl_dt < dt:
                dt = cfl_dt
                if cfl_dt < self.min_dt:
                    raise RuntimeError(
                        'CFL error: required time step smaller '
                        'than the minimum allowed: '
                        '{:.1f}s vs {:.1f}s. Happening at '
                        'simulation year {:.1f}, fl_id {}, '
                        'bin_id {} and max_D {:.3f} m2 yr-1.'
                        ''.format(cfl_dt, self.min_dt, self.yr, 0,
                                  np.argmax(np.abs(d_stag)),
                                  divisor * cfg.SEC_IN_YEAR))

        # calculate diagonals of Amat
        d0 = dt / dx ** 2 * (d_stag[:-1] + d_stag[1:]) / width
        dm = - dt / dx ** 2 * d_stag[:-1] / width
        dp = - dt / dx ** 2 * d_stag[1:] / width

        # construct banded form of the matrix, which is used during solving
        # (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_banded.html)
        # original matrix:
        # d_matrix = (np.diag(dp[:-1], 1) +
        #             np.diag(np.ones(len(d0)) + d0) +
        #             np.diag(dm[1:], -1))
        self.d_matrix_banded[0, 1:] = dp[:-1]
        self.d_matrix_banded[1, :] = np.ones(len(d0)) + d0
        self.d_matrix_banded[2, :-1] = dm[1:]

        # correction term for glacier bed (original equation is an equation for
        # the surface height s, which is transformed in an equation for h, as
        # s = h + b the term below comes from the '- b'
        b_corr = - d_stag * self.dbed_h_exp_dx
        smb = self.get_mb(surface_h, self.yr, fl_id=0, fls=self.fls)

        # prepare rhs
        rhs = thick + smb * dt + dt / width * (b_corr[:-1] - b_corr[1:]) / dx

        # solve matrix and update flowline thickness
        thick_new = utils.clip_min(
            solve_banded((1, 1), self.d_matrix_banded, rhs),
            0)
        fl.thick = thick_new

        # Retreat-based calving parametrisation (adapted from FluxBasedModel)
        # We need here less "if" statements as we dont deal with tributaries
        self.calving_rate_myr = 0.

        if self.do_calving and fl.has_ice():

            indices = np.nonzero((fl.surface_h > self.water_level) &
                                 (fl.thick > 0))[0]

            if indices.size == 0:
                # No ice above water level -> skip calving this step
                return

            # Identify last glacier grid cell with ice above water level
            last_above_wl = indices[-1]

            # Proceed only if terminus bed is below water (marine-terminating)
            if fl.bed_h[last_above_wl] <= self.water_level:

                # OK, we're really calving
                section = fl.section

                # Calving law
                q_calving = self.calving_law(self, fl, last_above_wl)

                # Add to the bucket and the diagnostics
                fl.calving_bucket_m3 += q_calving * dt
                self.calving_m3_since_y0 += q_calving * dt
                self.calving_rate_myr = (q_calving / section[last_above_wl] *
                                         cfg.SEC_IN_YEAR)

                # See if we have ice below sea-water to clean out first
                below_sl = (fl.surface_h < self.water_level) & (fl.thick > 0)
                to_remove = np.sum(section[below_sl]) * fl.dx_meter
                if 0 < to_remove < fl.calving_bucket_m3:
                    # This is easy, we remove everything
                    section[below_sl] = 0
                    fl.calving_bucket_m3 -= to_remove
                elif to_remove > 0:
                    # the conditions below I had to change them
                    # to prevent index out-of-bounds errors
                    # when updating the ice thickness near the calving front
                    # NEEDS checking!
                    section[below_sl] = 0
                    if (last_above_wl + 1) < len(section):
                        section[last_above_wl + 1] = ((to_remove - fl.calving_bucket_m3)
                                                      / fl.dx_meter)
                    else:
                        section[last_above_wl] = max(
                            section[last_above_wl] - (to_remove - fl.calving_bucket_m3) / fl.dx_meter, 0)
                    fl.calving_bucket_m3 = 0

                # The rest of the bucket might calve an entire grid point (or more?)
                vol_last = section[last_above_wl] * fl.dx_meter
                while fl.calving_bucket_m3 > vol_last:
                    fl.calving_bucket_m3 -= vol_last
                    section[last_above_wl] = 0

                    # OK check if we need to continue (unlikely)
                    last_above_wl -= 1

                    if last_above_wl < 0:
                        # All ice is removed; no further calving possible
                        fl.calving_bucket_m3 = 0
                        break

                    vol_last = section[last_above_wl] * fl.dx_meter

                # We update the glacier with our changes
                fl.section = section

        # Next step
        self.t += dt

        return dt

    def get_diagnostics(self, fl_id=-1):
        """Obtain model diagnostics in a pandas DataFrame.

        Parameters
        ----------
        fl_id : int
            the index of the flowline of interest, from 0 to n_flowline-1.
            Default is to take the last (main) one

        Returns
        -------
        a pandas DataFrame, which index is distance along flowline (m). Units:
            - surface_h, bed_h, ice_tick, section_width: m
            - section_area: m2
            - slope: -
            - ice_flux, tributary_flux: m3 of *ice* per second
            - ice_velocity: m per second (depth-section integrated)
            - surface_ice_velocity: m per second (corrected for surface - simplified)
        """
        import pandas as pd

        fl = self.fls[fl_id]
        nx = fl.nx

        df = pd.DataFrame(index=fl.dx_meter * np.arange(nx))
        df.index.name = 'distance_along_flowline'
        df['surface_h'] = fl.surface_h
        df['bed_h'] = fl.bed_h
        df['ice_thick'] = fl.thick
        df['section_width'] = fl.widths_m
        df['section_area'] = fl.section

        # Staggered
        var = self.slope_stag[fl_id]
        df['slope'] = (var[1:nx+1] + var[:nx])/2
        var = self.flux_stag[fl_id]
        df['ice_flux'] = (var[1:nx+1] + var[:nx])/2
        var = self.u_stag[fl_id]
        df['ice_velocity'] = (var[1:nx+1] + var[:nx])/2
        df['surface_ice_velocity'] = df['ice_velocity'] * self._surf_vel_fac
        df['calving_flux'] = self.calving_m3_since_y0 / self.t  # Average calving flux
        df['calving_rate'] = self.calving_rate_myr

        return df