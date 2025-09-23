import numpy as np
from numpy import * # For access to all fundamental functions, constants etc.
def residual_1(z,state,params,Ts):
    """ RESIDUAL_1 Sequential residual generator for model 'LiU_ICE model'
    Causality: int

    Structurally sensitive to faults: fiml, fyw_af

    Example of basic usage:
    Let z be the observations matrix, each column corresponding to a known signal and Ts the sampling time,
    then the residual generator can be simulated by:

    r = np.zeros(N) # N number of data points
    state = {'wg_pos': wg_pos_0, 'm_af': m_af_0, 'T_af': T_af_0, 'm_c': m_c_0, 'T_c': T_c_0, 'T_im': T_im_0, 'm_ic': m_ic_0, 'T_ic': T_ic_0, 'm_im': m_im_0, 'm_em': m_em_0, 'T_em': T_em_0, 'm_t': m_t_0, 'T_t': T_t_0, 'omega_tc': omega_tc_0}
    for k,zk in enumerate(z):
        r[k], state = residual_1( zk, state, params, Ts )

    State is a dictionary with the keys: wg_pos, m_af, T_af, m_c, T_c, T_im, m_ic, T_ic, m_im, m_em, T_em, m_t, T_t, omega_tc

    File generated Wed Feb 26 12:02:28 2025
    """
    def ApproxInt(dx, x0, Ts):
        return x0 + Ts*dx

    def residual_1_core(z, state, params, Ts):
        # Parameters
        H_exhaust = params['H_exhaust']
        plin_exh = params['plin_exh']
        H_intercooler = params['H_intercooler']
        plin_intercooler = params['plin_intercooler']
        PIli_th = params['PIli_th']
        gamma_air = params['gamma_air']
        PIli_wg = params['PIli_wg']
        gamma_exh = params['gamma_exh']
        R_air = params['R_air']
        cp_air = params['cp_air']
        V_af = params['V_af']
        V_c = params['V_c']
        V_ic = params['V_ic']
        V_im = params['V_im']
        R_exh = params['R_exh']
        cp_exh = params['cp_exh']
        lambda_af = params['lambda_af']
        n_r = params['n_r']
        r_c = params['r_c']
        V_D = params['V_D']
        CEva_cool = params['CEva_cool']
        D_c = params['D_c']
        K1 = params['K1']
        K2 = params['K2']
        fix_gain = params['fix_gain']
        eta_cmax = params['eta_cmax']
        eta_cmin = params['eta_cmin']
        T_std = params['T_std']
        k1 = params['k1']
        k2 = params['k2']
        cp_eg = params['cp_eg']
        gamma_eg = params['gamma_eg']
        tau_wg = params['tau_wg']
        h_tot = params['h_tot']
        Amax = params['Amax']
        A_0 = params['A_0']
        A_1 = params['A_1']
        A_2 = params['A_2']
        J_tc = params['J_tc']
        xi_fric_tc = params['xi_fric_tc']
        V_em = params['V_em']
        V_t = params['V_t']
        p_std = params['p_std']
        a1 = params['a1']
        a2 = params['a2']
        a3 = params['a3']
        a4 = params['a4']
        a5 = params['a5']
        Q_c11 = params['Q_c11']
        Q_c12 = params['Q_c12']
        Q_c22 = params['Q_c22']
        Cd = params['Cd']
        PI_cmax = params['PI_cmax']
        cv_exh = params['cv_exh']
        cv_air = params['cv_air']
        W_ccorrmax = params['W_ccorrmax']
        A_em = params['A_em']
        K_t = params['K_t']
        T0 = params['T0']
        Cic_1 = params['Cic_1']
        Cic_2 = params['Cic_2']
        Cic_3 = params['Cic_3']
        TOL = params['TOL']

        # Known signals
        y_T_ic = z[1]
        y_W_af = z[3]
        y_omega_e = z[4]
        y_alpha_th = z[5]
        y_u_wg = z[6]
        y_wfc = z[7]
        y_T_amb = z[8]
        y_p_amb = z[9]

        # Initialize state variables
        wg_pos = state['wg_pos']
        m_af = state['m_af']
        T_af = state['T_af']
        m_c = state['m_c']
        T_c = state['T_c']
        T_im = state['T_im']
        m_ic = state['m_ic']
        T_ic = state['T_ic']
        m_im = state['m_im']
        m_em = state['m_em']
        T_em = state['T_em']
        m_t = state['m_t']
        T_t = state['T_t']
        omega_tc = state['omega_tc']

        # Residual generator body
        W_fc = y_wfc # e92
        p_amb = y_p_amb # e94
        T_amb = y_T_amb # e93
        u_wg = y_u_wg # e91
        dwgdt_pos = (u_wg - wg_pos)/tau_wg # e79
        omega_e = y_omega_e # e89
        alpha_th = y_alpha_th # e90
        Aeff_wg = Amax*Cd*wg_pos # e78
        Aeff_th = A_0 + 43*pi*A_1*alpha_th/9000 + pi*A_1/45 + 1849*pi**2*A_2*alpha_th**2/81000000 + 43*pi**2*A_2*alpha_th/202500 + pi**2*A_2/2025 # e42
        W_af = y_W_af # e88
        T_in = CEva_cool - CEva_cool/lambda_af + T_im # e44
        U_c = D_c*omega_tc/2 # e66
        p_t = R_exh*T_t*m_t/V_t # e36
        p_em = R_exh*T_em*m_em/V_em # e31
        PI_wg = PI_wg_fun(p_t, p_em, gamma_air) # e7
        PSIli_wg = PSIli_wg_fun(PI_wg, PIli_wg, gamma_exh) # e8
        Tflow_wg = Tflow_wg_fun(p_t, p_em, T_em, T_t) # e10
        PI_t = PI_t_fun(p_t, p_em) # e75
        W_t = W_t_fun(p_em, k1, PI_t, k2, T_em) # e70
        dh_is = T_em*cp_exh*(1 - PI_t**((gamma_exh - 1)/gamma_exh)) # e73
        eta_t = max_fun(TOL, min_fun(1 - TOL, (1000*sqrt(T_em)*W_t*a_eta_t_fun(omega_tc) + p_em*b_eta_t_fun(omega_tc))/(dh_is*p_em))) # e72
        T_tout = T_em*(PI_t**((gamma_eg - 1)/gamma_eg)*eta_t - eta_t + 1) # e74
        Tq_t = Tq_t_fun(gamma_eg, cp_eg, eta_t, W_t, T_em, PI_t, omega_tc) # e71
        p_im = R_air*T_im*m_im/V_im # e26
        C_eta_vol = 1.0e-12*a1*max_fun(p_im, 25000)**4 + 1.0e-9*a2*max_fun(p_im, 25000)**3 + 1.0e-6*a3*max_fun(p_im, 25000)**2 + 0.001*a4*max_fun(p_im, 25000) + a5 # e46
        eta_vol = C_eta_vol*T_im*(r_c - (p_em/p_im)**(1/gamma_exh))/(T_in*(r_c - 1)) # e45
        W_ac = V_D*eta_vol*omega_e*p_im/(2*pi*R_air*T_im*n_r) # e43
        W_e = W_ac + W_fc # e57
        T_e = K_t*W_e + T0 # e58
        T_ti = (T_amb*exp(4*A_em*h_tot/(W_e*cp_exh)) - T_amb + T_e)*exp(-4*A_em*h_tot/(W_e*cp_exh)) # e59
        T_imcr = T_ic # e6
        p_c = R_air*T_c*m_c/V_c # e16
        p_af = R_air*T_af*m_af/V_af # e11
        PI_cnolim = p_c/p_af # e65
        PI_c = PI_c_fun(p_c, p_af) # e64
        PSI_c = 2*T_af*cp_air*(PI_c**((gamma_air - 1)/gamma_air) - 1)/U_c**2 # e63
        PHI_model = PHI_model_fun(K1, K2, PSI_c) # e62
        W_c = W_c_fun(PI_cnolim, p_af, U_c, D_c, T_af, R_air, PHI_model, fix_gain) # e61
        dmdt_af = W_af - W_c # e12
        dTdt_af = (R_air*(-T_af*W_c + T_amb*W_af) + W_af*cv_air*(-T_af + T_amb))/(cv_air*m_af) # e13
        W_ccorr = W_c*p_std*sqrt(T_af/T_std)/p_af # e69
        eta_c = eta_c_fun(eta_cmax, eta_cmin, W_ccorr, W_ccorrmax, PI_c, PI_cmax, Q_c11, Q_c12, Q_c22) # e68
        T_cout = T_af*(PI_c**((gamma_air - 1)/gamma_air) + eta_c - 1)/eta_c # e67
        Tq_c = W_c*cp_air*(-T_af + T_cout)/omega_tc # e60
        domegadt_tc = (-Tq_c + Tq_t - omega_tc*xi_fric_tc)/J_tc # e83
        W_wg = Aeff_wg*PSIli_wg*p_em/sqrt(R_exh*Tflow_wg) # e9
        T_turb = (T_tout*W_t + Tflow_wg*W_wg)/(W_t + W_wg) # e76
        W_twg = W_t + W_wg # e77
        dmdt_em = W_e - W_twg # e32
        dTdt_em = (R_exh*(-T_em*W_twg + T_ti*W_e) + W_e*cv_exh*(-T_em + T_ti))/(cv_exh*m_em) # e33
        p_ic = R_air*T_ic*m_ic/V_ic # e21
        W_ic = W_ic_fun(p_c, p_ic, plin_intercooler, H_intercooler, T_c) # e1
        dmdt_c = W_c - W_ic # e17
        dTdt_c = (R_air*(-T_c*W_ic + T_cout*W_c) + W_c*cv_air*(-T_c + T_cout))/(cv_air*m_c) # e18
        T_fwd_flow_ic = max_fun(T_amb, T_c - max_fun(TOL, -Cic_1*T_amb + Cic_1*T_c - Cic_2*T_amb**2/2 + Cic_2*T_c**2/2 - Cic_3*T_amb*W_ic + Cic_3*T_c*W_ic)) # e41
        PSI_th = PSI_th_fun(p_im, p_ic, gamma_air, PIli_th) # e4
        W_th = Aeff_th*PSI_th*p_ic/sqrt(R_air*T_ic) # e5
        dTdt_im = (R_air*(-T_im*W_ac + T_imcr*W_th) + W_th*cv_air*(-T_im + T_imcr))/(cv_air*m_im) # e28
        dmdt_ic = W_ic - W_th # e22
        dTdt_ic = (R_air*(T_fwd_flow_ic*W_ic - T_ic*W_th) + W_ic*cv_air*(T_fwd_flow_ic - T_ic))/(cv_air*m_ic) # e23
        dmdt_im = -W_ac + W_th # e27
        W_es = W_es_fun(p_amb, p_t, plin_exh, H_exhaust, T_t) # e3
        dmdt_t = -W_es + W_twg # e37
        dTdt_t = (R_exh*(-T_t*W_es + T_turb*W_twg) + W_twg*cv_exh*(-T_t + T_turb))/(cv_exh*m_t) # e38
         
        r = T_ic - y_T_ic # e86

        # Update integrator variables
        wg_pos = ApproxInt(dwgdt_pos, state['wg_pos'], Ts) # e80
        m_af = ApproxInt(dmdt_af, state['m_af'], Ts) # e14
        T_af = ApproxInt(dTdt_af, state['T_af'], Ts) # e15
        m_c = ApproxInt(dmdt_c, state['m_c'], Ts) # e19
        T_c = ApproxInt(dTdt_c, state['T_c'], Ts) # e20
        T_im = ApproxInt(dTdt_im, state['T_im'], Ts) # e30
        m_ic = ApproxInt(dmdt_ic, state['m_ic'], Ts) # e24
        T_ic = ApproxInt(dTdt_ic, state['T_ic'], Ts) # e25
        m_im = ApproxInt(dmdt_im, state['m_im'], Ts) # e29
        m_em = ApproxInt(dmdt_em, state['m_em'], Ts) # e34
        T_em = ApproxInt(dTdt_em, state['T_em'], Ts) # e35
        m_t = ApproxInt(dmdt_t, state['m_t'], Ts) # e39
        T_t = ApproxInt(dTdt_t, state['T_t'], Ts) # e40
        omega_tc = ApproxInt(domegadt_tc, state['omega_tc'], Ts) # e84

        # Update state variables
        state['wg_pos'] = wg_pos
        state['m_af'] = m_af
        state['T_af'] = T_af
        state['m_c'] = m_c
        state['T_c'] = T_c
        state['T_im'] = T_im
        state['m_ic'] = m_ic
        state['T_ic'] = T_ic
        state['m_im'] = m_im
        state['m_em'] = m_em
        state['T_em'] = T_em
        state['m_t'] = m_t
        state['T_t'] = T_t
        state['omega_tc'] = omega_tc

        return (r, state)

    return residual_1_core(z, state, params, Ts)
