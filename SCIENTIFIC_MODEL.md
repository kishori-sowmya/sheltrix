# Scientific Thermal Model — Governing Equations & Assumptions

## Core Heat Balance Equation

The Tier-1 physics model solves the lumped capacitance transient heat balance over 24 hours:

$$C_{\text{total}} \frac{dT_{\text{indoor}}}{dt} = Q_{\text{solar}} + Q_{\text{cond}} + Q_{\text{vent}} \pm Q_{\text{PCM}} + Q_{\text{heating}}$$

Where:
- $C_{\text{total}} = C_{\text{air}} + C_{\text{floor}} + C_{\text{wall}}$ (J/K)
- $Q_{\text{solar}} = \text{SHGC} \cdot A_{\text{glazing}} \cdot I_{\text{solar}} \cdot \cos(\theta) + 0.05 \cdot \alpha_{\text{wall}} \cdot A_{\text{wall}} \cdot I_{\text{solar}}$ (W)
- $Q_{\text{cond}} = \frac{A_{\text{wall}}}{R_{\text{wall}}} (T_{\text{amb}} - T_{\text{indoor}}) + \frac{A_{\text{roof}}}{R_{\text{roof}}} (T_{\text{amb}} - T_{\text{indoor}}) + \frac{A_{\text{window}}}{R_{\text{window}}} (T_{\text{amb}} - T_{\text{indoor}}) + \frac{A_{\text{floor}}}{R_{\text{floor}}} (T_{\text{ground}} - T_{\text{indoor}})$ (W)
- $Q_{\text{vent}} = \dot{m}_{\text{air}} \cdot c_p \cdot (T_{\text{amb}} - T_{\text{indoor}})$ (W)

## Envelope Thermal Resistance ($R$-value)

$$R_{\text{wall}} = \frac{t_{\text{base}}}{k_{\text{base}}} + \frac{t_{\text{insulation}}}{k_{\text{insulation}}} + R_{\text{film}}$$

- $R_{\text{film}} = 0.13 \text{ m}^2\cdot\text{K/W}$ (combined interior and exterior air surface film resistance)

## PCM Latent Heat Storage Modeling

Phase Change Material (PCM) latent heat storage is integrated using a smeared enthalpy formulation near the melting temperature $T_{\text{melt}} = 21^\circ\text{C}$:

$$C_{\text{effective}} = C_{\text{total}} + \frac{H_{\text{latent}} \cdot m_{\text{pcm}}}{4.0}$$

when $|T_{\text{indoor}} - T_{\text{melt}}| \le 2.0^\circ\text{C}$.

## Numerical Spinup Sequence

To eliminate initial temperature condition artifacts, simulations run a **3-day diurnal warm-up spinup (72 hours)**, retaining outputs from the final 24-hour cycle.
