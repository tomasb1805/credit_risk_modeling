SELECT
    l.*,
    a.avg_monthly_inflow,
    a.income_stability_ratio,
    a.spend_to_income_ratio,
    a.months_net_negative_6m,
    a.cash_withdrawal_ratio
FROM raw.loan_applications l
LEFT JOIN analytics.persona_features as a
    ON l.persona_name = a.persona_name;
