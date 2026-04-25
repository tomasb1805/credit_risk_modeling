-- Raw layer tables
CREATE TABLE IF NOT EXISTS raw.loan_applications (
    loan_id             SERIAL          PRIMARY KEY,
    customer_id         BIGINT          NOT NULL,
    application_date    DATE            NOT NULL,
    loan_amount         NUMERIC(12, 2)  NOT NULL,
    interest_rate       NUMERIC(5, 2)   NOT NULL,
    term_months         SMALLINT        NOT NULL,
    income              NUMERIC(12, 2)  NOT NULL,
    dtiratio            NUMERIC(5, 4)   NOT NULL,
    employment_length   SMALLINT,
    credit_history      SMALLINT,
    default_status      SMALLINT        NOT NULL CHECK (default_status IN (0, 1)),
    loan_grade          VARCHAR(2),
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS raw.bank_accounts (
    account_id          SERIAL          PRIMARY KEY,
    customer_id         BIGINT          NOT NULL,
    account_type        VARCHAR(50)     NOT NULL,
    current_balance     NUMERIC(12, 2)  NOT NULL,
    available_balance   NUMERIC(12, 2)  NOT NULL,
    overdraft_limit     NUMERIC(12, 2)  NOT NULL DEFAULT 0,
    last_updated        TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS raw.transactions (
    transaction_id      SERIAL          PRIMARY KEY,
    account_id          INT             NOT NULL REFERENCES raw.bank_accounts(account_id),
    transaction_date    TIMESTAMPTZ     NOT NULL,
    amount              NUMERIC(12, 2)  NOT NULL,
    direction           VARCHAR(6)      NOT NULL CHECK (direction IN ('debit', 'credit')),
    category            VARCHAR(100),
    description         TEXT,
    running_balance     NUMERIC(12, 2)  NOT NULL
);

-- Analytics layer tables
CREATE TABLE IF NOT EXISTS analytics.behavioral_features (
    customer_id             BIGINT          PRIMARY KEY,
    monthly_income_mean     NUMERIC(12, 2),
    expense_income_ratio    NUMERIC(5, 4),
    overdraft_days_30d      SMALLINT,
    cash_withdrawal_freq    SMALLINT,
    savings_trend_90d       NUMERIC(5, 4),
    last_calculated         TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS analytics.risk_scores (
    score_id                SERIAL          PRIMARY KEY,
    loan_id                 INT             NOT NULL REFERENCES raw.loan_applications(loan_id),
    customer_id             BIGINT          NOT NULL,
    pd_score                NUMERIC(6, 5)   NOT NULL,
    risk_band               VARCHAR(10)     NOT NULL CHECK (risk_band IN ('low', 'medium', 'high')),
    decision                VARCHAR(10)     NOT NULL CHECK (decision IN ('approve', 'decline', 'review')),
    feature_importance_json JSONB,
    scored_at               TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);
