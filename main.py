import pandas as pd

# ---------- Load Datasets ----------
cases = pd.read_csv('covid_19_india.csv')
vaccines = pd.read_csv('covid_vaccine_statewise.csv')
testing = pd.read_csv('StatewiseTestingDetails.csv')

# ---------- Clean COVID Case Data ----------
cases = cases.rename(columns={
    'Date': 'date',
    'State/UnionTerritory': 'state',
    'Cured': 'cured',
    'Deaths': 'deaths',
    'Confirmed': 'confirmed'
})
cases['date'] = pd.to_datetime(cases['date'], dayfirst=True)
cases = cases[['date', 'state', 'confirmed', 'cured', 'deaths']]

# ---------- Clean Vaccination Data ----------
vaccines = vaccines.rename(columns={
    'Updated On': 'date',
    'State': 'state',
    'Total Doses Administered': 'total_doses',
    'First Dose Administered': 'first_dose',
    'Second Dose Administered': 'second_dose',
    'Male (Doses Administered)': 'male',
    'Female (Doses Administered)': 'female',
    'Transgender (Doses Administered)': 'transgender'
})
vaccines['date'] = pd.to_datetime(vaccines['date'], dayfirst=True)
vaccines = vaccines[['date', 'state', 'total_doses', 'first_dose', 'second_dose', 'male', 'female', 'transgender']]

# Remove aggregate national rows if present
vaccines = vaccines[vaccines['state'] != 'India']

# Fill missing gender counts with 0
vaccines[['male', 'female', 'transgender']] = vaccines[['male', 'female', 'transgender']].fillna(0)

# ---------- Clean Testing Data ----------
testing = testing.rename(columns={
    'Date': 'date',
    'State': 'state',
    'TotalSamples': 'total_samples',
    'Negative': 'negative',
    'Positive': 'positive'
})
testing['date'] = pd.to_datetime(testing['date'], dayfirst=True)
testing = testing[['date', 'state', 'total_samples', 'negative', 'positive']]

# Fill missing numeric values with 0 or forward fill where appropriate
testing[['total_samples', 'negative', 'positive']] = testing[['total_samples', 'negative', 'positive']].fillna(method='ffill')

# ---------- Save Cleaned Data ----------
cases.to_csv('cleaned_covid_cases.csv', index=False)
vaccines.to_csv('cleaned_vaccinations.csv', index=False)
testing.to_csv('cleaned_testing_data.csv', index=False)

print("✅ Cleaned data saved successfully.")

# ---------- SQL SCHEMA ----------

# Table: covid_cases
# CREATE TABLE covid_cases (
#   date DATE,
#   state TEXT,
#   confirmed INT,
#   cured INT,
#   deaths INT
# );

# Table: vaccinations
# CREATE TABLE vaccinations (
#   date DATE,
#   state TEXT,
#   total_doses INT,
#   first_dose INT,
#   second_dose INT,
#   male INT,
#   female INT,
#   transgender INT
# );

# Table: testing
# CREATE TABLE testing (
#   date DATE,
#   state TEXT,
#   total_samples INT,
#   negative INT,
#   positive INT
# );

# ---------- SQL QUERIES ----------

# Total confirmed cases by state
# SELECT state, MAX(confirmed) AS total_cases
# FROM covid_cases
# GROUP BY state
# ORDER BY total_cases DESC;

# Daily new confirmed cases
# SELECT date, state,
#        confirmed - LAG(confirmed) OVER (PARTITION BY state ORDER BY date) AS new_cases
# FROM covid_cases;

# Vaccination progress by gender
# SELECT date, SUM(male) AS total_male, SUM(female) AS total_female
# FROM vaccinations
# GROUP BY date;

# Daily total vaccinations
# SELECT date, SUM(first_dose + second_dose) AS total_daily_vaccinated
# FROM vaccinations
# GROUP BY date
# ORDER BY date;

# Test positivity rate per state per day
# SELECT t.date, t.state,
#        c.confirmed, t.total_samples,
#        ROUND(CAST(c.confirmed AS FLOAT)/t.total_samples, 4) AS positivity_rate
# FROM covid_cases c
# JOIN testing t
#   ON c.date = t.date AND c.state = t.state
# WHERE t.total_samples > 0;

# Correlation proxy: lagged vaccination effect on new cases
# WITH daily_cases AS (
#   SELECT date, state,
#          confirmed - LAG(confirmed) OVER (PARTITION BY state ORDER BY date) AS new_cases
#   FROM covid_cases
# ),
# daily_vacc AS (
#   SELECT date, state, first_dose + second_dose AS total_vaccines
#   FROM vaccinations
# )
# SELECT d.date, d.state, d.new_cases, v.total_vaccines
# FROM daily_cases d
# JOIN daily_vacc v
#   ON d.date = v.date AND d.state = v.state
# WHERE d.new_cases IS NOT NULL;
