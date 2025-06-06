import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

# ---------- Load Cleaned Data ----------
cases = pd.read_csv('cleaned_covid_cases.csv', parse_dates=['date'])
vaccines = pd.read_csv('cleaned_vaccinations.csv', parse_dates=['date'])
testing = pd.read_csv('cleaned_testing_data.csv', parse_dates=['date'])

# ---------- Visualization 1: Line Chart - Daily Confirmed Cases by State ----------
cases['new_cases'] = cases.groupby('state')['confirmed'].diff()
line_data = cases[cases['state'].isin(['Maharashtra', 'Kerala', 'Delhi', 'Karnataka'])]  # example states
fig1 = px.line(line_data, x='date', y='new_cases', color='state', title='Daily Confirmed COVID-19 Cases by State')
fig1.show()

# ---------- Visualization 2: Bar Chart - Top 10 States by Total Vaccinations ----------
vax_total = vaccines.groupby('state')[['first_dose', 'second_dose']].max().reset_index()
vax_total['total'] = vax_total['first_dose'] + vax_total['second_dose']
vax_top10 = vax_total.sort_values(by='total', ascending=False).head(10)
fig2 = px.bar(vax_top10, x='state', y='total', title='Top 10 States by Total Vaccinations')
fig2.show()

# ---------- Visualization 3: Pie Chart - Gender-wise Vaccine Distribution ----------
gender_total = vaccines[['male', 'female', 'transgender']].sum()
fig3 = px.pie(values=gender_total, names=gender_total.index, title='Gender-wise Vaccine Distribution')
fig3.show()

# ---------- Visualization 4: Dual-Axis Chart - New Cases vs Vaccinations Over Time ----------
daily_cases = cases.groupby('date')['confirmed'].sum().diff().reset_index(name='new_cases')
daily_vax = vaccines.groupby('date')[['first_dose', 'second_dose']].sum().sum(axis=1).reset_index(name='total_vaccinations')
merged = pd.merge(daily_cases, daily_vax, on='date', how='inner')

fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=merged['date'], y=merged['new_cases'], mode='lines', name='New Cases'))
fig4.add_trace(go.Scatter(x=merged['date'], y=merged['total_vaccinations'], mode='lines', name='Vaccinations', yaxis='y2'))
fig4.update_layout(
    title='Daily New COVID-19 Cases vs. Vaccinations',
    yaxis=dict(title='New Cases'),
    yaxis2=dict(title='Vaccinations', overlaying='y', side='right')
)
fig4.show()
