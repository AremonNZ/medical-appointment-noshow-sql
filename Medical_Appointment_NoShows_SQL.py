#!/usr/bin/env python
# coding: utf-8

# # 🏥 Medical Appointment No-Shows — SQL & Data Analysis
# **Dataset:** Medical Appointment No-Shows — 110,527 real patient appointments, public
# health clinics in Vitória, Brazil (April–June 2016).
# Source: Kaggle, "Medical Appointment No Shows" (J. Hoppen).
# **Goal:** Use SQL to explore patient behaviour patterns and identify factors that
# predict appointment no-shows.
# **Author:** Arman Arabkhani | AUT Data Science

# ## 1. Setup — Loading Data into SQLite

# We load the real dataset (one row per appointment) into a SQLite in-memory
# database, so all analysis below runs as genuine SQL queries against real data.

import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette(['steelblue'])

df = pd.read_csv('noshowappointments.csv')

# --- Clean ---
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
df['Wait_Days'] = (df['AppointmentDay'].dt.normalize() - df['ScheduledDay'].dt.normalize()).dt.days
df = df.rename(columns={'Hipertension': 'Hypertension', 'No-show': 'No_show'})
df['No_show'] = (df['No_show'] == 'Yes').astype(int)

before = len(df)
df = df[(df['Age'] >= 0) & (df['Age'] <= 110)]
df = df[df['Wait_Days'] >= 0]
print(f"Dropped {before - len(df)} invalid rows (negative age or negative wait time)")

conn = sqlite3.connect(':memory:')
df.to_sql('appointments', conn, index=False, if_exists='replace')
print(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
print(f"No-show rate: {df['No_show'].mean():.1%}")


# ## 2. Basic SQL Exploration

def sql(query, conn=conn):
    """Helper to run SQL and return a DataFrame."""
    return pd.read_sql_query(query, conn)

sql("SELECT * FROM appointments LIMIT 5")

sql("""
SELECT
    COUNT(*)                            AS total_appointments,
    SUM(No_show)                        AS total_no_shows,
    ROUND(AVG(No_show) * 100, 1)        AS no_show_rate_pct,
    ROUND(AVG(Age), 1)                  AS avg_age,
    ROUND(AVG(Wait_Days), 1)            AS avg_wait_days
FROM appointments
""")


# ## 3. No-Show Rate by Gender

result = sql("""
SELECT
    Gender,
    COUNT(*)                            AS total,
    SUM(No_show)                        AS no_shows,
    ROUND(AVG(No_show) * 100, 1)        AS no_show_rate_pct
FROM appointments
GROUP BY Gender
ORDER BY no_show_rate_pct DESC
""")
print(result.to_string(index=False))

plt.figure(figsize=(7, 4))
plt.bar(result['Gender'], result['no_show_rate_pct'], color=['steelblue', 'coral'], edgecolor='white', width=0.5)
plt.title('No-Show Rate by Gender', fontsize=13)
plt.xlabel('Gender'); plt.ylabel('No-Show Rate (%)')
for i, (_, row) in enumerate(result.iterrows()):
    plt.text(i, row['no_show_rate_pct'] + 0.3, f"{row['no_show_rate_pct']}%", ha='center', fontsize=11)
plt.tight_layout(); plt.savefig('chart_gender.png'); plt.close()


# ## 4. No-Show Rate by Age Group

result = sql("""
SELECT
    CASE
        WHEN Age < 18  THEN 'Under 18'
        WHEN Age < 35  THEN '18-34'
        WHEN Age < 50  THEN '35-49'
        WHEN Age < 65  THEN '50-64'
        ELSE '65+'
    END AS age_group,
    COUNT(*)                            AS total,
    SUM(No_show)                        AS no_shows,
    ROUND(AVG(No_show) * 100, 1)        AS no_show_rate_pct
FROM appointments
GROUP BY age_group
ORDER BY no_show_rate_pct DESC
""")
print(result.to_string(index=False))

order = ['Under 18', '18-34', '35-49', '50-64', '65+']
result_sorted = result.set_index('age_group').reindex(order).reset_index()
plt.figure(figsize=(9, 4))
plt.bar(result_sorted['age_group'], result_sorted['no_show_rate_pct'], color='steelblue', edgecolor='white')
plt.title('No-Show Rate by Age Group', fontsize=13)
plt.xlabel('Age Group'); plt.ylabel('No-Show Rate (%)')
for i, row in result_sorted.iterrows():
    plt.text(i, row['no_show_rate_pct'] + 0.2, f"{row['no_show_rate_pct']}%", ha='center', fontsize=10)
plt.tight_layout(); plt.savefig('chart_age.png'); plt.close()


# ## 5. Impact of SMS Reminders — and checking the confound

result = sql("""
SELECT
    CASE WHEN SMS_received = 1 THEN 'SMS Sent' ELSE 'No SMS' END AS sms_status,
    COUNT(*)                            AS total,
    SUM(No_show)                        AS no_shows,
    ROUND(AVG(No_show) * 100, 1)        AS no_show_rate_pct
FROM appointments
GROUP BY SMS_received
ORDER BY SMS_received
""")
print(result.to_string(index=False))

# At face value, the SMS group has a HIGHER no-show rate. Before concluding
# "SMS reminders don't work" (or worse, "cause" no-shows), check whether SMS
# receipt is confounded with wait time — clinics can't send an SMS for a
# same-day booking, so SMS recipients are systematically the longer-wait group.
confound = sql("""
SELECT
    CASE WHEN SMS_received = 1 THEN 'SMS' ELSE 'No SMS' END AS sms_status,
    ROUND(AVG(Wait_Days), 1)            AS avg_wait_days
FROM appointments
GROUP BY SMS_received
""")
print(confound.to_string(index=False))
# SMS recipients wait ~19 days on average vs ~6 days for non-recipients.
# Wait time is a far stronger driver of no-shows (see Section 6), so the raw
# SMS comparison is confounded — it is not safe to read this as "SMS causes
# no-shows" without controlling for wait time.

plt.figure(figsize=(7, 4))
plt.bar(result['sms_status'], result['no_show_rate_pct'], color=['coral', 'steelblue'], edgecolor='white', width=0.5)
plt.title('No-Show Rate: SMS Reminder vs No Reminder (uncontrolled)', fontsize=13)
plt.xlabel('SMS Status'); plt.ylabel('No-Show Rate (%)')
for i, row in result.iterrows():
    plt.text(i, row['no_show_rate_pct'] + 0.2, f"{row['no_show_rate_pct']}%", ha='center', fontsize=11)
plt.tight_layout(); plt.savefig('chart_sms.png'); plt.close()


# ## 6. Impact of Waiting Time on No-Shows

result = sql("""
SELECT
    CASE
        WHEN Wait_Days = 0  THEN 'Same day'
        WHEN Wait_Days <= 7 THEN '1-7 days'
        WHEN Wait_Days <= 30 THEN '8-30 days'
        ELSE '30+ days'
    END AS wait_group,
    COUNT(*)                            AS total,
    SUM(No_show)                        AS no_shows,
    ROUND(AVG(No_show) * 100, 1)        AS no_show_rate_pct
FROM appointments
GROUP BY wait_group
ORDER BY no_show_rate_pct
""")
print(result.to_string(index=False))

order = ['Same day', '1-7 days', '8-30 days', '30+ days']
result_sorted = result.set_index('wait_group').reindex(order).reset_index()
plt.figure(figsize=(9, 4))
plt.bar(result_sorted['wait_group'], result_sorted['no_show_rate_pct'], color='steelblue', edgecolor='white')
plt.title('No-Show Rate by Waiting Time', fontsize=13)
plt.xlabel('Wait Time'); plt.ylabel('No-Show Rate (%)')
for i, row in result_sorted.iterrows():
    plt.text(i, row['no_show_rate_pct'] + 0.2, f"{row['no_show_rate_pct']}%", ha='center', fontsize=10)
plt.tight_layout(); plt.savefig('chart_wait.png'); plt.close()


# ## 7. No-Shows by Health Conditions & Scholarship

result = sql("""
SELECT 'Hypertension' AS condition,
    ROUND(AVG(CASE WHEN Hypertension=1 THEN No_show END)*100, 1) AS with_condition,
    ROUND(AVG(CASE WHEN Hypertension=0 THEN No_show END)*100, 1) AS without_condition
FROM appointments
UNION ALL
SELECT 'Diabetes',
    ROUND(AVG(CASE WHEN Diabetes=1 THEN No_show END)*100, 1),
    ROUND(AVG(CASE WHEN Diabetes=0 THEN No_show END)*100, 1)
FROM appointments
UNION ALL
SELECT 'Scholarship',
    ROUND(AVG(CASE WHEN Scholarship=1 THEN No_show END)*100, 1),
    ROUND(AVG(CASE WHEN Scholarship=0 THEN No_show END)*100, 1)
FROM appointments
""")
print(result.to_string(index=False))

x = np.arange(len(result)); width = 0.35
plt.figure(figsize=(10, 5))
plt.bar(x - width/2, result['with_condition'], width, label='With Condition', color='coral', edgecolor='white')
plt.bar(x + width/2, result['without_condition'], width, label='Without Condition', color='steelblue', edgecolor='white')
plt.xticks(x, result['condition'])
plt.title('No-Show Rate: With vs Without Health Condition / Scholarship', fontsize=13)
plt.ylabel('No-Show Rate (%)'); plt.legend()
plt.tight_layout(); plt.savefig('chart_conditions.png'); plt.close()


# ## 8. No-Show Rate by Neighbourhood

result = sql("""
SELECT
    Neighbourhood,
    COUNT(*)                            AS total,
    SUM(No_show)                        AS no_shows,
    ROUND(AVG(No_show) * 100, 1)        AS no_show_rate_pct
FROM appointments
GROUP BY Neighbourhood
HAVING COUNT(*) > 50
ORDER BY no_show_rate_pct DESC
LIMIT 10
""")
print(result.to_string(index=False))

plt.figure(figsize=(11, 5))
plt.barh(result['Neighbourhood'], result['no_show_rate_pct'], color='steelblue', edgecolor='white')
plt.gca().invert_yaxis()
plt.title('Top 10 Neighbourhoods by No-Show Rate', fontsize=13)
plt.xlabel('No-Show Rate (%)')
plt.tight_layout(); plt.savefig('chart_neighbourhood.png'); plt.close()


# ## 9. Advanced SQL — Window Functions & CTEs

result = sql("""
WITH neighbourhood_stats AS (
    SELECT
        Neighbourhood,
        COUNT(*)                        AS total,
        SUM(No_show)                    AS no_shows,
        ROUND(AVG(No_show)*100, 1)      AS no_show_rate_pct
    FROM appointments
    GROUP BY Neighbourhood
    HAVING COUNT(*) > 50
)
SELECT
    Neighbourhood, total, no_shows, no_show_rate_pct,
    RANK() OVER (ORDER BY no_show_rate_pct DESC) AS rank_worst
FROM neighbourhood_stats
ORDER BY rank_worst
""")
print(result.to_string(index=False))

result = sql("""
SELECT
    PatientId,
    COUNT(*)                            AS total_appointments,
    SUM(No_show)                        AS total_no_shows,
    ROUND(AVG(No_show)*100, 1)          AS personal_no_show_rate_pct
FROM appointments
GROUP BY PatientId
HAVING total_appointments >= 3
   AND total_no_shows >= 2
ORDER BY total_no_shows DESC
LIMIT 10
""")
print("High-risk repeat no-show patients:")
print(result.to_string(index=False))


# ## 10. Summary & Conclusions
#
# ### Key SQL Findings (real data, 110,516 appointments after cleaning)
#
# **Waiting time is the strongest predictor:**
# - Same-day appointments have a ~5% no-show rate; appointments booked 30+ days
#   out have a ~33% no-show rate. This is the single biggest lever in the data.
#
# **The SMS finding is a lesson in confounding, not a clean "reminders work" story:**
# - On its own, SMS-reminded patients have a HIGHER no-show rate (27.6% vs 16.7%).
# - But SMS recipients wait ~19 days on average vs ~6 days for non-recipients,
#   because same-day bookings can't receive a reminder in time. Wait time is
#   the real driver here, and SMS receipt is largely a proxy for it. A fair
#   read is: this dataset alone cannot support "SMS reminders reduce no-shows"
#   without controlling for wait time (e.g. comparing SMS vs no-SMS within the
#   same wait-time band).
#
# **Younger patients (18-34) have the highest no-show rate**, older patients
# (65+) are the most reliable attenders.
#
# **Patients with hypertension or diabetes are somewhat less likely to no-show**,
# possibly reflecting more consistent engagement with care for chronic conditions.
# **Scholarship recipients show a higher no-show rate**, consistent with
# transport/cost barriers as a contributing factor.
#
# ### SQL Techniques Demonstrated
# - `GROUP BY` with aggregates (`COUNT`, `SUM`, `AVG`)
# - `CASE WHEN` for conditional grouping and pivoting
# - `HAVING` for post-aggregation filtering
# - `UNION ALL` for combining multiple result sets
# - Window functions (`RANK() OVER`)
# - CTEs (`WITH` clause)
# - A basic confound check (grouping a second variable by the same segment)
#   before accepting a raw correlation as a finding
