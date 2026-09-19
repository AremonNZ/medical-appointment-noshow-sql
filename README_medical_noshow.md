# 🏥 Medical Appointment No-Shows — SQL Analysis

A SQL-driven analysis of patient behaviour and appointment no-shows, using the real
**Medical Appointment No Shows** dataset from Kaggle (110,527 appointments from public
health clinics in Vitória, Brazil, April–June 2016).

---

## 📌 Problem Statement

Missed medical appointments cost healthcare systems time and money, and can delay care
for other patients. This project uses SQL queries over the real appointment dataset to
uncover patterns in patient behaviour — does waiting time matter? Do SMS reminders help?
Are younger patients more likely to miss appointments?

---

## 📂 Dataset

- **Size:** 110,527 rows × 14 columns (110,516 after removing 11 rows with invalid age
  or negative wait times)
- **Source:** [Kaggle — Medical Appointment No Shows](https://www.kaggle.com/datasets/joniarroba/noshowappointments)
- **Target variable:** `No_show` (1 = missed appointment, 0 = attended)
- **Key features:** Age, gender, neighbourhood, wait days (derived from `ScheduledDay`
  and `AppointmentDay`), SMS reminder, health conditions (hypertension, diabetes,
  alcoholism), scholarship status

---

## 🔧 Tools & Libraries

| Tool | Purpose |
|------|---------|
| Python | Core language |
| SQLite | Relational database — real SQL, not a DataFrame substitute |
| Pandas | Reading SQL results into DataFrames |
| Matplotlib & Seaborn | Data visualisation |

---

## 🚀 Project Workflow

1. **Load & clean** — loaded the real CSV, parsed appointment/scheduled dates, derived
   `Wait_Days`, dropped 11 rows with invalid age or negative wait time
2. **Basic exploration** — summary statistics using `COUNT`, `SUM`, `AVG`
3. **No-show by gender** — grouped comparison
4. **No-show by age group** — conditional grouping with `CASE WHEN`
5. **SMS reminder impact** — compared attendance with/without reminders, then checked
   whether the result was confounded by wait time (it was — see Key Findings)
6. **Wait time analysis** — grouped by waiting period before appointment
7. **Health conditions analysis** — no-show rates for patients with/without chronic
   conditions
8. **Neighbourhood analysis** — identifying highest-risk areas
9. **Advanced SQL** — window functions (`RANK() OVER`) and CTEs (`WITH`) to rank
   neighbourhoods and find high-risk repeat no-show patients

---

## 🧠 SQL Techniques Demonstrated

- `GROUP BY` with aggregate functions (`COUNT`, `SUM`, `AVG`)
- `CASE WHEN` for conditional grouping and pivoting
- `HAVING` for post-aggregation filtering
- `UNION ALL` for combining multiple result sets
- **Window functions** (`RANK() OVER`)
- **Common Table Expressions** (`WITH` clause)

---

## 📊 Key Findings

- **Waiting time is the strongest predictor.** Same-day appointments have a ~5% no-show
  rate; appointments booked 30+ days out climb to ~33%.
- **The SMS result is a lesson in confounding, not a clean "reminders work" story.**
  Taken at face value, patients who received an SMS reminder had a *higher* no-show rate
  (27.6%) than those who didn't (16.7%) — the opposite of what you'd expect. Digging in,
  SMS recipients wait ~19 days on average vs. ~6 days for non-recipients (the clinic
  can't send a reminder for a same-day booking), and wait time is the dominant driver of
  no-shows. So the raw SMS comparison is confounded, and this dataset alone doesn't
  support "reminders reduce no-shows" without controlling for wait time first.
- **Younger patients (18–34)** have the highest no-show rate; patients aged **65+** are
  the most reliable attenders.
- Patients with **chronic conditions** (hypertension, diabetes) show somewhat lower
  no-show rates, possibly reflecting more consistent engagement with care.
- **Scholarship recipients** show a higher no-show rate, consistent with transport or
  cost barriers as a contributing factor.

---

**Dependencies:**
```bash
pip install pandas numpy matplotlib seaborn
```

---

## 👤 Author

**Arman Arabkhani**
Final-year Computer Science (Data Science) student @ Auckland University of Technology
📧 armanarabkhani.nz@yahoo.com
🔗 [LinkedIn](https://www.linkedin.com/in/arman-arabkhani-95903a384/) | [GitHub](https://github.com/AremonNZ)
an-arabkhani-95903a384/) | [GitHub](https://github.com/AremonNZ)
