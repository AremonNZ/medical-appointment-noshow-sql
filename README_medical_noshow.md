# 🏥 Medical Appointment No-Shows — SQL Analysis

A data analysis project using SQL to explore patient behaviour and identify the key factors that predict medical appointment no-shows.

---

## 📌 Problem Statement

Missed medical appointments cost healthcare systems time and money, and can delay care for other patients. This project uses SQL queries on a dataset of 5,500 appointments to uncover patterns in patient behaviour — answering questions like: Does sending an SMS reminder help? Are younger patients more likely to miss appointments? Does waiting time matter?

---

## 📂 Dataset

- **Size:** 5,500 rows × 12 columns
- **Target variable:** `No_show` (1 = missed appointment, 0 = attended)
- **Key features:** Age, gender, neighbourhood, wait days, SMS reminder, health conditions (hypertension, diabetes, alcoholism), scholarship status

---

## 🔧 Tools & Libraries

| Tool | Purpose |
|------|---------|
| Python | Core language |
| SQLite | In-memory relational database |
| Pandas | Reading SQL results into DataFrames |
| Matplotlib & Seaborn | Data visualisation |
| Jupyter Notebook | Development environment |

---

## 🚀 Project Workflow

1. **Setup** — loaded appointment data into a SQLite database to simulate a real-world relational data environment
2. **Basic Exploration** — summary statistics using `COUNT`, `SUM`, `AVG`
3. **No-Show by Gender** — grouped comparison
4. **No-Show by Age Group** — conditional grouping with `CASE WHEN`
5. **SMS Reminder Impact** — comparing attendance with and without reminders
6. **Wait Time Analysis** — grouped by waiting period before appointment
7. **Health Conditions Analysis** — comparing no-show rates for patients with/without chronic conditions
8. **Neighbourhood Analysis** — identifying highest-risk areas
9. **Advanced SQL** — window functions (`RANK() OVER`) and CTEs (`WITH`) to rank neighbourhoods and find high-risk repeat no-show patients

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

- **Waiting time** is the strongest predictor — appointments booked 30+ days out have much higher no-show rates
- **SMS reminders work** — patients who received a reminder were significantly more likely to attend
- **Younger patients (18–34)** have the highest no-show rates; patients aged 65+ are the most reliable
- Patients with **chronic conditions** (hypertension, diabetes) are less likely to miss appointments
- **Scholarship recipients** show slightly higher no-show rates, possibly reflecting transport or cost barriers

---

**Dependencies:**
```bash
pip install pandas numpy matplotlib seaborn
```

---

## 👤 Author

**Arman Arabkhani**  
Third-year Computer Science (Data Science) student @ Auckland University of Technology  
📧 armanarabkhani.nz@yahoo.com  
🔗 [LinkedIn](https://www.linkedin.com/in/arman-arabkhani-95903a384/) | [GitHub](https://github.com/AremonNZ)
