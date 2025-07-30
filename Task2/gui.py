import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Predict function
def predict():
    try:
        income = float(entry_income.get())
        debt = float(entry_debt.get())
        credit_history = float(entry_history.get())
        age = float(entry_age.get())

        features = np.array([[income, debt, credit_history, age]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]

        result = "Creditworthy" if prediction == 1 else "Not Creditworthy"
        messagebox.showinfo("Prediction Result", f"The applicant is {result}.")
    except ValueError:
        messagebox.showerror("Input error", "Please enter valid numbers.")

# GUI
root = tk.Tk()
root.title("Credit Scoring App")

tk.Label(root, text="Income").grid(row=0)
tk.Label(root, text="Debt").grid(row=1)
tk.Label(root, text="Credit History (1-10)").grid(row=2)
tk.Label(root, text="Age").grid(row=3)

entry_income = tk.Entry(root)
entry_debt = tk.Entry(root)
entry_history = tk.Entry(root)
entry_age = tk.Entry(root)

entry_income.grid(row=0, column=1)
entry_debt.grid(row=1, column=1)
entry_history.grid(row=2, column=1)
entry_age.grid(row=3, column=1)

tk.Button(root, text='Predict', command=predict).grid(row=4, column=0, pady=10)
tk.Button(root, text='Quit', command=root.quit).grid(row=4, column=1, pady=10)

root.mainloop()
