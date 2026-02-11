import numpy as np
import pandas as pd


# -------------------------
# Lección 1: Generación con NumPy
# -------------------------
def generar_datos_numpy(n_customers=500, n_transactions=5000, seed=42):
    np.random.seed(seed)

    customer_ids = np.arange(10000, 10000 + n_customers)
    ages = np.random.randint(18, 71, size=n_customers)
    tenure_months = np.random.randint(0, 61, size=n_customers)
    customers = np.column_stack([customer_ids, ages, tenure_months])

    transaction_ids = np.arange(1, n_transactions + 1)
    tx_customer_id = np.random.choice(customer_ids, size=n_transactions, replace=True)
    quantity = np.random.randint(1, 13, size=n_transactions)

    unit_price = np.random.gamma(shape=2.0, scale=10000.0, size=n_transactions)
    unit_price = np.clip(unit_price, 500, 150000).astype(int)

    total = quantity * unit_price
    transactions = np.column_stack([transaction_ids, tx_customer_id, quantity, unit_price, total])

    np.save("customers.npy", customers)
    np.save("transactions.npy", transactions)

    return customers, transactions


# -------------------------
# Lección 2: Pandas + CSV preliminar integrado
# -------------------------
def leccion_2_pandas(customers, transactions):
    customers_df = pd.DataFrame(customers, columns=["customer_id", "age", "tenure_months"])
    transactions_df = pd.DataFrame(transactions, columns=["transaction_id", "customer_id", "quantity", "unit_price", "total"])

    df_integrado = transactions_df.merge(customers_df, on="customer_id", how="left")
    df_integrado.to_csv("dataset_preliminar_integrado.csv", index=False)

    return df_integrado


# -------------------------
# Lección 3: Integración CSV + Excel + HTML
# -------------------------
def leccion_3_integracion(df_base_path="dataset_preliminar_integrado.csv",
                          excel_path="customers_country_latam.xlsx",
                          html_path="customer_payment_method.html"):
    df_base = pd.read_csv(df_base_path)
    df_excel = pd.read_excel(excel_path)
    df_html = pd.read_html(html_path)[0]

    df = df_base.merge(df_excel, on="customer_id", how="left")
    df = df.merge(df_html, on="customer_id", how="left")

    df.to_csv("dataset_consolidado.csv", index=False)
    return df


# -------------------------
# Lección 4: Limpieza nulos + outliers
# -------------------------
def cap_outliers_iqr(df, col, k=1.5):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    df[col] = df[col].clip(lower, upper)
    return df


def leccion_4_limpieza(path="dataset_consolidado.csv"):
    df = pd.read_csv(path)

    # Nulos: categóricas -> Desconocido; numéricas -> mediana
    for col in ["country", "payment_method"]:
        if col in df.columns:
            df[col] = df[col].fillna("Desconocido")

    for col in ["age", "tenure_months", "quantity", "unit_price", "total"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Outliers: capado IQR
    for col in ["total", "unit_price", "quantity"]:
        if col in df.columns:
            df = cap_outliers_iqr(df, col)

    df.to_csv("dataset_limpio.csv", index=False)
    return df


# -------------------------
# Lección 5: Data Wrangling (mínimo)
# -------------------------
def leccion_5_wrangling(path="dataset_limpio.csv"):
    df = pd.read_csv(path)

    df = df.drop_duplicates(subset=["transaction_id"])

    for c in ["transaction_id", "customer_id", "quantity", "unit_price", "total", "age", "tenure_months"]:
        if c in df.columns:
            df[c] = df[c].astype(int)

    df["avg_price_per_unit"] = df["total"] / df["quantity"]
    df["high_value_tx"] = df["total"].apply(lambda x: x > 150000)

    if "payment_method" in df.columns:
        df["payment_method"] = df["payment_method"].map(
            lambda x: "Crédito" if str(x) in ["CrÃ©dito", "Credito"] else x
        )

    df["age_group"] = pd.cut(
        df["age"],
        bins=[17, 30, 45, 60, 100],
        labels=["18-30", "31-45", "46-60", "60+"]
    )

    df.to_csv("dataset_transformado.csv", index=False)
    return df


# -------------------------
# Lección 6: Groupby + Pivot + Melt + Export
# -------------------------
def leccion_6_analisis_y_export(path="dataset_transformado.csv"):
    df = pd.read_csv(path)

    resumen_pais = (
        df.groupby("country")
          .agg(
              ventas_totales=("total", "sum"),
              ticket_promedio=("total", "mean"),
              transacciones=("transaction_id", "count")
          )
          .reset_index()
    )

    pivot_pais_pago = df.pivot_table(
        index="country",
        columns="payment_method",
        values="total",
        aggfunc="sum",
        fill_value=0
    )

    melt_pais_pago = pivot_pais_pago.reset_index().melt(
        id_vars="country",
        var_name="payment_method",
        value_name="ventas_totales"
    )

    # Export final dataset
    df.to_csv("dataset_final.csv", index=False)
    df.to_excel("dataset_final.xlsx", index=False)

    # También exportar resúmenes (opcional útil)
    resumen_pais.to_csv("resumen_pais.csv", index=False)
    melt_pais_pago.to_csv("ventas_por_pais_y_pago.csv", index=False)

    return df


# -------------------------
# MAIN
# -------------------------
def main():
    customers, transactions = generar_datos_numpy()
    leccion_2_pandas(customers, transactions)
    leccion_3_integracion()
    leccion_4_limpieza()
    leccion_5_wrangling()
    leccion_6_analisis_y_export()

    print("Proceso completo ejecutado ✅")
    print("Archivos clave generados:")
    print("- dataset_preliminar_integrado.csv")
    print("- dataset_consolidado.csv")
    print("- dataset_limpio.csv")
    print("- dataset_transformado.csv")
    print("- dataset_final.csv / dataset_final.xlsx")


if __name__ == "__main__":
    main()
