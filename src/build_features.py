# src/build_features va construire les jeux de données prêt à être passer dans le notebook des modèles
import pandas as pd
import numpy as np

REF_DATE = pd.to_datetime("2020-08-01").tz_localize("UTC")
REF_DATE_NAIVE = REF_DATE.tz_localize(None)

# Colonne relative à la part du paiement
PAY_COLS = ["payment_gift", "payment_cheque", "payment_cash", "payment_card"]

def build_features(df: pd.DataFrame) -> pd.DataFrame:

    """Construit des features au niveau du client à partir des transactions

    Fonctionne de manière identique pour le train et test
    """
    df=df.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

    # On part sur tous les clients de base, sans distinction sur les paniers (qui peuvent être positifs mais aussi négatifs)
    clients = pd.Index(df["id_client"].unique(), name="id_client")

    # 1. Sur les paniers en positifs (on aggrègera après sur les clients)
    pos = df[df["basket_value"] > 0]
    rfm = pos.groupby("id_client").agg(
        last_purchase=("transaction_date", "max"),
        first_purchase=("transaction_date", "min"),
        frequency=("transaction_date", "size"),
        monetary=("basket_value", "sum"),
        avg_basket=("basket_value", "mean"),)
    rfm["recency"] = (REF_DATE - rfm["last_purchase"]).dt.days
    rfm["tenure"] = (REF_DATE - rfm["first_purchase"]).dt.days
    rfm = rfm.drop(columns=["last_purchase", "first_purchase"])

    # 2. Dynamique du client

    # car dans le projet la prof dit que y a potentiellement +ieurs visites / client / j
    visits = (pos.assign(day=pos["transaction_date"].dt.normalize())
        .groupby(["id_client", "day"], as_index=False)
        .agg(amount=("basket_value", "sum"))
        .sort_values(["id_client", "day"]))
    rfm["n_visit_days"] = visits.groupby("id_client").size()

    # fenêtre temporelle
    dyn = pd.DataFrame(index=rfm.index)
    for d in (30, 60, 90):
        recent = visits[visits["day"] >= REF_DATE - pd.Timedelta(days=d)]
        dyn[f"visits_{d}d"] = recent.groupby("id_client").size()
        if d == 90:
            dyn["amount_90d"] = recent.groupby("id_client")["amount"].sum()
    dyn = dyn.fillna(0)
    dyn["share_visits_90d"] = dyn["visits_90d"] / rfm["n_visit_days"]

    # rythme du client
    gaps = visits.groupby("id_client")["day"].diff().dt.days
    mean_gap = gaps.groupby(visits["id_client"]).mean().rename("mean_gap")
    rhythm = pd.DataFrame(mean_gap)
    rhythm["gap_ratio"] = rfm["recency"] / rhythm["mean_gap"]

    # 3. Sur les paniers en négatif
    ret = df[df["basket_value"] < 0]
    ret_feat = ret.groupby("id_client").agg(
        nb_returns=("basket_value", "size"),
        return_amount=("basket_value", lambda s: -s.sum()),)
    n_tickets = df.groupby("id_client").size().rename("n_tickets")

    # 4. Part du mode de paiement pour chaque individu
    pay = pos.groupby("id_client")[PAY_COLS].mean().add_prefix("share_")

    # 5. Variables du profil client
    raw = df["card_subscription"]
    card_date = pd.to_datetime(raw, format="%d/%m/%Y", errors="coerce")
    n_fail = (raw.notna() & card_date.isna()).sum()
    if n_fail > 0.01 * raw.notna().sum():
        raise ValueError(f"card_subscription : {n_fail} dates non parsées, vérifier le format")
    card = card_date.groupby(df["id_client"]).min().rename("card_date")

    last = df.sort_values("transaction_date").groupby("id_client").agg(
        civility=("civility", "last"),
        multicard_raw=("multicard", "max"),
    )
    profile = last.join(card)
    profile["has_card"] = profile["card_date"].notna().astype(int)
    profile["card_age"] = (REF_DATE_NAIVE - profile["card_date"]).dt.days  # NaN si pas de carte
    m = profile["multicard_raw"]
    profile["has_multicard"] = (m > 0).astype(float).where(m.notna())  # NaN conservé = inconnu
    profile["civility"] = profile["civility"].fillna("unknown")
    profile = profile.drop(columns=["card_date", "multicard_raw"])

    # 6. On joint tous ces tableaux
    features = (
        pd.DataFrame(index=clients)
        .join(rfm)
        .join(dyn)
        .join(rhythm)
        .join(ret_feat)
        .join(n_tickets)
        .join(pay)
        .join(profile))

    # On crée de nouvelles features et des dummy variables
    features["no_positive_purchase"] = features["frequency"].isna().astype(int)
    na_cols = ["nb_returns", "return_amount", "visits_30d", "visits_60d",
              "visits_90d", "amount_90d"]
    features[na_cols] = features[na_cols].fillna(0)

    features["has_return"] = (features["nb_returns"] > 0).astype(int)
    features["return_rate"] = features["nb_returns"] / features["n_tickets"]
    features["return_share_amount"] = (
        features["return_amount"] / features["monetary"].replace(0, np.nan)
    ).clip(upper=1)

    return features
