#!/usr/bin/env python3

import argparse
import sys
import pandas as pd
import joblib
import os

def read_csv_flexible(path):
    for sep in [';', ',', '\t']:
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    return pd.read_csv(path)

def main():
    parser = argparse.ArgumentParser(description='Prédit si un billet est genuine et affiche le dataframe avec la colonne Prediction.')
    parser.add_argument('csv', help='fichier CSV en entrée')
    parser.add_argument('-m', '--model', default='./Weng_Tony_2_modèle_112025.pkl', help='fichier pipeline joblib (par défaut: Weng_Tony_2_modèle_112025.pkl)')
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Erreur: fichier '{args.csv}' introuvable.", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.model):
        print(f"Erreur: modèle '{args.model}' introuvable.", file=sys.stderr)
        sys.exit(1)

    try:
        pipeline = joblib.load(args.model)
    except Exception as e:
        print(f"Erreur chargement modèle: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        df = read_csv_flexible(args.csv)
    except Exception as e:
        print(f"Erreur lecture CSV: {e}", file=sys.stderr)
        sys.exit(1)

    required = ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Colonnes manquantes requises pour la prédiction: {missing}", file=sys.stderr)
        sys.exit(1)

    X = df[required]
    try:
        preds = pipeline.predict(X)
    except Exception as e:
        print(f"Erreur lors de la prédiction: {e}", file=sys.stderr)
        sys.exit(1)

    df['Prediction'] = preds 
    df['% (Faux)'] = pipeline.predict_proba(X)[:, 0] * 100
    df['% (Vrai)'] = pipeline.predict_proba(X)[:, 1] * 100
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(df.to_string(index=False))

    output_file = f'prediction_{os.path.basename(args.csv)}'
    df.to_csv(output_file, index=False)
    print(f"\nLes prédictions ont été sauvegardées dans '{output_file}'.")

if __name__ == '__main__':
    main()

# python3 prediction.py 'Nom du fichier CSV'