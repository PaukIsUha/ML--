import pandas as pd
import numpy as np
from consts import infl
from catboost import CatBoostClassifier


def quarter_to_num(txt):
    arr = txt.split('Q')
    return int(arr[0]) * 4 + int(arr[1])


def only_month(x):
    if x != x:
        return "nan"
    xarr = x.split("-")
    return xarr[0] + "-" + xarr[1]


print('Loading...')
df = pd.read_csv('test.csv', encoding='utf8')
df_alone = df.copy()
df_alone['quarter'] = df_alone['quarter'].apply(lambda x: quarter_to_num(x))
df_alone = df_alone.drop(columns=['client_id', 'npo_account_id', 'postal_code', ])
df_alone = pd.get_dummies(df_alone, columns=['slctn_nmbr'], prefix='slctn_nmbr', drop_first=True)
df_alone['frst_pmnt_date'] = df_alone["frst_pmnt_date"].apply(lambda x: only_month(x))
df_alone['lst_pmnt_date_per_qrtr'] = df_alone["lst_pmnt_date_per_qrtr"].apply(lambda x: only_month(x))
df_alone["frst_pmnt_date"] = df_alone["frst_pmnt_date"].replace(infl)
df_alone["lst_pmnt_date_per_qrtr"] = df_alone["lst_pmnt_date_per_qrtr"].replace(infl)

X = df_alone.drop(['region'], axis=1)
model = CatBoostClassifier()
model.load_model('catboost_model.cbm', format='cbm')

test_pred = model.predict(X)
df['churn'] = test_pred

df_test = df[['npo_account_id', 'quarter', 'churn']]
df_test.to_csv('result.csv', index=False)
print('Finished')
