import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import category_encoders as ce
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('../data/gurgaon_properties_post_feature_selection_v2.csv')
df['furnishing_type'] = df['furnishing_type'].replace({0.0:'unfurnished',1.0:'semifurnished',2.0:'furnished'})

X = df.drop(columns=['price'])
y = df['price']

y_transformed = np.log1p(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, test_size=0.2, random_state=42)

categorical_cols = ['property_type','sector', 'balcony', 'agePossession',
                    'furnishing_type', 'luxury_category', 'floor_category']

numeric_cols = ['bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', ce.TargetEncoder(), categorical_cols)
    ],
    remainder='drop'
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(
        learning_rate=0.05,
        n_estimators=400,
        max_depth=7,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.01,
        reg_lambda=1
    ))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

y_pred = pipeline.predict(X_test)
y_pred_raw = np.expm1(y_pred)
y_test_raw = np.expm1(y_test)

print("MAE :", mean_absolute_error(y_test_raw, y_pred_raw))
print("MSE :", mean_squared_error(y_test_raw, y_pred_raw))
print("RMSE:", mean_squared_error(y_test_raw, y_pred_raw) ** 0.5)
print("R2  :", r2_score(y_test_raw, y_pred_raw))

joblib.dump(pipeline, "../model/final_model.pkl")
