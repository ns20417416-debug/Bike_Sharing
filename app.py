import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# ------------------------------------------------------------
# ABC Ltd. - Demand Prediction Decision-Support Tool
# ------------------------------------------------------------

st.set_page_config(
    page_title="ABC Ltd. Demand Prediction",
    page_icon="📊",
    layout="wide"
)

# File names expected in the same GitHub repository as this app
BASE_DIR = Path(__file__).resolve().parent
LINEAR_MODEL_FILE = BASE_DIR / "linear_regression_model_pipeline.joblib"
LOGISTIC_MODEL_FILE = BASE_DIR / "logistic_regression_model_pipeline.joblib"
MEDIAN_FILE = BASE_DIR / "training_median_demand.joblib"


@st.cache_resource
def load_models():
    """Load the saved models and training median."""
    linear_model = joblib.load(LINEAR_MODEL_FILE)
    logistic_model = joblib.load(LOGISTIC_MODEL_FILE)
    training_median = joblib.load(MEDIAN_FILE)

    # Convert numpy scalar/1-value arrays to a simple float when possible
    if isinstance(training_median, (list, tuple, np.ndarray, pd.Series)):
        training_median = float(np.asarray(training_median).reshape(-1)[0])
    else:
        training_median = float(training_median)

    return linear_model, logistic_model, training_median


def get_input_dataframe(
    season,
    year,
    month,
    holiday,
    weekday,
    workingday,
    weather,
    temperature_c,
    humidity_pct,
    windspeed_kmh
):
    """
    Prepare one input row in the same general representation as the
    UCI Bike Sharing day.csv dataset.

    The UCI data stores:
      temp      = temperature / 41
      hum       = humidity / 100
      windspeed = windspeed / 67 (approximately)

    Season, month, weekday, etc. are retained as their dataset codes.
    """
    # Convert user-friendly units to the normalized units used in day.csv.
    temp_norm = float(temperature_c) / 41.0
    hum_norm = float(humidity_pct) / 100.0
    wind_norm = float(windspeed_kmh) / 67.0

    # Keep values within the approximate ranges used in the dataset.
    temp_norm = float(np.clip(temp_norm, 0, 1))
    hum_norm = float(np.clip(hum_norm, 0, 1))
    wind_norm = float(np.clip(wind_norm, 0, 1))

    data = {
        "season": int(season),
        "yr": int(year),
        "mnth": int(month),
        "holiday": int(holiday),
        "weekday": int(weekday),
        "workingday": int(workingday),
        "weathersit": int(weather),
        "temp": temp_norm,
        "hum": hum_norm,
        "windspeed": wind_norm,
    }

    return pd.DataFrame([data])


def friendly_error_message(error):
    """Return a more useful message for common deployment/model issues."""
    message = str(error)

    if "feature_names" in message.lower():
        return (
            "The saved model expects different input column names. "
            "Please make sure the model was trained with the same feature set "
            "used in this application."
        )

    if "not found" in message.lower() or "no such file" in message.lower():
        return (
            "One or more model files are missing from the GitHub repository. "
            "Make sure all three .joblib files are in the same folder as app.py."
        )

    return message


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

st.title("📊 ABC Ltd. Demand Prediction Tool")
st.markdown(
    """
    **Decision-support tool for non-technical managers**

    Enter the expected operating and environmental conditions to estimate
    daily demand and the probability of a high-demand day.
    """
)

st.info(
    "This tool provides decision support based on historical data. "
    "It should complement, not replace, managerial judgement."
)

# ------------------------------------------------------------
# Load models
# ------------------------------------------------------------

try:
    linear_model, logistic_model, training_median = load_models()
except Exception as e:
    st.error("The prediction models could not be loaded.")
    st.code(friendly_error_message(e))
    st.stop()

# ------------------------------------------------------------
# Input section
# ------------------------------------------------------------

st.subheader("1. Enter operating conditions")

col1, col2, col3 = st.columns(3)

with col1:
    season_label = st.selectbox(
        "Season",
        options=["Spring", "Summer", "Fall", "Winter"],
        index=1,
        help="UCI dataset codes: 1=Spring, 2=Summer, 3=Fall, 4=Winter."
    )
    season_map = {"Spring": 1, "Summer": 2, "Fall": 3, "Winter": 4}
    season = season_map[season_label]

    year_label = st.selectbox(
        "Year",
        options=["2011", "2012"],
        index=1,
        help="2011 is coded as 0 and 2012 as 1 in the dataset."
    )
    year = 0 if year_label == "2011" else 1

    month = st.slider(
        "Month",
        min_value=1,
        max_value=12,
        value=6,
        help="1 = January, 12 = December."
    )

with col2:
    holiday_label = st.selectbox("Holiday", ["No", "Yes"], index=0)
    holiday = 0 if holiday_label == "No" else 1

    weekday_label = st.selectbox(
        "Weekday",
        ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
        index=1
    )
    weekday_map = {
        "Sunday": 0,
        "Monday": 1,
        "Tuesday": 2,
        "Wednesday": 3,
        "Thursday": 4,
        "Friday": 5,
        "Saturday": 6,
    }
    weekday = weekday_map[weekday_label]

    workingday_label = st.selectbox("Working day", ["Yes", "No"], index=0)
    workingday = 1 if workingday_label == "Yes" else 0

with col3:
    weather_label = st.selectbox(
        "Weather situation",
        [
            "Clear / Few clouds",
            "Mist / Cloudy",
            "Light rain / Snow",
            "Heavy rain / Snow"
        ],
        index=0
    )
    weather_map = {
        "Clear / Few clouds": 1,
        "Mist / Cloudy": 2,
        "Light rain / Snow": 3,
        "Heavy rain / Snow": 4,
    }
    weather = weather_map[weather_label]

    temperature_c = st.number_input(
        "Temperature (°C)",
        min_value=0.0,
        max_value=41.0,
        value=25.0,
        step=0.5
    )

    humidity_pct = st.slider(
        "Humidity (%)",
        min_value=0,
        max_value=100,
        value=60,
        step=1
    )

    windspeed_kmh = st.number_input(
        "Wind speed (km/h)",
        min_value=0.0,
        max_value=70.0,
        value=15.0,
        step=1.0
    )

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

st.subheader("2. Generate prediction")

if st.button("🔮 Predict Demand", type="primary", use_container_width=True):

    try:
        input_df = get_input_dataframe(
            season=season,
            year=year,
            month=month,
            holiday=holiday,
            weekday=weekday,
            workingday=workingday,
            weather=weather,
            temperature_c=temperature_c,
            humidity_pct=humidity_pct,
            windspeed_kmh=windspeed_kmh
        )

        # Linear Regression prediction
        linear_prediction = float(np.asarray(linear_model.predict(input_df)).reshape(-1)[0])
        linear_prediction = max(0.0, linear_prediction)

        # Logistic Regression prediction
        logistic_class = int(
            np.asarray(logistic_model.predict(input_df)).reshape(-1)[0]
        )

        high_demand_probability = None
        if hasattr(logistic_model, "predict_proba"):
            probabilities = logistic_model.predict_proba(input_df)
            classes = getattr(logistic_model, "classes_", None)

            # Support normal sklearn pipelines where classes are exposed
            # by the final estimator through the pipeline.
            if classes is not None:
                classes = list(classes)
                if 1 in classes:
                    high_idx = classes.index(1)
                else:
                    high_idx = int(np.argmax(classes))
            else:
                high_idx = 1 if probabilities.shape[1] > 1 else 0

            high_demand_probability = float(probabilities[0, high_idx]) * 100
        else:
            high_demand_probability = 100.0 if logistic_class == 1 else 0.0

        model_category = "High Demand" if logistic_class == 1 else "Low Demand"

        # --------------------------------------------------------
        # Results
        # --------------------------------------------------------

        st.subheader("3. Prediction results")

        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric(
                "Expected Daily Demand",
                f"{linear_prediction:,.0f}"
            )

        with r2:
            st.metric(
                "High-Demand Probability",
                f"{high_demand_probability:.1f}%"
            )

        with r3:
            if model_category == "High Demand":
                st.success("Demand Category\n\n### HIGH DEMAND")
            else:
                st.info("Demand Category\n\n### LOW DEMAND")

        # Benchmark against training median
        st.markdown("### Demand benchmark")
        st.write(
            f"The training-data median demand is **{training_median:,.0f}**. "
            f"The predicted demand is "
            f"**{linear_prediction / training_median * 100:.1f}%** of that benchmark."
            if training_median != 0
            else "The training-data median demand is unavailable or zero."
        )

        # Managerial interpretation
        st.markdown("### Managerial interpretation")

        if model_category == "High Demand":
            st.write(
                "The model classifies the current conditions as **High Demand**. "
                "ABC Ltd. may consider preparing sufficient operational capacity "
                "and monitoring demand closely. The prediction should be reviewed "
                "alongside current market information and managerial experience."
            )
        else:
            st.write(
                "The model classifies the current conditions as **Low Demand**. "
                "ABC Ltd. may consider planning resources accordingly while still "
                "checking for unusual market or operational conditions."
            )

        # Input summary
        with st.expander("View model input values"):
            st.dataframe(
                input_df.rename(
                    columns={
                        "yr": "Year Code",
                        "mnth": "Month",
                        "weathersit": "Weather Code",
                        "temp": "Normalized Temperature",
                        "hum": "Normalized Humidity",
                        "windspeed": "Normalized Windspeed",
                    }
                ),
                use_container_width=True
            )

        st.caption(
            "Note: The model predicts from historical relationships in the training "
            "data and does not establish causality."
        )

    except Exception as e:
        st.error("The prediction could not be generated.")
        st.code(friendly_error_message(e))

# ------------------------------------------------------------
# Methodology section
# ------------------------------------------------------------

with st.expander("About this tool"):
    st.markdown(
        """
        **Linear Regression** estimates the expected daily demand as a number.

        **Logistic Regression** estimates whether the day is more consistent
        with the model's High Demand or Low Demand category and provides a
        probability where supported by the model.

        The saved model pipelines include the preprocessing used during
        model training. This is important because the application's inputs
        must be transformed consistently with the training data.
        """
    )
