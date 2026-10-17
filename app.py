"""Streamlit UI for hotel booking cancellation prediction.

Run:  streamlit run app.py
"""
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from src import predictor as P

st.set_page_config(page_title="Hotel Cancellation Predictor", page_icon="🏨", layout="wide")


@st.cache_resource
def load_predictor():
    return P.Predictor()


st.title("🏨 Hotel Booking Cancellation Predictor")
st.caption("Enter booking details to estimate the chance the guest cancels.")

# No st.form here: widgets inside a form do not rerun until submit, and the date-driven
# fields below must update as soon as a date changes.
with st.container():
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Stay")
        hotel = st.selectbox("Hotel", P.HOTELS)
        booked_on = st.date_input("Booking date", date.today())
        arrival = st.date_input("Check-in date", date.today() + timedelta(days=30), min_value=booked_on)
        checkout = st.date_input("Check-out date", arrival + timedelta(days=3), min_value=arrival + timedelta(days=1))
        # Derived automatically from the three dates
        nights = [arrival + timedelta(days=i) for i in range((checkout - arrival).days)]
        weekend = sum(d.weekday() in (4, 5) for d in nights)   # Fri/Sat nights, as in the dataset
        week = len(nights) - weekend
        lead_time = (arrival - booked_on).days
        st.caption(f"Lead time: **{lead_time}** days · Nights: **{len(nights)}** "
                   f"({weekend} weekend, {week} week) · Arrival month/week: "
                   f"**{arrival:%B}** / **{arrival.isocalendar()[1]}**")
        adr = st.number_input("Average daily rate (ADR)", 0.0, 999.0, 100.0)
    with c2:
        st.subheader("Guests")
        adults = st.number_input("Adults", 0, 55, 2)
        children = st.number_input("Children", 0, 10, 0)
        babies = st.number_input("Babies", 0, 3, 0)
        country = st.selectbox("Country", P.COUNTRIES)
        customer_type = st.selectbox("Customer type", P.CUSTOMER_TYPES, index=2)
        repeated = st.checkbox("Repeated guest")
        prev_cancel = st.number_input("Previous cancellations", 0, 30, 0)
        prev_ok = st.number_input("Previous bookings not cancelled", 0, 80, 0)
    with c3:
        st.subheader("Booking")
        segment = st.selectbox("Market segment", P.MARKET_SEGMENTS, index=6)
        channel = st.selectbox("Distribution channel", P.DISTRIBUTION_CHANNELS, index=3)
        deposit = st.selectbox("Deposit type", P.DEPOSIT_TYPES)
        meal = st.selectbox("Meal plan", P.MEALS)
        room = st.selectbox("Reserved room type", P.ROOM_TYPES)
        has_agent = st.checkbox("Booked via agent")
        changes = st.number_input("Booking changes", 0, 20, 0)
        waiting = st.number_input("Days in waiting list", 0, 400, 0)
        parking = st.number_input("Parking spaces", 0, 8, 0)
        special_requests = st.number_input("Special requests", 0, 5, 0)
    submitted = st.button("Predict", type="primary")

if submitted:
    if adults + children + babies == 0:
        st.error("A booking needs at least one guest.")
        st.stop()
    if not 1 <= len(nights) <= 70:
        st.error("Stay must be between 1 and 70 nights.")
        st.stop()
    booking = dict(
        hotel=hotel, arrival_date=arrival, lead_time=lead_time,
        stays_in_weekend_nights=weekend, stays_in_week_nights=week,
        adults=adults, children=children, babies=babies, meal=meal, country=country,
        market_segment=segment, distribution_channel=channel, is_repeated_guest=int(repeated),
        previous_cancellations=prev_cancel, previous_bookings_not_canceled=prev_ok,
        reserved_room_type=room, booking_changes=changes, deposit_type=deposit,
        has_agent=has_agent, days_in_waiting_list=waiting, customer_type=customer_type,
        adr=adr, required_car_parking_spaces=parking, total_of_special_requests=special_requests,
    )
    st.session_state["result"] = load_predictor().predict(booking)

if "result" in st.session_state:
    result = st.session_state["result"]
    prob = result["cancel_probability"]

    m1, m2, m3 = st.columns(3)
    m1.metric("Cancellation probability", f"{prob:.0%}")
    m2.metric("Risk level", result["risk_level"])
    m3.metric("Prediction", "Likely to cancel" if result["will_cancel"] else "Likely to stay")
    st.progress(prob)

    st.subheader("Why this score?")
    st.caption("Impact is in log-odds: positive pushes towards cancellation, negative towards staying.")
    factors = pd.DataFrame(result["top_factors"])
    st.bar_chart(factors.set_index("feature")["impact"])
    st.dataframe(factors[["feature", "value", "direction", "impact"]], hide_index=True)

    if st.button("Generate staff summary (Gemini)"):
        try:
            with st.spinner("Writing summary..."):
                st.info(P.gemini_summary(prob, result["top_factors"]))
        except Exception as e:
            st.warning(f"Summary unavailable: {e}")
