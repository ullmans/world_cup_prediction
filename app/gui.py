import json

import streamlit as st

from world_cup_prediction.config import TEAMS_MAPPING_JSON
from world_cup_prediction.memory import update_tournament_memory
from world_cup_prediction.predictor import main_live_predictor

with open(TEAMS_MAPPING_JSON, encoding="utf-8") as f:
    TEAMS_MAPPING = json.load(f)

hebrew_team_names = sorted(TEAMS_MAPPING.keys())

st.set_page_config(page_title="World Cup Predictor", page_icon="⚽", layout="centered")
st.title("⚽ World Cup Live Predictor")
tab_predict, tab_update = st.tabs(["🔮 חיזוי משחק חדש", "📝 הזנת תוצאת אמת (למידה)"])

with tab_predict:
    st.markdown("בחר את הנבחרות כדי לקבל את 3 התוצאות הסבירות ביותר למשחק.")

    col_input1, col_input2 = st.columns(2)
    with col_input1:
        team1_hebrew = st.selectbox("נבחרת מארחת:", hebrew_team_names, key="pred_t1")
    with col_input2:
        team2_hebrew = st.selectbox("נבחרת אורחת:", hebrew_team_names, key="pred_t2")

    if st.button("🚀 Predict Match"):
        if team1_hebrew == team2_hebrew:
            st.error("שגיאה: אי אפשר לבחור את אותה נבחרת פעמיים!")
        else:
            team1_english = TEAMS_MAPPING[team1_hebrew]
            team2_english = TEAMS_MAPPING[team2_hebrew]

            with st.spinner('מריץ את המודל...'):
                predictions = main_live_predictor(team1_english, team2_english)

            st.success("החיזוי הושלם!")
            st.markdown("### 🏆 Top 3 Predicted Outcomes")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.info("🥇 סבירות הגבוהה ביותר")
                st.metric(label="תוצאה חזויה", value=str(predictions[0][0]) + "-" + str(predictions[0][1]))
                st.progress(int(predictions[0][2]))
                st.caption(f"סיכוי: {predictions[0][2]}%")

            with col2:
                st.warning("🥈 אפשרות שנייה")
                st.metric(label="תוצאה חזויה", value=str(predictions[1][0]) + "-" + str(predictions[1][1]))
                st.progress(int(predictions[1][2]))
                st.caption(f"סיכוי: {predictions[1][2]}%")

            with col3:
                st.error("🥉 אפשרות שלישית")
                st.metric(label="תוצאה חזויה", value=str(predictions[2][0]) + "-" + str(predictions[2][1]))
                st.progress(int(predictions[2][2]))
                st.caption(f"סיכוי: {predictions[2][2]}%")

with tab_update:
    st.markdown(
        "הזן תוצאה של משחק שכבר שוחק. הנתונים יישלחו ל-Backend כדי לעדכן את המודל לקראת המשחקים הבאים."
    )

    col_u1, col_u2 = st.columns(2)
    with col_u1:
        team1_upd_hebrew = st.selectbox("נבחרת מארחת:", hebrew_team_names, key="upd_t1")
        score1_update = st.number_input("Home Score", min_value=0, step=1, key="upd_s1")
    with col_u2:
        team2_upd_hebrew = st.selectbox("נבחרת אורחת:", hebrew_team_names, key="upd_t2")
        score2_update = st.number_input("Away Score", min_value=0, step=1, key="upd_s2")

    if st.button("💾 שמור תוצאה ועדכן מודל"):
        if team1_upd_hebrew == team2_upd_hebrew:
            st.error("שגיאה: אי אפשר לעדכן משחק של נבחרת נגד עצמה!")
        else:
            team1_upd_english = TEAMS_MAPPING[team1_upd_hebrew]
            team2_upd_english = TEAMS_MAPPING[team2_upd_hebrew]

            with st.spinner("שולח נתונים ל-Backend..."):
                success = update_tournament_memory(
                    team1_upd_english, team2_upd_english, int(score1_update), int(score2_update)
                )

            if success:
                st.success(
                    f"התוצאה ({team1_upd_hebrew} {score1_update} - {score2_update} {team2_upd_hebrew}) "
                    "נשמרה בהצלחה והמערכת עודכנה!"
                )
