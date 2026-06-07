import streamlit as st
import time

# --- כאן תייבא את הפונקציה האמיתית שלך מה-Backend ---
from live_predictor import main_live_predictor
from update_memory import update_tournament_memory

def mock_main_live_predictor(team1: str, team2: str):
    """
    פונקציית דמי שמדמה את ה-Backend שלך.
    מקבלת שתי קבוצות באנגלית ומחזירה את 3 התחזיות המובילות.
    """
    # סימולציה של השהיית חישוב
    time.sleep(1.5)
    
    return [
        {"result": f"{team1} 2 - 1 {team2}", "probability": 35.5},
        {"result": "Draw 1 - 1", "probability": 28.0},
        {"result": f"{team2} 1 - 0 {team1}", "probability": 15.2}
    ]


def mock_update_actual_result(team1: str, team2: str, score1: int, score2: int):
    """
    פונקציה שמדמה שליחת נתונים ל-backend כדי לשמור תוצאת אמת
    שתשפיע על המשך הטורניר.
    """
    time.sleep(1)
    return True

# --- הגדרות עיצוב ---
st.set_page_config(page_title="World Cup Predictor", page_icon="⚽", layout="centered")
st.title("⚽ World Cup Live Predictor")
tab_predict, tab_update = st.tabs(["🔮 חיזוי משחק חדש", "📝 הזנת תוצאת אמת (למידה)"])

# ==========================================
# לשונית 1: חיזוי משחקים
# ==========================================
with tab_predict:
    # --- אזור הקלט ---
    st.markdown("הכנס את שמות הקבוצות באנגלית כדי לקבל את 3 התוצאות הסבירות ביותר למשחק.")
    col_input1, col_input2 = st.columns(2)
    with col_input1:
        team1_input = st.text_input("Home Team (English)", placeholder="e.g., Argentina")
    with col_input2:
        team2_input = st.text_input("Away Team (English)", placeholder="e.g., France")

    # --- הרצת החיזוי ---
    if st.button("🚀 Predict Match"):
        if team1_input and team2_input:
            with st.spinner('מריץ את המודל...'):
                # קריאה לפונקציית ה-Backend (כאן קוראים לפונקציית ה-Mock)
                predictions = main_live_predictor(team1_input, team2_input)
                
            st.success("החיזוי הושלם!")
            st.markdown("### 🏆 Top 3 Predicted Outcomes")
            
            # --- תצוגת 3 התחזיות המובילות ---
            # שימוש בעמודות כדי להציג את התחזיות זו לצד זו בצורה יפה
            col1, col2, col3 = st.columns(3)
            
            # תחזית 1 (הכי סבירה)
            with col1:
                st.info("🥇 סבירות הגבוהה ביותר")
                st.metric(label="תוצאה חזויה", value=str(predictions[0][0])+"-"+str(predictions[0][1]))
                st.progress(int(predictions[0][2]))
                st.caption(f"סיכוי: {predictions[0][2]}%")
                
            # תחזית 2
            with col2:
                st.warning("🥈 אפשרות שנייה")
                st.metric(label="תוצאה חזויה", value=str(predictions[1][0])+"-"+str(predictions[1][1]))
                st.progress(int(predictions[1][2]))
                st.caption(f"סיכוי: {predictions[1][2]}%")
                
            # תחזית 3
            with col3:
                st.error("🥉 אפשרות שלישית")
                st.metric(label="תוצאה חזויה", value=str(predictions[2][0])+"-"+str(predictions[2][1]))
                st.progress(int(predictions[2][2]))
                st.caption(f"סיכוי: {predictions[2][2]}%")
                
        else:
            st.warning("נא להזין את שמות שתי הקבוצות לפני הרצת החיזוי.")


# ==========================================
# לשונית 2: הזנת תוצאות אמת למערכת
# ==========================================
with tab_update:
    st.markdown("הזן תוצאה של משחק שכבר שוחק. הנתונים יישלחו ל-Backend כדי לעדכן את המודל לקראת המשחקים הבאים.")
    
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        team1_update = st.text_input("Home Team", key="upd_t1")
        score1_update = st.number_input("Home Score", min_value=0, step=1, key="upd_s1")
    with col_u2:
        team2_update = st.text_input("Away Team", key="upd_t2")
        score2_update = st.number_input("Away Score", min_value=0, step=1, key="upd_s2")
        
    if st.button("💾 שמור תוצאה ועדכן מודל"):
        if team1_update and team2_update:
            with st.spinner("שולח נתונים ל-Backend..."):
                # כאן אתה קורא לפונקציית ה-Backend שלך שמעדכנת את מצב הטורניר/המודל
                success = update_tournament_memory(str(team1_update), str(team2_update), int(score1_update), int(score2_update))
            
            if success:
                st.success(f"התוצאה ({team1_update} {score1_update} - {score2_update} {team2_update}) נשמרה בהצלחה והמערכת עודכנה!")
        else:
            st.error("נא להזין את שמות שתי הקבוצות לפני השמירה.")