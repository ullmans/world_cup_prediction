import streamlit as st
import time

TEAMS_MAPPING = {
    "אוזבקיסטן": "Uzbekistan",
    "אוסטריה": "Austria",
    "אוסטרליה": "Australia",
    "אורוגוואי": "Uruguay",
    "איראן": "Iran",
    "אלג'יריה": "Algeria",
    "אנגליה": "England",
    "אקוודור": "Ecuador",
    "ארגנטינה": "Argentina",
    "ארצות הברית": "United States",
    "בוסניה והרצגובינה": "Bosnia and Herzegovina",
    "בלגיה": "Belgium",
    "ברזיל": "Brazil",
    "גאנה": "Ghana",
    "גרמניה": "Germany",
    "דרום קוריאה": "South Korea",
    "האיטי": "Haiti",
    "הולנד": "Netherlands",
    "הרפובליקה הדמוקרטית של קונגו": "DR Congo",
    "חוף השנהב": "Ivory Coast",
    "טורקיה": "Turkey",
    "יפן": "Japan",
    "ירדן": "Jordan",
    "כף ורדה": "Cape Verde",
    "מצרים": "Egypt",
    "מקסיקו": "Mexico",
    "מרוקו": "Morocco",
    "נורווגיה": "Norway",
    "ניו זילנד": "New Zealand",
    "סנגל": "Senegal",
    "ספרד": "Spain",
    "סקוטלנד": "Scotland",
    "עיראק": "Iraq",
    "ערב הסעודית": "Saudi Arabia",
    "פורטוגל": "Portugal",
    "פנמה": "Panama",
    "פרגוואי": "Paraguay",
    "צ'כיה": "Czech Republic",
    "צרפת": "France",
    "קולומביה": "Colombia",
    "קורסאו": "Curaçao",
    "קוריאה הדרומית": "South Korea",
    "קטאר": "Qatar",
    "קנדה": "Canada",
    "קרואטיה": "Croatia",
    "שוודיה": "Sweden",
    "שווייץ": "Switzerland",
    "תוניסיה": "Tunisia"
}

# יצירת רשימה ממוינת של הנבחרות בעברית עבור תיבות הבחירה
hebrew_team_names = sorted(list(TEAMS_MAPPING.keys()))

# --- כאן תייבא את הפונקציה האמיתית שלך מה-Backend ---
from live_predictor import main_live_predictor
from update_memory import update_tournament_memory

# --- הגדרות עיצוב ---
st.set_page_config(page_title="World Cup Predictor", page_icon="⚽", layout="centered")
st.title("⚽ World Cup Live Predictor")
tab_predict, tab_update = st.tabs(["🔮 חיזוי משחק חדש", "📝 הזנת תוצאת אמת (למידה)"])

# ==========================================
# לשונית 1: חיזוי משחקים
# ==========================================
with tab_predict:
    st.markdown("בחר את הנבחרות כדי לקבל את 3 התוצאות הסבירות ביותר למשחק.")
    
    col_input1, col_input2 = st.columns(2)
    with col_input1:
        # החלפה ל-selectbox בעברית
        team1_hebrew = st.selectbox("נבחרת מארחת:", hebrew_team_names, key="pred_t1")
    with col_input2:
        # החלפה ל-selectbox בעברית
        team2_hebrew = st.selectbox("נבחרת אורחת:", hebrew_team_names, key="pred_t2")

    # --- הרצת החיזוי ---
    if st.button("🚀 Predict Match"):
        if team1_hebrew == team2_hebrew:
            st.error("שגיאה: אי אפשר לבחור את אותה נבחרת פעמיים!")
        else:
            # תרגום לאנגלית מאחורי הקלעים עבור המודל
            team1_english = TEAMS_MAPPING[team1_hebrew]
            team2_english = TEAMS_MAPPING[team2_hebrew]
            
            with st.spinner('מריץ את המודל...'):
                # קריאה לפונקציית ה-Backend עם השמות באנגלית
                predictions = main_live_predictor(team1_english, team2_english)
                
            st.success("החיזוי הושלם!")
            st.markdown("### 🏆 Top 3 Predicted Outcomes")
            
            # --- תצוגת 3 התחזיות המובילות ---
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


# ==========================================
# לשונית 2: הזנת תוצאות אמת למערכת
# ==========================================
with tab_update:
    st.markdown("הזן תוצאה של משחק שכבר שוחק. הנתונים יישלחו ל-Backend כדי לעדכן את המודל לקראת המשחקים הבאים.")
    
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        # החלפה ל-selectbox בעברית
        team1_upd_hebrew = st.selectbox("נבחרת מארחת:", hebrew_team_names, key="upd_t1")
        score1_update = st.number_input("Home Score", min_value=0, step=1, key="upd_s1")
    with col_u2:
        # החלפה ל-selectbox בעברית
        team2_upd_hebrew = st.selectbox("נבחרת אורחת:", hebrew_team_names, key="upd_t2")
        score2_update = st.number_input("Away Score", min_value=0, step=1, key="upd_s2")
        
    if st.button("💾 שמור תוצאה ועדכן מודל"):
        if team1_upd_hebrew == team2_upd_hebrew:
            st.error("שגיאה: אי אפשר לעדכן משחק של נבחרת נגד עצמה!")
        else:
            # תרגום לאנגלית מאחורי הקלעים
            team1_upd_english = TEAMS_MAPPING[team1_upd_hebrew]
            team2_upd_english = TEAMS_MAPPING[team2_upd_hebrew]
            
            with st.spinner("שולח נתונים ל-Backend..."):
                # קריאה לפונקציית העדכון עם השמות באנגלית
                success = update_tournament_memory(team1_upd_english, team2_upd_english, int(score1_update), int(score2_update))
            
            if success:
                st.success(f"התוצאה ({team1_upd_hebrew} {score1_update} - {score2_update} {team2_upd_hebrew}) נשמרה בהצלחה והמערכת עודכנה!")