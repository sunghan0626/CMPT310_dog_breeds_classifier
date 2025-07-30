# ✅ 여기 try/except 블록 종료됨

# 새로고침 버튼 누르면 세션 초기화
if st.query_params:
    st.session_state.clear()
    st.experimental_rerun()
# 세션 초기화 (버튼 클릭 시 rerun)